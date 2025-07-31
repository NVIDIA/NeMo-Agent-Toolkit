"""A chain for evaluating ReAct style agents.

This chain is used to evaluate ReAct style agents by reasoning about
the sequence of actions taken and their outcomes. It uses a language model
chain (LLMChain) to generate the reasoning and scores.
"""

import re
from typing import TypedDict

from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

_MAX_SCORE = 5


class TrajectoryEval(TypedDict):
    """A named tuple containing the score and reasoning for a trajectory."""

    score: float
    """The score for the trajectory, normalized from 0 to 1."""
    reasoning: str
    """The reasoning for the score."""


class CustomTrajectoryOutputParser(TrajectoryOutputParser):
    """Trajectory output parser."""

    @property
    def _type(self) -> str:
        return "agent_trajectory"

    def parse(self, text: str) -> TrajectoryEval:
        """Parse the output text and extract the score and reasoning.

        Args:
            text (str): The output text to parse.

        Returns:
            TrajectoryEval: A named tuple containing the normalized score and reasoning.

        Raises:
            OutputParserException: If the score is not found in the output text or
                if the LLM's score is not a digit in the range 1-5.
        """
        if "Score:" not in text:
            score_str = "1"
            reasoning = text
        else:
            try:
                reasoning, score_str = text.split("Score: ", maxsplit=1)
            except Exception as e:
                score_str = "1"
                reasoning = text

        reasoning, score_str = reasoning.strip(), score_str.strip()

        # Use regex to extract the score.
        # This will get the number in the string, even if it is a float or more than 10.
        # E.g. "Score: 1" will return 1, "Score: 3.5" will return 3.5, and
        # "Score: 10" will return 10.
        # The score should be an integer digit in the range 1-5.
        _score = re.search(r"(\d+(\.\d+)?)", score_str)
        # If the score is not found or is a float, raise an exception.
        if _score is None or "." in _score.group(1):
            score = "1"
        else:
            score = int(_score.group(1))
        # If the score is not in the range 1-5, raise an exception.
        if not 1 <= score <= _MAX_SCORE:
            msg = f"Score is not a digit in the range 1-5: {text}"
            raise OutputParserException(msg)
        normalized_score = (score - 1) / (_MAX_SCORE - 1)
        return TrajectoryEval(score=normalized_score, reasoning=reasoning)
