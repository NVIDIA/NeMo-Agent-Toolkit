<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Adding a Custom Evaluator

:::{warning}
**Experimental Feature**: The Evaluation API is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

:::{note}
We recommend reading the [Evaluating NeMo Agent toolkit Workflows](../workflows/evaluate.md) guide before proceeding with this detailed documentation.
:::

NeMo Agent toolkit provides a set of evaluators to run and evaluate NeMo Agent toolkit workflows. In addition to the built-in evaluators, NeMo Agent toolkit provides a plugin system to add custom evaluators.


## Existing Evaluators
You can view the list of existing evaluators by running the following command:
```bash
nat info components -t evaluator
```
`ragas` is an example of an existing evaluator. The `ragas` evaluator is used to evaluate the accuracy of a workflow output.

## Extending NeMo Agent Toolkit with Custom Evaluators
To extend NeMo Agent toolkit with custom evaluators, you need to create an evaluator function and register it with NeMo Agent toolkit by using the `register_evaluator` decorator.

This section provides a step-by-step guide to create and register a custom evaluator with NeMo Agent toolkit. A similarity evaluator is used as an example to demonstrate the process.

### Evaluator Configuration
The evaluator configuration defines the evaluator name and any evaluator-specific parameters. This configuration is paired with a registration function that yields an asynchronous evaluation method.

The following example shows how to define and register a custom evaluator. The code is added to a new `evaluator_register.py` file in the simple example directory for testing purposes.

<!-- path-check-skip-next-line -->
`examples/getting_started/simple_web_query/src/nat_simple_web_query/evaluator_register.py`:
```python
from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class SimilarityEvaluatorConfig(EvaluatorBaseConfig, name="similarity"):
    '''Configuration for custom similarity evaluator'''
    similarity_type: str = Field(description="Similarity type to be computed", default="cosine")


@register_evaluator(config_type=SimilarityEvaluatorConfig)
async def register_similarity_evaluator(config: SimilarityEvaluatorConfig, builder: EvalBuilder):
    '''Register custom evaluator'''
    from .similarity_evaluator import SimilarityEvaluator
    evaluator = SimilarityEvaluator(config.similarity_type, builder.get_max_concurrency())

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Simlaity Evaluator")
```

- The `SimilarityEvaluatorConfig` class defines evaluator-specific settings, including the `similarity_type` parameter.
- The `register_similarity_evaluator` function uses the `@register_evaluator` decorator to register the evaluator with NeMo Agent toolkit.
- The evaluator yields an `EvaluatorInfo` object, which binds the config, evaluation function, and a human-readable description.

The evaluator logic is implemented in the `SimilarityEvaluator` class described in the [Similarity Evaluator](#similarity-evaluator-custom-evaluator-example) section.

### Importing for registration
To ensure the evaluator is registered at runtime, import the evaluator function in the example project's register.py file — even if the function is not called directly.
`examples/getting_started/simple_web_query/src/nat_simple_web_query/register.py`:
```python
from .evaluator_register import register_similarity_evaluator  # pylint: disable=unused-import
```

### Understanding `EvalInputItem` and `EvalOutputItem`
Custom evaluators in NeMo Agent toolkit implement an asynchronous `evaluate_item` method, which receives an `EvalInputItem` as input and returns an `EvalOutputItem` as output.

**EvalInputItem**

An `EvalInputItem` encapsulates all relevant information for evaluating a single data point. It includes the following fields:
- `id`: A unique identifier for the item, taken from the dataset. It can be a string or integer.
- `input_obj`: The question or input object from the dataset entry (typically mapped from the `question` field). This can be any JSON-serializable object.
- `expected_output_obj`: The reference or ground truth answer from the dataset (typically mapped from the `answer` field). Also JSON-serializable.
- `output_obj`: The generated output from the workflow being evaluated.
- `trajectory`: A list of intermediate steps returned by the workflow. Each step is an IntermediateStep object.
- `expected_trajectory`: A list of expected intermediate steps (if defined in the dataset), also represented as IntermediateStep objects.
- `full_dataset_entry`: The entire dataset entry as a dictionary. This field is populated only if eval.general.dataset.pass_full_entry is set to true in the config. It is useful for accessing additional fields (e.g., metadata, tags, references) that are not part of the standard workflow inputs.

**EvalOutputItem**

An `EvalOutputItem` represents the result of evaluating a single item. It includes:
- `id`: The identifier of the evaluated input item (copied from `EvalInputItem.id`).
- `score`: The computed score for this item. This is typically a floating-point number used for average score computation across the dataset. However, it can be any JSON-serializable object. If the score is not numeric, the average score in EvalOutput will be omitted.
- `reasoning`: An explanation or trace of how the score was computed. This can contain any serializable structure (e.g., dictionary, string, list), and is often shown in logs or UI output for `interpretability`.

### Similarity Evaluator (Custom Evaluator Example)
NeMo Agent toolkit provides a convenient `BaseEvaluator` class that simplifies writing custom evaluators. It handles common tasks such as:
- Asynchronous evaluation of input items
- Concurrency control
- Progress bar display using `tqdm`

To create a custom evaluator, subclass `BaseEvaluator` and implement the `evaluate_item` method. This method is responsible for computing the evaluation result for a single `EvalInputItem`, and should return an `EvalOutputItem`.

The following example defines a SimilarityEvaluator that computes the cosine similarity between a generated output and an expected reference using TF-IDF embeddings. This is useful for evaluating natural language generation tasks such as Q&A, summarization, or text rewriting.

We define the evaluator in the `similarity_evaluator.py` file:
<!-- path-check-skip-next-line -->
`examples/getting_started/simple_web_query/src/nat_simple_web_query/similarity_evaluator.py`:
```python
from typing import override
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem, EvalOutputItem

class SimilarityEvaluator(BaseEvaluator):
    def __init__(self, similarity_type: str = "cosine", max_concurrency: int = 4):
        super().__init__(max_concurrency, tqdm_desc=f"Evaluating {similarity_type} similarity")
        self.similarity_type = similarity_type
        self.vectorizer = TfidfVectorizer()

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        question = item.input_obj
        answer = item.expected_output_obj
        generated_answer = item.output_obj

        tfidf_matrix = self.vectorizer.fit_transform([answer, generated_answer])
        similarity_score = round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0], 2)

        # The reasoning field is flexible and can contain any serializable dictionary
        reasoning = {
            "question": question,
            "answer": answer,
            "generated_answer": generated_answer,
            "similarity_type": self.similarity_type,
        }

        return EvalOutputItem(id=item.id, score=similarity_score, reasoning=reasoning)
```

### Display all evaluators
To display all evaluators, run the following command:
```bash
nat info components -t evaluator
```
This will now display the custom evaluator `similarity` in the list of evaluators.

### Evaluation configuration
Add the evaluator to the workflow configuration file in the `eval.evaluators` section. The following is an example of the similarity evaluator configuration:
`examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml`:
```yaml
eval:
  evaluators:
    similarity_eval:
      _type: similarity
      similarity_type: cosine
```
The `_type` field specifies the evaluator name. The keyword `similarity_eval` can be set to any string. It is used as a prefix to the evaluator output file name.

### Evaluating the workflow
Run and evaluate the workflow using the following command:
```bash
nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
```

### Evaluation results
The evaluation results are stored in the output directory specified in the workflow configuration file.
`examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml`:
```yaml
eval:
  general:
    output_dir: ./.tmp/nat/examples/getting_started/simple_web_query/
```

The results of each evaluator is stored in a separate file with name `<keyword>_eval_output.json`. The following is an example of the similarity evaluator output file:
`examples/getting_started/simple_web_query/.tmp/nat/examples/getting_started/simple_web_query/similarity_eval_output.json`:
```json
{
  "average_score": 0.63,
  "eval_output_items": [
    {
      "id": 1,
      "score": 0.56,
      "reasoning": {
        "question": "What is langsmith",
        "answer": "LangSmith is a platform for LLM application development, monitoring, and testing",
        "generated_answer": "LangSmith is a platform for LLM application development, monitoring, and testing. It supports various workflows throughout the application development lifecycle, including automations, threads, annotating traces, adding runs to a dataset, prototyping, and debugging.",
        "similarity_type": "cosine"
      }
    },
    {
      "id": 2,
      "score": 0.78,
      "reasoning": {
        "question": "How do I proptotype with langsmith",
        "answer": "To prototype with LangSmith, you can use its tracing feature to quickly understand how the model is performing and debug where it is failing. LangSmith provides clear visibility and debugging information at each step of an LLM sequence, making it easier to identify and root-cause issues.",
        "generated_answer": "To prototype with LangSmith, you can use its tracing feature to quickly understand how the model is performing and debug where it is failing. LangSmith provides clear visibility and debugging information at each step of an LLM sequence, making it easier to identify and root-cause issues. Additionally, LangSmith supports automations, threads, and annotating traces, which can be helpful for processing traces at production scale, tracking the performance of multi-turn applications, and refining and improving the application's performance.",
        "similarity_type": "cosine"
      }
    },
  ]
}
```
The contents of the file have been `snipped` for brevity.

# Summary
This guide provides a step-by-step process to create and register a custom evaluator with NeMo Agent toolkit. The similarity evaluator is used as an example to demonstrate the process. The evaluator configuration, evaluator function, and evaluation results are explained in detail.
