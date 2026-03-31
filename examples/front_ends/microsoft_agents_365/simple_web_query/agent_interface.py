# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from microsoft_agents.hosting.core import Authorization, TurnContext


class AgentInterface(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def process_user_message(
        self,
        message: str,
        auth: Authorization,
        context: TurnContext,
        auth_handler_name: str | None = None,
    ) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass


def check_agent_inheritance(agent_class) -> bool:
    if not issubclass(agent_class, AgentInterface):
        print(f"Agent {agent_class.__name__} does not inherit from AgentInterface")
        return False
    print(f"Agent {agent_class.__name__} properly inherits from AgentInterface")
    return True
