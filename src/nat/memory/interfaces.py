# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable

from .models import MemoryItem


class MemoryEditor(ABC):
    """
    Abstract interface for editing and
    retrieving memory items.

    A MemoryEditor is responsible for adding, searching, and
    removing MemoryItems.

    Implementations may integrate with
    vector stores or other indexing backends.
    """

    @abstractmethod
    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert multiple MemoryItems into the memory.

        Args:
            items (list[MemoryItem]): The items to be added.
        """
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query.
        Relevance criteria depend on implementation.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            kwargs (dict): Keyword arguments to pass to the search method.

        Returns:
            list[MemoryItem]: The most relevant MemoryItems.
        """
        raise NotImplementedError

    @abstractmethod
    async def remove_items(self, **kwargs) -> None:
        """
        Remove items. Additional parameters
        needed for deletion can be specified in keyword arguments.

        Args:
            kwargs (dict): Keyword arguments to pass to the remove-items method.
        """
        raise NotImplementedError


class MemoryIOBase(ABC):
    """
    Base abstract class for I/O operations
    on memory, providing a common interface for

    MemoryReader and MemoryWriter to interact
    with a MemoryEditor.

    Concrete subclasses should hold a
    reference to a MemoryEditor instance.
    """

    def __init__(self, editor: MemoryEditor) -> None:
        self._editor = editor


class MemoryReader(MemoryIOBase):
    """
    Responsible for retrieving MemoryItems
    from the MemoryEditor based on context or queries.
    """

    @abstractmethod
    async def retrieve(self, context: str, top_k: int = 5) -> list[MemoryItem]:
        """
        Retrieve a subset of
        MemoryItems relevant to the provided context.

        Args:
            context (str): A string representing
            the current user context or query.
            top_k (int): Maximum number of items to return.

        Returns:
            list[MemoryItem]: Relevant MemoryItems.
        """
        raise NotImplementedError


class MemoryWriter(MemoryIOBase):
    """
    Responsible for converting new observations
    (textual inputs) into MemoryItems andstoring
    them via the MemoryEditor.
    """

    @abstractmethod
    async def write(self, observation: str, context: str | None = None) -> list[MemoryItem]:
        """
        Process the given observation and store the resulting MemoryItems.

        Args:
            observation (str): The new textual input to record.
            context (Optional[str]): Additional
            context that might influence how the observation is stored.

        Returns:
            list[MemoryItem]: The newly created MemoryItems.
        """
        raise NotImplementedError


class MemoryManager(ABC):
    """
    Manages the lifecycle of the stored
    memory by applying policies such as summarization,
    reflection, forgetting, and mergingn
    to ensure long-term coherence and relevance.
    """

    @abstractmethod
    async def summarize(self) -> None:
        """
        Summarize long or numerous MemoryItems into a more compact form.
        This may remove the original items and store a new summary item.
        """
        raise NotImplementedError

    @abstractmethod
    async def reflect(self) -> None:
        """
        Generate higher-level insights or
        abstractions from existing MemoryItems.
        This may call out to an LLM or other
        logic to produce conceptual memory.
        """
        raise NotImplementedError

    @abstractmethod
    async def forget(self, criteria: Callable[[MemoryItem], bool]) -> None:
        """
        Remove MemoryItems that are no
        longer relevant or have low importance.

        Args:
            criteria (Callable[[MemoryItem], bool]): A function that
            returns True for items to forget.
        """
        raise NotImplementedError

    @abstractmethod
    async def merge(self, criteria: Callable[[MemoryItem, MemoryItem], bool]) -> None:
        """
        Merge similar or redundant MemoryItems
        into a smaller set of more concise items.

        Args:
            criteria (Callable[[MemoryItem, MemoryItem], bool]): A function
            that determines which items can be merged.
        """
        raise NotImplementedError
