import asyncio
from typing import Any

from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem


class MemMachineEditor(MemoryEditor):
    """
    Wrapper class that implements NAT interfaces for MemMachine Integrations.
    Uses the MemMachine Python SDK (MemMachineClient) as documented at:
    https://github.com/MemMachine/MemMachine/blob/main/docs/examples/python.mdx
    
    Supports both episodic and semantic memory through the unified SDK interface.

    User needs to add MemMachine SDK ids as metadata to the MemoryItem:
    - session_id
    - agent_id
    - group_id
    - project_id
    - org_id

    Group ID is optional. If not provided, the memory will be added to the 'default' group.
    """

    def __init__(self, memmachine_instance: Any):
        """
        Initialize class with MemMachine instance.

        Args:
            memmachine_instance: Preinstantiated MemMachineClient or Project object
            from the MemMachine Python SDK. If a MemMachineClient is provided,
            projects will be created/retrieved as needed. If a Project is provided,
            it will be used directly.
        """
        self._memmachine = memmachine_instance
        # Check if it's a client or project
        self._is_client = hasattr(memmachine_instance, 'create_project')
        self._is_project = hasattr(memmachine_instance, 'memory') and not self._is_client

    def _get_memory_instance(
        self,
        user_id: str,
        session_id: str,
        agent_id: str,
        group_id: str = "default",
        project_id: str | None = None,
        org_id: str | None = None
    ) -> Any:
        """
        Get or create a memory instance for the given context using the MemMachine SDK.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            group_id: Group identifier (default: "default")
            project_id: Optional project identifier (default: "default-project")
            org_id: Optional organization identifier (default: "default-org")
            
        Returns:
            Memory instance from MemMachine SDK
        """
        # Use defaults if not provided
        if not org_id:
            org_id = "default-org"
        if not project_id:
            project_id = "default-project"
        
        # If we have a client, get or create the project first
        if self._is_client:
            # Use create_project which returns existing project if it exists
            project = self._memmachine.create_project(
                org_id=org_id,
                project_id=project_id,
                description=f"Project for {user_id}"
            )
        elif self._is_project:
            # Use the project directly
            project = self._memmachine
        else:
            # Fallback: assume it's already a memory instance or try to use it directly
            return self._memmachine
        
        # Create memory instance from project
        return project.memory(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            group_id=group_id
        )

    async def add_items(self, items: list[MemoryItem]) -> None:
        """
        Insert Multiple MemoryItems into the memory using the MemMachine SDK.
        Each MemoryItem is translated and uploaded through the MemMachine API.
        """
        # Run synchronous operations in thread pool to make them async
        tasks = []

        for memory_item in items:
            # Make a copy of metadata to avoid modifying the original
            item_meta = memory_item.metadata.copy() if memory_item.metadata else {}
            conversation = memory_item.conversation
            user_id = memory_item.user_id
            tags = memory_item.tags
            memory_text = memory_item.memory

            # Extract session_id, agent_id, group_id, project_id, and org_id from metadata if present
            session_id = item_meta.pop("session_id", "default_session")
            agent_id = item_meta.pop("agent_id", "default_agent")
            group_id = item_meta.pop("group_id", "default")
            project_id = item_meta.pop("project_id", None)
            org_id = item_meta.pop("org_id", None)
            
            # Extract use_semantic_memory once per item (not per message)
            # Store it before popping so we can use it in both conversation and memory_text paths
            # Check if key exists to distinguish between "not set" (None) and "explicitly False"
            use_semantic = item_meta.get("use_semantic_memory") if "use_semantic_memory" in item_meta else None
            # Remove it from metadata so it doesn't get passed to MemMachine
            item_meta.pop("use_semantic_memory", None)

            # Get memory instance using MemMachine SDK
            memory = self._get_memory_instance(
                user_id, session_id, agent_id, group_id, project_id, org_id
            )
            
            # Prepare content for MemMachine
            # If we have a conversation, add each message separately
            # Otherwise, use memory_text or skip if no content
            if conversation:
                # Add each message in the conversation with its role
                for msg in conversation:
                    msg_role = msg.get('role', 'user')
                    msg_content = msg.get('content', '')
                    
                    if not msg_content:
                        continue
                    
                    # Determine episode_type based on metadata
                    # For conversations: if use_semantic_memory is True, use semantic; otherwise episodic (default)
                    episode_type = "semantic" if use_semantic is True else "episodic"
                    
                    # Add tags to metadata if present
                    # MemMachine SDK expects tags as a string, not a list
                    metadata = item_meta.copy() if item_meta else {}
                    if tags:
                        # Convert list to comma-separated string
                        metadata["tags"] = ", ".join(tags) if isinstance(tags, list) else str(tags)
                    
                    # Capture variables in closure to avoid late binding issues
                    def add_memory(content=msg_content, role=msg_role, ep_type=episode_type, meta=metadata):
                        # Use MemMachine SDK add() method
                        # API: memory.add(content, role="user", metadata={}, episode_type="text")
                        memory.add(
                            content=content,
                            role=role,
                            metadata=meta if meta else None,
                            episode_type=ep_type
                        )
                    
                    task = asyncio.to_thread(add_memory)
                    tasks.append(task)
            elif memory_text:
                # Add as a single memory item
                # For memory_text (non-conversation), default to semantic (facts/preferences)
                # unless explicitly set to episodic via use_semantic_memory=False
                # use_semantic was already extracted above
                # If use_semantic is True or None (not set), use semantic; if False, use episodic
                episode_type = "episodic" if use_semantic is False else "semantic"
                
                # Add tags to metadata if present
                # MemMachine SDK expects tags as a string, not a list
                metadata = item_meta.copy() if item_meta else {}
                if tags:
                    # Convert list to comma-separated string
                    metadata["tags"] = ", ".join(tags) if isinstance(tags, list) else str(tags)
                
                def add_memory():
                    # Use MemMachine SDK add() method
                    memory.add(
                        content=memory_text,
                        role="user",
                        metadata=metadata if metadata else None,
                        episode_type=episode_type
                    )
                
                task = asyncio.to_thread(add_memory)
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        Retrieve items relevant to the given query using the MemMachine SDK.

        Args:
            query (str): The query string to match.
            top_k (int): Maximum number of items to return.
            kwargs: Other keyword arguments for search.
                Must include 'user_id'. May include 'session_id', 'agent_id', 'project_id', 'org_id'.

        Returns:
            list[MemoryItem]: The most relevant MemoryItems for the given query.
        """
        user_id = kwargs.pop("user_id")  # Ensure user ID is in keyword arguments
        session_id = kwargs.pop("session_id", "default_session")
        agent_id = kwargs.pop("agent_id", "default_agent")
        group_id = kwargs.pop("group_id", "default")
        project_id = kwargs.pop("project_id", None)
        org_id = kwargs.pop("org_id", None)

        # Get memory instance using MemMachine SDK
        memory = self._get_memory_instance(
            user_id, session_id, agent_id, group_id, project_id, org_id
        )

        # Perform search using MemMachine SDK
        def perform_search():
            # MemMachine SDK search() method signature:
            # search(query, limit=None, filter_dict=None, timeout=None)
            # Returns dict with 'episodic_memory', 'semantic_memory', 'episode_summary'
            return memory.search(query=query, limit=top_k)

        search_results = await asyncio.to_thread(perform_search)

        # Construct MemoryItem instances from search results
        memories = []

        if not search_results:
            return memories

        # MemMachine SDK returns a dict with episodic_memory and semantic_memory
        # episodic_memory is a list of episodes
        # semantic_memory is a list of semantic features
        episodic_results = search_results.get("episodic_memory", [])
        semantic_results = search_results.get("semantic_memory", [])
        
        # Process episodic memories - group by conversation if possible
        # Episodes from the same conversation should be grouped together
        episodic_by_conversation = {}  # Key: conversation identifier, Value: list of episodes
        standalone_episodic = []  # Episodes that don't belong to a conversation
        
        for episode in episodic_results:
            if isinstance(episode, dict):
                # Check if episode has role information (producer_role field)
                episode_role = episode.get("producer_role") or episode.get("role")
                episode_metadata = episode.get("metadata", {})
                
                # Group episodes by test_id or similar identifier in metadata
                # This groups episodes from the same conversation
                conv_id = episode_metadata.get("test_id") or episode_metadata.get("conversation_id")
                
                if episode_role and conv_id:
                    # This is part of a conversation - group it
                    if conv_id not in episodic_by_conversation:
                        episodic_by_conversation[conv_id] = []
                    episodic_by_conversation[conv_id].append(episode)
                else:
                    # Standalone episode
                    standalone_episodic.append(episode)
            else:
                standalone_episodic.append(episode)
        
        # Reconstruct conversations from grouped episodes
        for conv_key, conv_episodes in episodic_by_conversation.items():
            # Sort episodes by created_at timestamp if available
            try:
                conv_episodes.sort(key=lambda e: e.get("created_at") or e.get("timestamp") or "")
            except:
                pass
            
            # Extract conversation messages
            conversation_messages = []
            memory_text = None
            item_meta = {}
            tags = []
            
            for episode in conv_episodes:
                # Get role from producer_role field
                episode_role = episode.get("producer_role") or episode.get("role") or "user"
                episode_content = episode.get("content") or episode.get("text") or ""
                
                if episode_content:
                    conversation_messages.append({
                        "role": episode_role,
                        "content": episode_content
                    })
                    
                    # Use first episode's metadata and tags
                    if not item_meta:
                        item_meta = episode.get("metadata", {}).copy()
            
            # Extract tags from metadata
            if "tags" in item_meta:
                tags_raw = item_meta.pop("tags", [])
                if isinstance(tags_raw, str):
                    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                elif isinstance(tags_raw, list):
                    tags = tags_raw
                else:
                    tags = []
            
            # Create memory text from conversation (use first message or combine)
            if conversation_messages:
                memory_text = conversation_messages[0].get("content", "")
                # Only set conversation if we have multiple messages
                memories.append(
                    MemoryItem(
                        conversation=conversation_messages if len(conversation_messages) > 1 else None,
                        user_id=user_id,
                        memory=memory_text,
                        tags=tags,
                        metadata=item_meta
                    )
                )
        
        # Process standalone episodic memories
        for result in standalone_episodic:
            memory_text = None
            conversation = None
            item_meta = {}
            tags = []
            
            if isinstance(result, dict):
                memory_text = result.get("content") or result.get("text")
                item_meta = result.get("metadata", {})
                
                # Extract tags
                if "tags" in item_meta:
                    tags_raw = item_meta.pop("tags", [])
                    if isinstance(tags_raw, str):
                        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                    elif isinstance(tags_raw, list):
                        tags = tags_raw
                    else:
                        tags = []
            elif hasattr(result, 'content'):
                memory_text = result.content
                if hasattr(result, 'metadata'):
                    item_meta = result.metadata or {}
                if hasattr(result, 'tags'):
                    tags = result.tags or []
            else:
                memory_text = str(result)

            if memory_text:
                memories.append(
                    MemoryItem(
                        conversation=conversation,
                        user_id=user_id,
                        memory=memory_text,
                        tags=tags,
                        metadata=item_meta
                    )
                )
        
        # Process semantic memories
        for result in semantic_results:
            memory_text = None
            item_meta = {}
            tags = []
            
            if isinstance(result, dict):
                memory_text = result.get("feature") or result.get("content") or result.get("text")
                item_meta = result.get("metadata", {})
                
                # Extract tags
                if "tags" in item_meta:
                    tags_raw = item_meta.pop("tags", [])
                    if isinstance(tags_raw, str):
                        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                    elif isinstance(tags_raw, list):
                        tags = tags_raw
                    else:
                        tags = []
            elif hasattr(result, 'feature'):
                memory_text = result.feature
                if hasattr(result, 'metadata'):
                    item_meta = result.metadata or {}
            else:
                memory_text = str(result)

            if memory_text:
                memories.append(
                    MemoryItem(
                        conversation=None,
                        user_id=user_id,
                        memory=memory_text,
                        tags=tags,
                        metadata=item_meta
                    )
                )
        
        # Limit to top_k
        return memories[:top_k]

    async def remove_items(self, **kwargs) -> None:
        """
        Remove items using the MemMachine SDK. Additional parameters
        needed for deletion can be specified in keyword arguments.

        Args:
            kwargs (dict): Keyword arguments to pass to the remove-items method.
                Should include either 'memory_id' (episodic_id or semantic_id) or 'user_id'.
                May include 'session_id', 'agent_id', 'group_id', 'project_id', 'org_id'.
                For memory_id deletion, may include 'memory_type' ('episodic' or 'semantic').
        """
        if "memory_id" in kwargs:
            memory_id = kwargs.pop("memory_id")
            memory_type = kwargs.pop("memory_type", "episodic")  # Default to episodic
            user_id = kwargs.pop("user_id", None)
            session_id = kwargs.pop("session_id", "default_session")
            agent_id = kwargs.pop("agent_id", "default_agent")
            group_id = kwargs.pop("group_id", "default")
            project_id = kwargs.pop("project_id", None)
            org_id = kwargs.pop("org_id", None)

            if not user_id:
                raise ValueError(
                    "user_id is required when deleting by memory_id. "
                    "A memory instance is needed to perform deletion, which requires user_id."
                )

            def delete_memory():
                memory = self._get_memory_instance(
                    user_id, session_id, agent_id, group_id, project_id, org_id
                )
                # Use MemMachine SDK to delete specific memory
                # API: memory.delete_episodic(episodic_id) or memory.delete_semantic(semantic_id)
                if memory_type.lower() == "semantic":
                    memory.delete_semantic(semantic_id=memory_id)
                else:
                    memory.delete_episodic(episodic_id=memory_id)

            await asyncio.to_thread(delete_memory)

        elif "user_id" in kwargs:
            user_id = kwargs.pop("user_id")
            session_id = kwargs.pop("session_id", "default_session")
            agent_id = kwargs.pop("agent_id", "default_agent")
            group_id = kwargs.pop("group_id", "default")
            project_id = kwargs.pop("project_id", None)
            org_id = kwargs.pop("org_id", None)
            delete_semantic = kwargs.pop("delete_semantic_memory", False)

            # Note: MemMachine SDK doesn't have a delete_all method
            # We would need to search for all memories and delete them individually
            # For now, we'll raise a NotImplementedError with guidance
            raise NotImplementedError(
                "Bulk deletion by user_id is not directly supported by MemMachine SDK. "
                "To delete all memories for a user, you would need to: "
                "1. Search for all memories with that user_id "
                "2. Extract memory IDs from results "
                "3. Delete each memory individually using delete_episodic() or delete_semantic(). "
                "Alternatively, delete specific memories using memory_id parameter."
            )