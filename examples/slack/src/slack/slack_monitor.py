# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Slack API client module.
Provides functionality for monitoring Slack channels and summarizing conversations.
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

import httpx
from pydantic import Field
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import OpenAI

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class SlackMonitorClientConfig(FunctionBaseConfig, name="slack_monitor_client"):
    """Configuration for Slack API client."""
    slack_token: Optional[str] = Field(default=None)
    nvidia_api_key: Optional[str] = Field(default=None)
    default_channel_id: str = Field(default="C0946F2HVLH")  # Channel with threads
    default_hours: int = Field(default=5)
    timeout: int = Field(default=30)
    users_config_path: str = Field(default="config/users.yml")

@register_function(config_type=SlackMonitorClientConfig)
async def slack_monitor_client(config: SlackMonitorClientConfig, builder: Builder):
    """Register the Slack monitor client function."""

    async def monitor_slack_channel(
        channel_id: Optional[str] = None,
        hours_back: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Monitor a Slack channel and provide conversation summary.
        
        Args:
            channel_id: Slack channel ID (defaults to ngc-sre-blr channel)
            hours_back: Number of hours to look back (defaults to 5 hours)
        """
        try:
            # Use defaults if not provided
            channel_id = channel_id or config.default_channel_id
            hours_back = hours_back or config.default_hours
            
            # Get API tokens
            slack_token = config.slack_token or os.getenv('SLACK_BOT_TOKEN')
            nvidia_api_key = config.nvidia_api_key or os.getenv('NVIDIA_API_KEY')
            
            if not slack_token:
                return {
                    "status": "error",
                    "error": "Slack bot token is required"
                }
            
            if not nvidia_api_key:
                return {
                    "status": "error",
                    "error": "NVIDIA API key is required for summarization"
                }

            # Initialize Slack client
            slack_client = WebClient(token=slack_token)
            
            # Calculate timestamp for N hours ago
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Convert to Unix timestamp
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            
            logger.debug(f"Fetching messages from {start_time} to {end_time}")
            
            # Get channel history
            response = slack_client.conversations_history(
                channel=channel_id,
                oldest=start_ts,
                latest=end_ts,
                limit=1000
            )
            
            messages = response['messages']
            
            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda x: float(x['ts']))
            
            logger.debug(f"Found {len(messages)} messages")
            
            if not messages:
                return {
                    "status": "success",
                    "channel_id": channel_id,
                    "hours_back": hours_back,
                    "message_count": 0,
                    "summary": "No messages found in the specified time range.",
                    "conversation_excerpt": ""
                }
            
            # Create user lookup table
            user_lookup = {}
            unique_users = set()
            
            # Collect unique user IDs from messages
            for msg in messages:
                user_id = msg.get('user')
                if user_id and not msg.get('bot_id'):
                    unique_users.add(user_id)
            
            # Build user lookup table using users.profile.get
            logger.debug(f"Looking up {len(unique_users)} unique users")
            for user_id in unique_users:
                try:
                    profile_response = slack_client.users_profile_get(user=user_id)
                    profile = profile_response.get('profile', {})
                    
                    # Priority: display_name > real_name > first_name + last_name
                    display_name = (
                        profile.get('display_name') or
                        profile.get('real_name') or
                        f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or
                        f"User_{user_id[-4:]}"  # Fallback
                    )
                    
                    user_lookup[user_id] = display_name
                    logger.debug(f"User {user_id} -> {display_name}")
                    
                except SlackApiError as e:
                    logger.warning(f"Could not get profile for user {user_id}: {e.response['error']}")
                    user_lookup[user_id] = f"User_{user_id[-4:]}"
                except Exception as e:
                    logger.warning(f"Error getting profile for user {user_id}: {str(e)}")
                    user_lookup[user_id] = f"User_{user_id[-4:]}"
            
            # Format conversation with real names and thread replies
            conversation = []
            total_length = 0
            max_length = 8000  # Limit conversation length to avoid API timeouts
            processed_threads = set()  # Track processed threads to avoid duplicates
            
            for msg in messages:
                timestamp = datetime.fromtimestamp(float(msg['ts']))
                user_id = msg.get('user', 'Unknown')
                text = msg.get('text', '')
                
                # Skip bot messages and empty messages
                if msg.get('bot_id') or not text.strip():
                    continue
                
                # Get real name from lookup table
                user_name = user_lookup.get(user_id, f"User_{user_id[-4:] if user_id != 'Unknown' else 'Unknown'}")
                
                # Check if this message has thread replies
                thread_ts = msg.get('thread_ts')
                is_thread_parent = thread_ts and thread_ts == msg.get('ts')
                
                if is_thread_parent and thread_ts not in processed_threads:
                    # This is a thread parent, get all replies
                    try:
                        replies_response = slack_client.conversations_replies(
                            channel=channel_id,
                            ts=thread_ts,
                            limit=50
                        )
                        
                        thread_replies = replies_response.get('messages', [])
                        if len(thread_replies) > 1:  # More than just the parent message
                            # Add thread header
                            thread_header = f"[{timestamp.strftime('%H:%M:%S')}] {user_name} (THREAD START): {text}"
                            conversation.append(thread_header)
                            total_length += len(thread_header)
                            
                            # Add thread replies
                            for reply in thread_replies[1:]:  # Skip the first one (parent message)
                                reply_timestamp = datetime.fromtimestamp(float(reply['ts']))
                                reply_user_id = reply.get('user', 'Unknown')
                                reply_text = reply.get('text', '')
                                
                                if reply_text.strip() and not reply.get('bot_id'):
                                    reply_user_name = user_lookup.get(reply_user_id, f"User_{reply_user_id[-4:] if reply_user_id != 'Unknown' else 'Unknown'}")
                                    reply_msg = f"  └─ [{reply_timestamp.strftime('%H:%M:%S')}] {reply_user_name}: {reply_text}"
                                    
                                    if total_length + len(reply_msg) > max_length:
                                        break
                                    
                                    conversation.append(reply_msg)
                                    total_length += len(reply_msg)
                            
                            processed_threads.add(thread_ts)
                            continue  # Skip adding the parent message again
                            
                    except SlackApiError as e:
                        logger.warning(f"Could not get thread replies for {thread_ts}: {e.response['error']}")
                    except Exception as e:
                        logger.warning(f"Error getting thread replies for {thread_ts}: {str(e)}")
                
                # Regular message (not a thread parent or thread reply)
                if not thread_ts or thread_ts == msg.get('ts'):
                    formatted_msg = f"[{timestamp.strftime('%H:%M:%S')}] {user_name}: {text}"
                    
                    # Check if adding this message would exceed the limit
                    if total_length + len(formatted_msg) > max_length:
                        break
                        
                    conversation.append(formatted_msg)
                    total_length += len(formatted_msg)
            
            conversation_text = '\n'.join(conversation)
            
            if not conversation_text.strip():
                return {
                    "status": "success",
                    "channel_id": channel_id,
                    "hours_back": hours_back,
                    "message_count": len(messages),
                    "summary": "No meaningful conversation found to summarize.",
                    "conversation_excerpt": ""
                }
            
            # Initialize OpenAI client for NVIDIA API
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_api_key
            )
            
            # Prepare the prompt for summarization
            prompt = f"""Please provide a concise summary of the following Slack conversation from the channel. Focus on key topics, decisions made, and important information shared:

{conversation_text}

Summary:"""
            
            # Generate summary using NVIDIA API
            try:
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True
                )
                
                # Collect streaming response
                summary_parts = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        summary_parts.append(chunk.choices[0].delta.content)
                
                summary = ''.join(summary_parts)
                
            except Exception as e:
                logger.error(f"Error calling NVIDIA API: {e}")
                summary = f"Error generating summary: {str(e)}"
            
            # Compile response
            result = {
                "status": "success",
                "channel_id": channel_id,
                "hours_back": hours_back,
                "message_count": len(messages),
                "conversation_length": len(conversation_text),
                "summary": summary,
                "conversation_excerpt": conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text,
                "time_range": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
            }
            
            return result

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return {
                "status": "error",
                "error": f"Slack API error: {e.response['error']}"
            }
        except Exception as e:
            logger.exception(f"Error monitoring Slack channel: {e}")
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }

    yield FunctionInfo.from_fn(
        monitor_slack_channel,
        description="Monitors a Slack channel and provides conversation summary for the last N hours. Defaults to ngc-sre-blr channel and 5 hours if not specified."
    ) 