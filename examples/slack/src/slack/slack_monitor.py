# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Slack API client module.
Provides functionality for monitoring Slack channels and summarizing conversations.
"""

import logging
import os
import yaml
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

import httpx
from pydantic import Field

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



def calculate_time_based_hours(question: str, current_time: datetime) -> int:
    """
    Calculate hours_back based on time keywords in the question.
    
    Args:
        question: The question being asked
        current_time: Current datetime object
    
    Returns:
        hours_back: Number of hours to look back
    """
    question_lower = question.lower()
    
    # Add debug logging
    logger.debug(f"Calculating hours for question: '{question}'")

    # PRIORITIZE 'today' and 'tonight' first
    if "today" in question_lower or "tonight" in question_lower:
        logger.debug("Found 'today' reference (priority match)")
        calculated_hours = 24
        logger.debug(f"Calculated hours for 'today': {calculated_hours}")
        return calculated_hours
    
    # Day of week mapping
    day_mapping = {
        'monday': 0, 'mon': 0,
        'tuesday': 1, 'tue': 1, 'tues': 1,
        'wednesday': 2, 'wed': 2,
        'thursday': 3, 'thu': 3, 'thurs': 3,
        'friday': 4, 'fri': 4,
        'saturday': 5, 'sat': 5,
        'sunday': 6, 'sun': 6
    }
    
    # Check for specific day references (e.g., "on Monday", "on Tuesday")
    for day_name, day_num in day_mapping.items():
        if f"on {day_name}" in question_lower or f" {day_name}" in question_lower:
            logger.debug(f"Found day reference: {day_name}")
            # Calculate hours from that day of current week to current time
            current_weekday = current_time.weekday()  # Monday=0, Sunday=6 (ISO standard)
            target_weekday = day_num  # Already in Monday=0 format
            
            # Calculate days difference (Monday=0, Sunday=6)
            days_diff = current_weekday - target_weekday
            if days_diff < 0:
                days_diff += 7  # Wrap around to next week
            
            # Calculate hours from that day to current time
            target_date = current_time - timedelta(days=days_diff)
            target_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            hours_diff = (current_time - target_date).total_seconds() / 3600
            calculated_hours = max(1, int(hours_diff))
            logger.debug(f"Calculated hours for {day_name}: {calculated_hours}")
            return calculated_hours
    
    # Check for week references
    if "this week" in question_lower:
        logger.debug("Found 'this week' reference")
        # Calculate hours from Monday of current week to current time (ISO standard)
        current_weekday = current_time.weekday()  # Monday=0, Sunday=6
        
        # Calculate days since Monday (start of week)
        days_since_monday = current_weekday  # Monday=0, so days since Monday = current_weekday
        
        hours_since_monday = (days_since_monday * 24) + current_time.hour + (current_time.minute / 60)
        calculated_hours = max(1, int(hours_since_monday))
        logger.debug(f"Calculated hours for 'this week': {calculated_hours}")
        return calculated_hours
    
    elif "last week" in question_lower:
        logger.debug("Found 'last week' reference")
        # Calculate hours for last week (previous Monday to Sunday)
        current_weekday = current_time.weekday()  # Monday=0, Sunday=6
        
        # Calculate days to go back to last Monday
        days_to_last_monday = current_weekday + 7
        
        # Calculate hours to last Monday
        hours_to_last_monday = days_to_last_monday * 24
        
        # For "last week", we want to search from last Monday to last Sunday
        # So we go back to last Monday and add 7 days (168 hours) for the full week
        calculated_hours = hours_to_last_monday + 168  # Last Monday + 7 days
        
        logger.debug(f"Calculated hours for 'last week': {calculated_hours}")
        return calculated_hours
    
    # Check for other time references
    elif "yesterday" in question_lower or "yday" in question_lower or "last night" in question_lower:
        logger.debug("Found 'yesterday' reference")
        calculated_hours = 48
        logger.debug(f"Calculated hours for 'yesterday': {calculated_hours}")
        return calculated_hours
    elif "this month" in question_lower:
        logger.debug("Found 'this month' reference")
        # Calculate hours from 1st of current month to current time
        first_of_month = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        hours_diff = (current_time - first_of_month).total_seconds() / 3600
        calculated_hours = max(1, int(hours_diff))
        logger.debug(f"Calculated hours for 'this month': {calculated_hours}")
        return calculated_hours
    elif "last month" in question_lower:
        logger.debug("Found 'last month' reference")
        calculated_hours = 1440  # 60 days (approximate)
        logger.debug(f"Calculated hours for 'last month': {calculated_hours}")
        return calculated_hours
    elif "recently" in question_lower:
        logger.debug("Found 'recently' reference")
        calculated_hours = 72  # 3 days
        logger.debug(f"Calculated hours for 'recently': {calculated_hours}")
        return calculated_hours
    elif "lately" in question_lower:
        logger.debug("Found 'lately' reference")
        calculated_hours = 120  # 5 days
        logger.debug(f"Calculated hours for 'lately': {calculated_hours}")
        return calculated_hours
    
    # Health/wellness keywords - extend search
    health_keywords = [
        'sick', 'not feeling well', 'unwell', 'ill', 'health issue', 'medical',
        'doctor', 'hospital', 'recovery', 'symptoms', 'fever', 'cold', 'flu',
        'covid', 'positive', 'negative', 'test', 'quarantine', 'isolation'
    ]
    
    if any(keyword in question_lower for keyword in health_keywords):
        logger.debug("Found health-related keywords")
        calculated_hours = 168  # 1 week for health-related questions
        logger.debug(f"Calculated hours for health keywords: {calculated_hours}")
        return calculated_hours
    
    # Default for answer operations
    logger.debug("No time keywords found, using default 24 hours")
    return 24

def get_current_time() -> datetime:
    """Get current datetime."""
    return datetime.now()

class SlackMonitorClientConfig(FunctionBaseConfig, name="slack_monitor_client"):
    """Configuration for Slack API client."""
    slack_token: Optional[str] = Field(default=None)
    nvidia_api_key: Optional[str] = Field(default=None)
    default_channel_id: str = Field(default="C029H7U1TEV")  
    default_hours: int = Field(default=24)
    timeout: int = Field(default=30)
    users_config_path: str = Field(default="config/users.yml")

@register_function(config_type=SlackMonitorClientConfig)
async def slack_monitor_client(config: SlackMonitorClientConfig, builder: Builder):
    """Register the Slack monitor client function."""

    async def slack_operations(
        operation: str,
        channel_id: Optional[str] = None,
        hours_back: Optional[int] = None,
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform Slack operations including monitoring and question answering.
        
        Args:
            operation: Type of operation ("monitor" or "answer")
            channel_id: Slack channel ID (defaults to C029H7U1TEV)
            hours_back: Number of hours to look back (auto-calculated for answer operations)
            question: The specific question to answer (required for "answer" operation)
        """
        try:
            # Validate operation type
            if operation not in ["monitor", "answer"]:
                return {
                    "status": "error",
                    "error": "Operation must be either 'monitor' or 'answer'"
                }
            
            # Validate question for answer operation
            if operation == "answer" and not question:
                return {
                    "status": "error",
                    "error": "Question is required for answer operation"
                }
            
            # Use defaults if not provided
            channel_id = channel_id or config.default_channel_id
            
            # Auto-calculate hours_back for answer operations based on question content
            if operation == "answer" and question:
                current_time = get_current_time()
                logger.debug(f"Current time: {current_time}")
                calculated_hours = calculate_time_based_hours(question, current_time)
                hours_back = hours_back or calculated_hours
                logger.debug(f"Calculated hours_back: {calculated_hours} for question: {question}")
            else:
                # For monitor operations, use hours_back or default
                hours_back = hours_back or config.default_hours
            
            # Calculate timestamps from hours_back
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            logger.debug(f"Using hours_back: {hours_back}, calculated timestamps: {start_time} to {end_time}")
            
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
                    "error": "NVIDIA API key is required for summarization and question answering"
                }

            # Convert to Unix timestamp
            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()
            
            logger.debug(f"Fetching messages from {start_time} to {end_time} (hours_back: {hours_back})")
            if operation == "answer":
                logger.debug(f"Question: {question}")
                logger.debug(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Headers for Slack API
            headers = {
                "Authorization": f"Bearer {slack_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                # Get channel history
                history_url = "https://slack.com/api/conversations.history"
                params = {
                    "channel": channel_id,
                    "oldest": start_ts,
                    "latest": end_ts,
                    "limit": 1000
                }
                
                response = await client.get(history_url, headers=headers, params=params)
                
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"Failed to fetch Slack messages: {response.status_code} - {response.text}"
                    }
                
                data = response.json()
                if not data.get("ok"):
                    return {
                        "status": "error",
                        "error": f"Slack API error: {data.get('error', 'Unknown error')}"
                    }
                
                messages = data.get("messages", [])
                
                # Sort messages by timestamp (oldest first)
                messages.sort(key=lambda x: float(x['ts']))
                
                logger.debug(f"Found {len(messages)} messages")
                
                if not messages:
                    result = {
                        "status": "success",
                        "operation": operation,
                        "channel_id": channel_id,
                        "hours_back": hours_back,
                        "message_count": 0,
                        "conversation_excerpt": "",
                        "time_range": {
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "calculated_hours": hours_back,
                            "time_range_description": f"From {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    }
                    
                    if operation == "monitor":
                        result["summary"] = "No messages found in the specified time range."
                    else:
                        result["question"] = question
                        result["answer"] = "No messages found in the specified time range to answer this question."
                    
                    return result
                
                # Create user lookup table
                user_lookup = {}
                unique_users = set()
                
                # Collect unique user IDs from messages
                for msg in messages:
                    user_id = msg.get('user')
                    if user_id and not msg.get('bot_id'):
                        unique_users.add(user_id)
                
                # Build user lookup table
                logger.debug(f"Looking up {len(unique_users)} unique users")
                for user_id in unique_users:
                    try:
                        profile_url = "https://slack.com/api/users.profile.get"
                        profile_params = {"user": user_id}
                        
                        profile_response = await client.get(profile_url, headers=headers, params=profile_params)
                        
                        if profile_response.status_code == 200:
                            profile_data = profile_response.json()
                            if profile_data.get("ok"):
                                profile = profile_data.get("profile", {})
                                
                                # Priority: display_name > real_name > first_name + last_name
                                display_name = (
                                    profile.get('display_name') or
                                    profile.get('real_name') or
                                    f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip() or
                                    f"User_{user_id[-4:]}"  # Fallback
                                )
                                
                                user_lookup[user_id] = display_name
                                logger.debug(f"User {user_id} -> {display_name}")
                            else:
                                user_lookup[user_id] = f"User_{user_id[-4:]}"
                        else:
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
                            replies_url = "https://slack.com/api/conversations.replies"
                            replies_params = {
                                "channel": channel_id,
                                "ts": thread_ts,
                                "limit": 50
                            }
                            
                            replies_response = await client.get(replies_url, headers=headers, params=replies_params)
                            
                            if replies_response.status_code == 200:
                                replies_data = replies_response.json()
                                if replies_data.get("ok"):
                                    thread_replies = replies_data.get("messages", [])
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
                    result = {
                        "status": "success",
                        "operation": operation,
                        "channel_id": channel_id,
                        "hours_back": hours_back,
                        "message_count": len(messages),
                        "conversation_excerpt": "",
                        "time_range": {
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat(),
                            "calculated_hours": hours_back,
                            "time_range_description": f"From {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    }
                    
                    if operation == "monitor":
                        result["summary"] = "No meaningful conversation found to summarize."
                    else:
                        result["question"] = question
                        result["answer"] = "No meaningful conversation found to answer this question."
                    
                    return result
                
                # Generate response using NVIDIA API
                nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                nvidia_headers = {
                    "Authorization": f"Bearer {nvidia_api_key}",
                    "Content-Type": "application/json"
                }
                
                if operation == "monitor":
                    # Prepare the prompt for summarization
                    prompt = f"""Please provide a concise summary of the following Slack conversation from the channel. Focus on key topics, decisions made, and important information shared:

{conversation_text}

Summary:"""
                    
                    nvidia_payload = {
                        "model": "meta/llama-3.1-70b-instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "top_p": 0.7,
                        "max_tokens": 1024,
                        "stream": False
                    }
                else:
                    # Prepare the prompt for question answering
                    prompt = f"""Based on the following Slack conversation from the channel, please answer this specific question: "{question}"

Focus on finding direct answers to the question. If the information is not available in the conversation, clearly state that.

Slack Conversation:
{conversation_text}

Question: {question}
Answer:"""
                    
                    nvidia_payload = {
                        "model": "meta/llama-3.1-70b-instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,  # Lower temperature for more focused answers
                        "top_p": 0.7,
                        "max_tokens": 1024,
                        "stream": False
                    }
                
                try:
                    nvidia_response = await client.post(nvidia_url, headers=nvidia_headers, json=nvidia_payload)
                    
                    if nvidia_response.status_code == 200:
                        nvidia_data = nvidia_response.json()
                        ai_response = nvidia_data['choices'][0]['message']['content']
                    else:
                        logger.error(f"Error calling NVIDIA API: {nvidia_response.status_code}")
                        ai_response = f"Error generating response: API returned {nvidia_response.status_code}"
                        
                except Exception as e:
                    logger.error(f"Error calling NVIDIA API: {e}")
                    ai_response = f"Error generating response: {str(e)}"
                
                # Compile response
                result = {
                    "status": "success",
                    "operation": operation,
                    "channel_id": channel_id,
                    "hours_back": hours_back,
                    "message_count": len(messages),
                    "conversation_length": len(conversation_text),
                    "conversation_excerpt": conversation_text[:500] + "..." if len(conversation_text) > 500 else conversation_text,
                    "time_range": {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "calculated_hours": hours_back,
                        "time_range_description": f"From {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }
                
                if operation == "monitor":
                    result["summary"] = ai_response
                else:
                    result["question"] = question
                    result["answer"] = ai_response
                
                return result

        except httpx.TimeoutException:
            logger.error("Timeout accessing Slack API")
            return {
                "status": "error",
                "error": f"Timeout: Could not access Slack API within {config.timeout} seconds"
            }

        except Exception as e:
            logger.exception(f"Error in Slack operations: {e}")
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }

    yield FunctionInfo.from_fn(
        slack_operations,
        description="Performs Slack operations including monitoring channels and answering questions with intelligent hours_back calculation. Use operation='monitor' for conversation summary or operation='answer' with a question parameter for specific queries. Automatically calculates hours_back based on time keywords in questions (e.g., 'this week', 'last week', 'on Monday', 'yesterday'). All queries use hours_back from current time. Defaults to channel C029H7U1TEV."
    ) 