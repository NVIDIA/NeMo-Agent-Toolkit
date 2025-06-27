# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LokiLogAnalyzerConfig(FunctionBaseConfig, name="loki_log_analyzer"):
    """
    Tool that analyzes logs from Loki using LogQL queries.
    Provides real-time log analysis, error detection, and pattern recognition.
    """
    loki_url: str = Field(
        default="http://localhost:3100",
        description="Base URL for the Loki instance"
    )
    timeout: int = Field(
        default=30,
        description="Timeout for Loki API requests in seconds"
    )
    default_time_range: str = Field(
        default="1h",
        description="Default time range for log queries (e.g., '1h', '30m', '24h')"
    )
    max_log_lines: int = Field(
        default=1000,
        description="Maximum number of log lines to retrieve"
    )
    auth_username: Optional[str] = Field(
        default=None,
        description="Username for Loki authentication (if required)"
    )
    auth_password: Optional[str] = Field(
        default=None,
        description="Password for Loki authentication (if required)"
    )
    api_token: Optional[str] = Field(
        default=None,
        description="API token for Loki authentication (alternative to username/password)"
    )
    bearer_token: Optional[str] = Field(
        default=None,
        description="Bearer token for Loki authentication (alternative to username/password)"
    )


@register_function(config_type=LokiLogAnalyzerConfig)
async def loki_log_analyzer(config: LokiLogAnalyzerConfig, builder: Builder):
    """Register the Loki log analyzer function."""

    async def analyze_logs(
        query: str,
        time_range: Optional[str] = None,
        service_filter: Optional[str] = None,
        log_level: Optional[str] = None,
        cluster: Optional[str] = None,
        hostname: Optional[str] = None,
        namespace: Optional[str] = None,
        app_name: Optional[str] = None,
        container: Optional[str] = None,
        system_type: Optional[str] = None
    ) -> str:
        """
        Analyze logs from Loki based on natural language query.
        
        Args:
            query (str): Natural language description of what to search for
            time_range (str, optional): Time range for the search (e.g., '1h', '30m', '24h')
            service_filter (str, optional): Filter by specific service name
            log_level (str, optional): Filter by log level (error, warn, info, debug)
            cluster (str, optional): Filter by cluster name (e.g., 'gcp-cbf-cs-002', 'kratos-multitenant')
            hostname (str, optional): Filter by hostname (e.g., 'a4xl-007', 'cpu-small-001')
            namespace (str, optional): Filter by Kubernetes namespace
            app_name (str, optional): Filter by application name (e.g., 'python3', 'sshd', 'systemd')
            container (str, optional): Filter by container name
            system_type (str, optional): Filter by system type (e.g., 'gb200')
            
        Returns:
            str: Formatted analysis of log findings with errors, warnings, and patterns
        """
        try:
            logger.info(f"Analyzing logs for query: {query}")
            
            # Use provided time range or default
            search_time_range = time_range or config.default_time_range
            
            # Convert natural language query to LogQL
            logql_query = build_logql_query(
                query, service_filter, log_level, cluster, hostname, 
                namespace, app_name, container, system_type
            )
            
            logger.info(f"Generated LogQL query: {logql_query}")
            
            # Calculate time range for API call
            end_time = datetime.now()
            start_time = parse_time_range(search_time_range, end_time)
            
            # Query Loki API
            log_entries = await query_loki_logs(
                config, logql_query, start_time, end_time
            )
            
            if not log_entries:
                return f"📋 **Loki Log Analysis Results**\n\n" \
                       f"**Query:** {query}\n" \
                       f"**Time Range:** {search_time_range}\n" \
                       f"**LogQL:** `{logql_query}`\n\n" \
                       f"❌ **No log entries found** for the specified criteria."
            
            # Analyze log entries
            analysis = analyze_log_entries(log_entries, query)
            
            # Format results
            formatted_result = format_analysis_results(
                analysis, query, search_time_range, logql_query
            )
            
            logger.info(f"Analysis completed. Found {len(log_entries)} log entries")
            return formatted_result
            
        except Exception as e:
            logger.exception(f"Error analyzing logs: {e}")
            return f"❌ **Error analyzing logs:** {str(e)}"

    def build_logql_query(
        query: str, 
        service_filter: Optional[str], 
        log_level: Optional[str],
        cluster: Optional[str] = None,
        hostname: Optional[str] = None,
        namespace: Optional[str] = None,
        app_name: Optional[str] = None,
        container: Optional[str] = None,
        system_type: Optional[str] = None
    ) -> str:
        """Convert natural language query to LogQL using available labels."""
        
        # Check if the query is already a LogQL query (starts with { )
        if query.strip().startswith('{'):
            logger.info(f"Using provided LogQL query directly: {query}")
            return query.strip()
        
        # Parse UI-structured filters if present
        ui_filters = parse_ui_filters(query)
        if ui_filters:
            logger.info(f"Using UI-structured filters: {ui_filters}")
            # Override parameters with UI filters
            if 'region' in ui_filters:
                # Map region to appropriate cluster/hostname/etc based on your setup
                region = ui_filters['region']
                if region == 'us-east-1':
                    cluster = cluster or 'gcp-cbf-cs-002'
                elif region == 'us-west-2':
                    cluster = cluster or 'aws-prod-cluster'
                elif region == 'eu-north-1':
                    namespace = namespace or 'production'
                elif region == 'ap-northeast-1':
                    cluster = cluster or 'kratos-multitenant'
            
            # Apply other UI filters
            for label_key, label_value in ui_filters.items():
                if label_key == 'service':
                    service_filter = label_value
                elif label_key == 'level':
                    log_level = label_value
                elif label_key == 'cluster':
                    cluster = label_value
                elif label_key == 'hostname':
                    hostname = label_value
                elif label_key == 'namespace':
                    namespace = label_value
                elif label_key == 'app':
                    app_name = label_value
                elif label_key == 'container':
                    container = label_value
                elif label_key == 'system_type':
                    system_type = label_value
        
        logql_parts = []
        
        # Start with label selectors based on available labels
        labels = []
        
        # Add cluster filter (highest priority for targeting)
        if cluster:
            labels.append(f'cluster="{cluster}"')
        
        # Add hostname filter
        if hostname:
            labels.append(f'hostName="{hostname}"')
        
        # Add namespace filter (for Kubernetes logs)
        if namespace:
            labels.append(f'namespace="{namespace}"')
        
        # Add application name filter
        if app_name:
            labels.append(f'appName="{app_name}"')
        
        # Add container filter
        if container:
            labels.append(f'container="{container}"')
        
        # Add system type filter
        if system_type:
            labels.append(f'system_type="{system_type}"')
        
        # Add service filter (legacy support)
        if service_filter:
            # Try both service_name and service labels
            labels.append(f'service_name="{service_filter}"')
        
        # Add log level filter using the 'level' label
        if log_level:
            # Map common log levels to actual values seen in the logs
            level_mapping = {
                'error': 'ERROR',
                'warn': 'WARN', 
                'warning': 'WARN',
                'info': 'INFO',
                'debug': 'DEBUG'
            }
            actual_level = level_mapping.get(log_level.lower(), log_level.upper())
            labels.append(f'level="{actual_level}"')
        
        # Build label selector - ensure we have at least one non-empty label
        if labels:
            label_selector = "{" + ",".join(labels) + "}"
        else:
            # Use the most common pattern from discovery - cluster-based targeting
            # This is much better than generic job matching
            common_patterns = [
                '{cluster=~".+"}',           # Target any cluster
                '{hostName=~".+"}',          # Target any host
                '{namespace=~".+"}',         # Target any namespace
                '{appName=~".+"}',           # Target any application
                '{job=~".+"}',               # Fallback to job
            ]
            label_selector = common_patterns[0]  # Default to cluster
        
        logql_parts.append(label_selector)
        
        # Add text filters based on query content
        query_lower = query.lower()
        
        # Common error patterns
        if any(word in query_lower for word in ["error", "exception", "fail", "crash"]):
            logql_parts.append('|~ "(?i)(error|exception|fail|crash)"')
        
        # Warning patterns  
        elif any(word in query_lower for word in ["warn", "warning"]):
            logql_parts.append('|~ "(?i)(warn|warning)"')
        
        # Database related
        elif any(word in query_lower for word in ["database", "db", "sql", "connection"]):
            logql_parts.append('|~ "(?i)(database|db|sql|connection)"')
        
        # Memory related
        elif any(word in query_lower for word in ["memory", "oom", "heap", "gc"]):
            logql_parts.append('|~ "(?i)(memory|oom|heap|gc)"')
        
        # Network related
        elif any(word in query_lower for word in ["network", "timeout", "connection", "http"]):
            logql_parts.append('|~ "(?i)(network|timeout|connection|http)"')
        
        # General search - extract key terms
        else:
            # Extract potential search terms (simple approach)
            words = re.findall(r'\b\w{3,}\b', query_lower)
            if words:
                # Use first few meaningful words
                search_terms = words[:3]
                search_pattern = "|".join(search_terms)
                logql_parts.append(f'|~ "(?i)({search_pattern})"')
        
        return " ".join(logql_parts)

    def parse_ui_filters(query: str) -> Dict[str, str]:
        """Parse UI-structured filters from query string."""
        filters = {}
        
        # Simple if-else logic to parse filters from UI format
        # Format: "query text [Region: us-east-1, service: user-service]"
        if '[' in query and ']' in query:
            # Extract filter section
            start_idx = query.find('[')
            end_idx = query.find(']')
            if start_idx != -1 and end_idx != -1:
                filter_section = query[start_idx+1:end_idx]
                
                # Parse key-value pairs
                pairs = filter_section.split(',')
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        filters[key] = value
        
        return filters

    def parse_time_range(time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start time."""
        time_range = time_range.lower().strip()
        
        # Extract number and unit
        match = re.match(r'(\d+)([smhd])', time_range)
        if not match:
            # Default to 1 hour if parsing fails
            return end_time - timedelta(hours=1)
        
        amount = int(match.group(1))
        unit = match.group(2)
        
        if unit == 's':
            delta = timedelta(seconds=amount)
        elif unit == 'm':
            delta = timedelta(minutes=amount)
        elif unit == 'h':
            delta = timedelta(hours=amount)
        elif unit == 'd':
            delta = timedelta(days=amount)
        else:
            delta = timedelta(hours=1)  # Default
        
        return end_time - delta

    async def query_loki_logs(
        config: LokiLogAnalyzerConfig,
        logql_query: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Query Loki API for log entries."""
        
        # Prepare API request - handle Grafana proxy URLs correctly
        if "datasources/proxy" in config.loki_url:
            # This is a Grafana proxy URL - append Loki API path directly
            url = f"{config.loki_url}/loki/api/v1/query_range"
        else:
            # This is a direct Loki URL
            url = urljoin(config.loki_url, "/loki/api/v1/query_range")
        
        params = {
            "query": logql_query,
            "start": int(start_time.timestamp() * 1_000_000_000),  # Nanoseconds
            "end": int(end_time.timestamp() * 1_000_000_000),      # Nanoseconds
            "limit": config.max_log_lines,
            "direction": "backward"  # Most recent first
        }
        
        # Setup authentication if provided
        auth = None
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if config.auth_username and config.auth_password:
            # Basic authentication
            auth = (config.auth_username, config.auth_password)
        elif config.api_token:
            # API token authentication - for Grafana proxy, this might need to be in headers
            if "datasources/proxy" in config.loki_url:
                # Grafana proxy might use different auth header
                headers["Authorization"] = f"Bearer {config.api_token}"
                # Also try X-API-Key for some Grafana setups
                headers["X-API-Key"] = config.api_token
            else:
                headers["Authorization"] = f"Bearer {config.api_token}"
        elif config.bearer_token:
            # Bearer token authentication
            headers["Authorization"] = f"Bearer {config.bearer_token}"
        
        # Make API request
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(url, params=params, auth=auth, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Loki API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Extract log entries from Loki response format
            log_entries = []
            
            if data.get("status") == "success" and data.get("data", {}).get("result"):
                for stream in data["data"]["result"]:
                    stream_labels = stream.get("stream", {})
                    
                    for entry in stream.get("values", []):
                        timestamp_ns, log_line = entry
                        
                        # Convert nanosecond timestamp to datetime
                        timestamp = datetime.fromtimestamp(int(timestamp_ns) / 1_000_000_000)
                        
                        log_entries.append({
                            "timestamp": timestamp,
                            "message": log_line,
                            "labels": stream_labels
                        })
            
            # Sort by timestamp (most recent first)
            log_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return log_entries

    def analyze_log_entries(log_entries: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """Analyze log entries for patterns, errors, and insights."""
        
        analysis = {
            "total_entries": len(log_entries),
            "errors": [],
            "warnings": [],
            "patterns": {},
            "services": {},
            "time_distribution": {},
            "insights": []
        }
        
        for entry in log_entries:
            message = entry["message"].lower()
            labels = entry["labels"]
            timestamp = entry["timestamp"]
            
            # Extract service name from multiple sources
            service = "unknown"
            
            # Try to get service from labels first
            if "service_name" in labels:
                service = labels["service_name"]
            elif "service" in labels:
                service = labels["service"]
            elif "container" in labels:
                service = labels["container"]
            elif "appName" in labels:
                service = labels["appName"]
            elif "job" in labels:
                service = labels["job"]
            
            # If still unknown, try to parse from JSON message content
            if service == "unknown":
                try:
                    import json
                    # Check if the message is JSON
                    if entry["message"].strip().startswith('{'):
                        log_json = json.loads(entry["message"])
                        # Look for service information in JSON
                        if "resources" in log_json and "service.name" in log_json["resources"]:
                            service = log_json["resources"]["service.name"]
                        elif "attributes" in log_json and "service_name" in log_json["attributes"]:
                            service = log_json["attributes"]["service_name"]
                        elif "serviceName" in log_json:
                            service = log_json["serviceName"]
                except:
                    pass  # Keep as unknown if JSON parsing fails
            
            # If we have hostName in labels, use it as a fallback identifier
            if service == "unknown" and "hostName" in labels:
                service = f"host:{labels['hostName']}"
            elif service == "unknown" and "hostname" in labels:
                service = f"host:{labels['hostname']}"
            
            # Count by service
            analysis["services"][service] = analysis["services"].get(service, 0) + 1
            
            # Time distribution (by hour)
            hour_key = timestamp.strftime("%H:00")
            analysis["time_distribution"][hour_key] = analysis["time_distribution"].get(hour_key, 0) + 1
            
            # Error detection
            if any(error_word in message for error_word in ["error", "exception", "fail", "crash", "fatal"]):
                analysis["errors"].append({
                    "timestamp": timestamp,
                    "service": service,
                    "message": entry["message"][:200] + "..." if len(entry["message"]) > 200 else entry["message"]
                })
            
            # Warning detection
            elif any(warn_word in message for warn_word in ["warn", "warning"]):
                analysis["warnings"].append({
                    "timestamp": timestamp,
                    "service": service,
                    "message": entry["message"][:200] + "..." if len(entry["message"]) > 200 else entry["message"]
                })
            
            # Pattern detection
            for pattern in ["timeout", "connection", "memory", "database", "http", "ssl", "auth"]:
                if pattern in message:
                    analysis["patterns"][pattern] = analysis["patterns"].get(pattern, 0) + 1
        
        # Generate insights
        if analysis["errors"]:
            analysis["insights"].append(f"Found {len(analysis['errors'])} error entries")
        
        if analysis["warnings"]:
            analysis["insights"].append(f"Found {len(analysis['warnings'])} warning entries")
        
        # Top patterns
        if analysis["patterns"]:
            top_pattern = max(analysis["patterns"], key=analysis["patterns"].get)
            analysis["insights"].append(f"Most common pattern: '{top_pattern}' ({analysis['patterns'][top_pattern]} occurrences)")
        
        # Service distribution
        if len(analysis["services"]) > 1:
            top_service = max(analysis["services"], key=analysis["services"].get)
            analysis["insights"].append(f"Most active service: '{top_service}' ({analysis['services'][top_service]} entries)")
        
        return analysis

    def format_analysis_results(
        analysis: Dict[str, Any],
        original_query: str,
        time_range: str,
        logql_query: str
    ) -> str:
        """Format analysis results into a readable report."""
        
        result = f"📋 **Loki Log Analysis Results**\n\n"
        result += f"**Query:** {original_query}\n"
        result += f"**Time Range:** {time_range}\n"
        result += f"**LogQL:** `{logql_query}`\n"
        result += f"**Total Entries:** {analysis['total_entries']}\n\n"
        
        # Errors section
        if analysis["errors"]:
            result += f"🚨 **Errors Found ({len(analysis['errors'])} total):**\n"
            for i, error in enumerate(analysis["errors"][:5]):  # Show top 5
                result += f"**{i+1}.** [{error['timestamp'].strftime('%H:%M:%S')}] [{error['service']}]\n"
                result += f"   {error['message']}\n\n"
            
            if len(analysis["errors"]) > 5:
                result += f"   ... and {len(analysis['errors']) - 5} more errors\n\n"
        
        # Warnings section
        if analysis["warnings"]:
            result += f"⚠️ **Warnings Found ({len(analysis['warnings'])} total):**\n"
            for i, warning in enumerate(analysis["warnings"][:3]):  # Show top 3
                result += f"**{i+1}.** [{warning['timestamp'].strftime('%H:%M:%S')}] [{warning['service']}]\n"
                result += f"   {warning['message']}\n\n"
            
            if len(analysis["warnings"]) > 3:
                result += f"   ... and {len(analysis['warnings']) - 3} more warnings\n\n"
        
        # Patterns section
        if analysis["patterns"]:
            result += f"🔍 **Detected Patterns:**\n"
            sorted_patterns = sorted(analysis["patterns"].items(), key=lambda x: x[1], reverse=True)
            for pattern, count in sorted_patterns[:5]:
                result += f"- **{pattern.title()}**: {count} occurrences\n"
            result += "\n"
        
        # Services section
        if analysis["services"]:
            result += f"🏗️ **Services Activity:**\n"
            sorted_services = sorted(analysis["services"].items(), key=lambda x: x[1], reverse=True)
            for service, count in sorted_services:
                result += f"- **{service}**: {count} entries\n"
            result += "\n"
        
        # Time distribution
        if analysis["time_distribution"]:
            result += f"⏰ **Time Distribution:**\n"
            sorted_times = sorted(analysis["time_distribution"].items())
            for time_hour, count in sorted_times:
                result += f"- **{time_hour}**: {count} entries\n"
            result += "\n"
        
        # Insights section
        if analysis["insights"]:
            result += f"💡 **Key Insights:**\n"
            for insight in analysis["insights"]:
                result += f"- {insight}\n"
            result += "\n"
        
        # Recommendations
        result += f"🎯 **Recommendations:**\n"
        if analysis["errors"]:
            result += f"- Investigate the {len(analysis['errors'])} error entries for root causes\n"
        if analysis["patterns"].get("timeout", 0) > 0:
            result += f"- Check timeout configurations due to {analysis['patterns']['timeout']} timeout occurrences\n"
        if analysis["patterns"].get("memory", 0) > 0:
            result += f"- Monitor memory usage due to {analysis['patterns']['memory']} memory-related entries\n"
        if len(analysis["services"]) > 1:
            top_service = max(analysis["services"], key=analysis["services"].get)
            result += f"- Focus on '{top_service}' service which has the highest log activity\n"
        
        return result

    yield FunctionInfo.from_fn(
        analyze_logs,
        description="Analyze logs from Loki using natural language queries to detect errors, warnings, and patterns"
    )

 