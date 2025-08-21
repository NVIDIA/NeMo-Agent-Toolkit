from typing import Any
from datetime import datetime, timedelta
from .data_models import FormSubmission, FUNCTION_INTENTS


class MockMCPServer:
    """Mock MCP Server for testing purposes"""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
    
    async def call_function(self, function_name: str, data: dict[str, str]) -> dict[str, Any]:
        """Mock function call to MCP server"""
        # Simulate some processing time and return structured response
        return {
            "server": self.server_name,
            "function": function_name,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "data": data,
            "result": self._generate_mock_result(function_name, data)
        }
    
    def _generate_mock_result(self, function_name: str, data: dict[str, str]) -> dict[str, Any]:
        """Generate mock results based on function type"""
        if function_name == "Fee Inquiry":
            return {
                "account_fees": [
                    {
                        "fee_type": "Monthly Maintenance",
                        "amount": 12.95,
                        "date": data.get("transaction_date", "2024-01-15"),
                        "waivable": True
                    },
                    {
                        "fee_type": "ATM Fee",
                        "amount": 3.00,
                        "date": data.get("transaction_date", "2024-01-14"),
                        "waivable": False
                    }
                ],
                "total_fees": 15.95,
                "account_id": data.get("account_id", "Unknown")
            }
        elif function_name == "Sales Opportunity":
            return {
                "eligible_products": [
                    {
                        "product_type": "Credit Card",
                        "product_name": "Royal Bank Visa Infinite",
                        "annual_fee": 120.00,
                        "credit_limit": 15000,
                        "interest_rate": "19.99%"
                    },
                    {
                        "product_type": "Personal Loan",
                        "product_name": "Flexible Personal Loan",
                        "max_amount": 50000,
                        "interest_rate": "6.99%",
                        "term_months": 60
                    }
                ],
                "customer_name": data.get("customer_name", "Unknown"),
                "account_id": data.get("account_id", "Unknown"),
                "pre_approved": True
            }
        elif function_name == "Knowledge Base Inquiry":
            return {
                "knowledge_base_results": [
                    {
                        "topic": "Account Services",
                        "content": "Royal Bank of Canada offers comprehensive "
                                   "banking services including checking accounts, "
                                   "savings accounts, credit cards, and lending products.",
                        "relevance_score": 0.95
                    }
                ],
                "customer_name": data.get("customer_name", "Unknown"),
                "account_id": data.get("account_id", "Unknown")
            }
        else:
            return {"error": f"Unknown function: {function_name}"}


class DateTimeConversionServer:
    """LLM-powered MCP Server for DateTime conversion operations"""
    
    def __init__(self, builder=None, llm_name: str = "llama3.1-8b"):
        self.server_name = "DateTime Conversion Server"
        self.builder = builder
        self.llm_name = llm_name
        self.llm = None
    
    async def initialize(self):
        """Initialize the LLM for date conversion"""
        if self.builder and not self.llm:
            from aiq.builder.framework_enum import LLMFrameworkEnum
            self.llm = await self.builder.get_llm(self.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    def _get_date_conversion_prompt(self, date_expression: str, current_datetime: str) -> str:
        """Generate prompt for LLM-based date conversion"""
        return f"""
You are a specialized DateTime conversion AI. Your job is to convert natural language date expressions into structured datetime information.

CURRENT CONTEXT:
- Current date and time (UTC): {current_datetime}

INPUT DATE EXPRESSION: "{date_expression}"

TASK:
Convert this natural language date expression into a specific date. Handle all possible forms including:
- Relative dates (yesterday, tomorrow, last week, next month)
- Specific days (last Thursday, this Friday, next Monday)
- Partial dates (March 15th, December, summer 2023)
- Informal expressions (a few days ago, end of last month, beginning of next year)
- Context-dependent dates (payday, weekend, holiday season)

RULES:
1. If the expression is ambiguous, make the most reasonable assumption
2. For incomplete dates, infer the most likely year/month from context
3. For relative expressions, calculate from the current UTC date and time provided
4. All output dates should be in UTC timezone
5. Return JSON ONLY with this exact structure:

{{
    "absolute_date": "YYYY-MM-DD",
    "formatted_date": "Month DD, YYYY",
    "day_of_week": "Day",
    "iso_format": "YYYY-MM-DDTHH:MM:SS",
    "confidence": 0.95,
    "reasoning": "Brief explanation of date calculation",
    "status": "success"
}}

CONFIDENCE LEVELS:
- 1.0: Exact date specified
- 0.9: Clear relative date with unambiguous calculation
- 0.8: Reasonably clear with minor ambiguity
- 0.7: Some ambiguity but reasonable interpretation
- 0.5: Highly ambiguous, best guess provided

If you cannot parse the date at all, return:
{{
    "absolute_date": null,
    "formatted_date": "{date_expression}",
    "day_of_week": null,
    "iso_format": null,
    "confidence": 0.0,
    "reasoning": "Could not parse date expression",
    "status": "failed"
}}

Return JSON only, no explanations.
"""
    
    async def convert_relative_date(self, relative_date_text: str) -> dict[str, Any]:
        """Convert natural language date expressions using LLM"""
        try:
            if not self.llm:
                await self.initialize()
            
            if not self.llm:
                # Fallback if LLM not available
                return self._fallback_conversion(relative_date_text)
            
            # Get current UTC datetime
            current_utc = datetime.utcnow()
            current_datetime_str = current_utc.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")
            
            # Generate prompt for LLM
            prompt = self._get_date_conversion_prompt(relative_date_text, current_datetime_str)
            
            # Get LLM response
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response
            import json
            llm_result = json.loads(response_text)
            
            # Validate and format response
            return {
                "server": self.server_name,
                "function": "convert_relative_date",
                "input": relative_date_text,
                "output": {
                    "absolute_date": llm_result.get("absolute_date"),
                    "formatted_date": llm_result.get("formatted_date"),
                    "day_of_week": llm_result.get("day_of_week"),
                    "iso_format": llm_result.get("iso_format"),
                    "confidence": llm_result.get("confidence", 0.0),
                    "reasoning": llm_result.get("reasoning", "")
                },
                "status": llm_result.get("status", "success"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except json.JSONDecodeError:
            return self._fallback_conversion(relative_date_text)
        except Exception as e:
            return {
                "server": self.server_name,
                "function": "convert_relative_date",
                "input": relative_date_text,
                "error": str(e),
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _fallback_conversion(self, date_text: str) -> dict[str, Any]:
        """Simple fallback conversion for common patterns when LLM fails"""
        date_lower = date_text.lower().strip()
        current_utc = datetime.utcnow()
        
        # Basic patterns as fallback
        if "last thursday" in date_lower:
            days_back = (current_utc.weekday() + 4) % 7
            if days_back == 0:
                days_back = 7
            target_date = current_utc - timedelta(days=days_back)
        elif "yesterday" in date_lower:
            target_date = current_utc - timedelta(days=1)
        elif "today" in date_lower:
            target_date = current_utc
        else:
            return {
                "server": self.server_name,
                "function": "convert_relative_date",
                "input": date_text,
                "output": {
                    "absolute_date": None,
                    "formatted_date": date_text,
                    "day_of_week": None,
                    "iso_format": None,
                    "confidence": 0.0,
                    "reasoning": "Fallback: Could not parse date expression"
                },
                "status": "failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "server": self.server_name,
            "function": "convert_relative_date",
            "input": date_text,
            "output": {
                "absolute_date": target_date.strftime("%Y-%m-%d"),
                "formatted_date": target_date.strftime("%B %d, %Y"),
                "day_of_week": target_date.strftime("%A"),
                "iso_format": target_date.isoformat(),
                "confidence": 0.8,
                "reasoning": "Fallback: Basic pattern matching"
            },
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }


class MCPServerManager:
    """Manages multiple MCP servers"""
    
    def __init__(self, builder=None):
        self.builder = builder
        self.servers = {
            "fee_inquiry_server": MockMCPServer("Fee Inquiry Server"),
            "sales_opportunity_server": MockMCPServer("Sales Opportunity Server"),
            "knowledge_base_server": MockMCPServer("Knowledge Base Server"),
            "datetime_conversion_server": DateTimeConversionServer(builder=builder)
        }
    
    async def initialize(self):
        """Initialize all servers that require LLMs"""
        datetime_server = self.servers.get("datetime_conversion_server")
        if datetime_server and hasattr(datetime_server, 'initialize'):
            await datetime_server.initialize()
    
    async def call_server(self, server_name: str, function_name: str, data: dict[str, str]) -> dict[str, Any]:
        """Call a specific MCP server with the given function and data"""
        if server_name not in self.servers:
            return {
                "error": f"Unknown server: {server_name}",
                "status": "error"
            }
        
        server = self.servers[server_name]
        return await server.call_function(function_name, data)
    
    async def process_form_submission(self, form_submission: FormSubmission) -> dict[str, Any]:
        """Process a form submission by calling the appropriate MCP server"""
        function_intent = FUNCTION_INTENTS.get(form_submission.function_name)
        
        if not function_intent:
            return {
                "error": f"Unknown function intent: {form_submission.function_name}",
                "status": "error"
            }
        
        # Call the appropriate MCP server
        result = await self.call_server(
            function_intent.mcp_server,
            form_submission.function_name,
            form_submission.data
        )
        
        return result
    
    async def convert_relative_date(self, relative_date_text: str) -> dict[str, Any]:
        """Convert relative date using the DateTime conversion server"""
        datetime_server = self.servers.get("datetime_conversion_server")
        if isinstance(datetime_server, DateTimeConversionServer):
            return await datetime_server.convert_relative_date(relative_date_text)
        else:
            return {
                "error": "DateTime conversion server not available",
                "status": "error"
            }
    
    def get_available_servers(self) -> list[str]:
        """Get list of available server names"""
        return list(self.servers.keys())
    
    def get_server_functions(self, server_name: str) -> list[str]:
        """Get list of functions available on a specific server"""
        server_functions = {
            "fee_inquiry_server": ["Fee Inquiry"],
            "sales_opportunity_server": ["Sales Opportunity"],
            "knowledge_base_server": ["Knowledge Base Inquiry"],
            "datetime_conversion_server": ["convert_relative_date"]
        }
        return server_functions.get(server_name, [])
