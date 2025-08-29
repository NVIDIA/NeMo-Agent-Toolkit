#!/usr/bin/env python3
"""
🏦 Banking Customer Data Generator

This script generates synthetic banking customer personas, queries, and tools using NeMo Data Designer.
It focuses ONLY on data generation, not evaluation.

Requirements:
- NeMo Data Designer microservice deployed locally via docker compose
- NeMo Microservices SDK installed
- Access to the hackathon cluster (https://nmp.aire.nvidia.com)
"""

import json
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field

from nemo_microservices import NeMoMicroservices
from nemo_microservices.beta.data_designer import (
    DataDesignerConfigBuilder,
    DataDesignerClient,
)
from nemo_microservices.beta.data_designer.config import columns as C
from nemo_microservices.beta.data_designer.config import params as P


class CustomerQuery(BaseModel):
    """Structure for customer queries"""
    query: str = Field(description="The customer's request or question")
    urgency: Literal["Low", "Medium", "High"] = Field(description="Urgency level of the request")
    complexity: Literal["Simple", "Moderate", "Complex"] = Field(description="Complexity of the request")


class ToolParameter(BaseModel):
    """Structure for tool parameters"""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, integer, float, boolean)")
    description: str = Field(description="Parameter description")
    required: bool = Field(description="Whether this parameter is required")


class BankingTool(BaseModel):
    """Structure for banking tools"""
    name: str = Field(description="Tool function name")
    description: str = Field(description="What this tool does")
    parameters: list[ToolParameter] = Field(description="List of parameters this tool accepts")


class ToolCall(BaseModel):
    """Structure for tool calls"""
    function_name: str = Field(description="Name of the function to call")
    arguments: dict = Field(description="Arguments to pass to the function")


def initialize_data_designer():
    """Initialize the NeMo Data Designer client and config builder"""
    print("🚀 Initializing NeMo Data Designer...")
    
    # Initialize the NDD client
    ndd = DataDesignerClient(client=NeMoMicroservices(base_url="http://localhost:8000", timeout=600))
    
    # Configure model endpoints for the hackathon cluster - using 70B model for intelligence
    candidate_model_endpoint = "https://nim.aire.nvidia.com/v1/"
    candidate_model_id = "meta/llama-3.3-70b-instruct"  # Using 70B for intelligence
    candidate_model_alias = "llama33-70b"

    # Create model configuration
    model_configs = [
        P.ModelConfig(
            alias=candidate_model_alias,
            inference_parameters=P.InferenceParameters(
                max_tokens=2048,  # Increased for 70B model to handle complex outputs
                temperature=0.7,
                top_p=1.0,
            ),
            model=P.Model(api_endpoint=P.ApiEndpoint(
                model_id=candidate_model_id,
                url=candidate_model_endpoint,
            ), ),
        ),
    ]

    config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
    
    print("✅ Data Designer initialized successfully")
    return ndd, config_builder


def add_customer_persona_columns(config_builder):
    """Add columns for customer persona generation"""
    print("👤 Adding customer persona columns...")
    
    # Customer demographic information
    config_builder.add_column(
        C.SamplerColumn(
            name="customer_name",
            type=P.SamplerType.PERSON,
            params=P.PersonSamplerParams(age_range=[18, 75]),
            description="Customer's full name"
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="customer_age",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=18, high=75),
            convert_to="int",
            description="Customer's age"
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="monthly_income",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["$0-$2,000", "$2,000-$5,000", "$5,000-$10,000", "$10,000-$20,000", "$20,000+"],
                weights=[2, 3, 3, 2, 1]
            ),
            description="Customer's monthly income range"
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="account_type",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["Checking", "Savings", "Premium", "Student", "Senior"],
                weights=[4, 3, 1, 1, 1]
            ),
            description="Type of bank account"
        )
    )

    config_builder.add_column(
        C.SamplerColumn(
            name="banking_experience",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["Beginner", "Intermediate", "Advanced"],
                weights=[2, 3, 1]
            ),
            description="Customer's banking experience level"
        )
    )

    config_builder.validate()
    print("✅ Customer persona columns added successfully")


def add_customer_query_generation(config_builder):
    """Add customer query generation column"""
    print("💬 Adding customer query generation...")
    
    config_builder.add_column(
        C.LLMStructuredColumn(
            name="customer_query",
            prompt=(
                "Generate a realistic customer query for a banking customer. "
                "Customer: {{ customer_name.first_name }} {{ customer_name.last_name }}, "
                "Age: {{ customer_age }}, Income: {{ monthly_income }}, "
                "Account: {{ account_type }}, Experience: {{ banking_experience }}. "
                "Generate a natural, realistic banking request that this customer might make. "
                "Examples: account balance, money transfer, loan application, card issues, etc."
            ),
            system_prompt=(
                "You are a banking customer service representative. Generate realistic customer queries "
                "that match the customer's profile and banking experience level. Keep queries natural "
                "and appropriate for the customer's situation."
            ),
            model_alias="llama33-70b",  # Using 70B model for intelligence
            output_format=CustomerQuery,
        )
    )

    config_builder.validate()
    print("✅ Customer query generation added successfully")


def add_tool_generation(config_builder):
    """Add banking tool generation column"""
    print("🛠️ Adding banking tool generation...")
    
    config_builder.add_column(
        C.LLMStructuredColumn(
            name="required_tools",
            prompt=(
                "Based on the customer query: '{{ customer_query.query }}', "
                "generate 2-3 simple banking tools. "
                "Use basic banking functions like: "
                "check_account_balance, search_customer_profile, process_transfer, "
                "verify_transaction, check_fraud_risk."
            ),
            system_prompt=(
                "You are a banking system architect. Generate appropriate tools that an AI agent "
                "would need to handle the customer's request. Each tool should have a clear purpose "
                "and appropriate parameters. Return exactly the number of tools requested."
            ),
            model_alias="llama33-70b",  # Using 70B model for intelligence
            output_format=BankingTool,
        )
    )

    config_builder.validate()
    print("✅ Tool generation added successfully")


def add_tool_call_generation(config_builder):
    """Add tool call generation column"""
    print("📞 Adding tool call generation...")
    
    config_builder.add_column(
        C.LLMStructuredColumn(
            name="tool_calls",
            prompt=(
                "Based on the customer query '{{ customer_query.query }}', "
                "generate simple tool calls with basic arguments. "
                "Example: for 'check_account_balance', use arguments like "
                "customer_id and account_number."
            ),
            system_prompt=(
                "You are an AI agent that needs to execute banking tools. Generate specific tool calls "
                "with realistic arguments based on the customer's request and available tools. "
                "Make sure the arguments match the tool parameters and are appropriate for the query."
            ),
            model_alias="llama33-70b",  # Using 70B model for intelligence
            output_format=ToolCall,
        )
    )

    config_builder.validate()
    print("✅ Tool call generation added successfully")


def generate_preview(ndd, config_builder):
    """Generate a preview of the dataset"""
    print("👀 Generating dataset preview...")
    
    try:
        preview = ndd.preview(config_builder, verbose_logging=True)
        print("✅ Preview generated successfully")
        print(f"Preview dataset shape: {preview.dataset.shape}")
        return preview
    except Exception as e:
        print(f"❌ Error generating preview: {e}")
        return None


def generate_full_dataset(ndd, config_builder, num_records=5):
    """Generate the full dataset"""
    print(f"🧬 Generating full dataset with {num_records} records...")
    
    try:
        results = ndd.create(config_builder, num_records=num_records, wait_until_done=True)
        dataset = results.load_dataset()
        
        print(f"✅ Dataset generated successfully with {len(dataset)} records")
        print(f"Dataset columns: {dataset.columns.tolist()}")
        
        return dataset, results
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return None, None


def post_process_dataset(dataset):
    """Post-process the dataset to extract structured data"""
    print("🔧 Post-processing dataset...")
    
    def extract_json_field(json_str, field_name):
        try:
            if isinstance(json_str, str):
                # Handle Python dict strings (single quotes) by converting to valid JSON
                if json_str.startswith("{") and json_str.endswith("}"):
                    # Convert Python dict string to valid JSON
                    import ast
                    data = ast.literal_eval(json_str)
                else:
                    data = json.loads(json_str)
            else:
                data = json_str
            return data.get(field_name, None)
        except Exception as e:
            print(f"  Parse Error for field '{field_name}': {e}")
            return None

    # Extract customer query fields
    dataset['query_text'] = dataset['customer_query'].apply(lambda x: extract_json_field(x, 'query'))
    dataset['query_urgency'] = dataset['customer_query'].apply(lambda x: extract_json_field(x, 'urgency'))
    dataset['query_complexity'] = dataset['customer_query'].apply(lambda x: extract_json_field(x, 'complexity'))

    # Extract customer name fields
    dataset['first_name'] = dataset['customer_name'].apply(lambda x: extract_json_field(x, 'first_name'))
    dataset['last_name'] = dataset['customer_name'].apply(lambda x: extract_json_field(x, 'last_name'))

    print("✅ Dataset post-processing completed")
    return dataset


def export_dataset(dataset, filename='banking_agent_evaluation_data.json'):
    """Export the dataset to JSON format with realistic banking tools and expected tool calls"""
    print(f"💾 Exporting dataset to {filename}...")
    
    # Define realistic banking tools that would be available
    banking_tools = [
        {
            "type": "function",
            "function": {
                "name": "check_account_balance",
                "description": "Check the current balance of a customer's account",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_number": {
                            "type": "string",
                            "description": "The account number to check"
                        },
                        "customer_id": {
                            "type": "string", 
                            "description": "The customer's unique identifier"
                        }
                    },
                    "required": ["account_number", "customer_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "setup_automatic_bill_payment",
                "description": "Set up automatic bill payments for utility bills",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer's unique identifier"
                        },
                        "utility_provider": {
                            "type": "string",
                            "description": "Name of the utility provider"
                        },
                        "account_number": {
                            "type": "string",
                            "description": "Utility account number"
                        },
                        "payment_amount": {
                            "type": "number",
                            "description": "Monthly payment amount"
                        },
                        "payment_date": {
                            "type": "string",
                            "description": "Day of month for payment (1-31)"
                        }
                    },
                    "required": ["customer_id", "utility_provider", "account_number", "payment_amount"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_customer_profile",
                "description": "Search for customer profile information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer's unique identifier"
                        },
                        "search_field": {
                            "type": "string",
                            "description": "Field to search (name, email, phone, etc.)"
                        }
                    },
                    "required": ["customer_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "process_money_transfer",
                "description": "Process a money transfer between accounts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {
                            "type": "string",
                            "description": "Source account number"
                        },
                        "to_account": {
                            "type": "string",
                            "description": "Destination account number"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Transfer amount"
                        },
                        "customer_id": {
                            "type": "string",
                            "description": "The customer's unique identifier"
                        }
                    },
                    "required": ["from_account", "to_account", "amount", "customer_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "verify_transaction_fraud",
                "description": "Verify if a transaction is fraudulent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "Unique transaction identifier"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Transaction amount"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination account or merchant"
                        }
                    },
                    "required": ["transaction_id", "amount", "destination"]
                }
            }
        }
    ]
    
    export_data = []

    for _, row in dataset.iterrows():
        # Generate expected tool calls based on the customer query
        query = row.get('query_text', '').lower()
        expected_tool_calls = []
        
        if "account balance" in query or "balance" in query:
            expected_tool_calls.append({
                "function": {
                    "name": "check_account_balance",
                    "arguments": {
                        "account_number": "ACC123456789",
                        "customer_id": "CUST001"
                    }
                }
            })
        
        if "automatic bill payment" in query or "bill payment" in query or "utility" in query:
            expected_tool_calls.append({
                "function": {
                    "name": "setup_automatic_bill_payment",
                    "arguments": {
                        "customer_id": "CUST001",
                        "utility_provider": "City Utilities",
                        "account_number": "UTIL987654321",
                        "payment_amount": 150.00,
                        "payment_date": "15"
                    }
                }
            })
        
        if "transfer" in query or "send money" in query:
            expected_tool_calls.append({
                "function": {
                    "name": "process_money_transfer",
                    "arguments": {
                        "from_account": "ACC123456789",
                        "to_account": "ACC987654321",
                        "amount": 2000.00,
                        "customer_id": "CUST001"
                    }
                }
            })
        
        if "fraud" in query or "suspicious" in query:
            expected_tool_calls.append({
                "function": {
                    "name": "verify_transaction_fraud",
                    "arguments": {
                        "transaction_id": "TXN123456",
                        "amount": 500.00,
                        "destination": "Unknown Merchant"
                    }
                }
            })
        
        # If no specific tool calls were identified, add a general customer profile search
        if not expected_tool_calls:
            expected_tool_calls.append({
                "function": {
                    "name": "search_customer_profile",
                    "arguments": {
                        "customer_id": "CUST001",
                        "search_field": "profile"
                    }
                }
            })
        
        # Create the export record
        export_record = {
            "customer_profile": {
                "name": f"{row.get('first_name', '')} {row.get('last_name', '')}",
                "age": row.get('customer_age', ''),
                "monthly_income": row.get('monthly_income', ''),
                "account_type": row.get('account_type', ''),
                "banking_experience": row.get('banking_experience', '')
            },
            "messages": [
                {
                    "role": "user",
                    "content": row.get('query_text', '')
                }
            ],
            "tools": banking_tools,
            "tool_calls": expected_tool_calls,
            "query_metadata": {
                "urgency": row.get('query_urgency', ''),
                "complexity": row.get('query_complexity', '')
            }
        }
        
        export_data.append(export_record)

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Exported {len(export_data)} records to '{filename}'")
    return export_data


def display_sample_data(export_data):
    """Display sample data for verification"""
    print("\n📋 Sample Generated Data:")
    print("=" * 50)
    
    if export_data:
        sample = export_data[0]
        print(f"Customer: {sample['customer_profile']['name']}")
        print(f"Age: {sample['customer_profile']['age']}")
        print(f"Income: {sample['customer_profile']['monthly_income']}")
        print(f"Account: {sample['customer_profile']['account_type']}")
        print(f"Experience: {sample['customer_profile']['banking_experience']}")
        print(f"Query: {sample['messages'][0]['content']}")
        print(f"Urgency: {sample['query_metadata']['urgency']}")
        print(f"Complexity: {sample['query_metadata']['complexity']}")
        print(f"Tools Available: {len(sample['tools'])}")
        print(f"Expected Tool Calls: {len(sample['tool_calls'])}")
        
        print("\n🔧 Sample Tool:")
        if sample['tools']:
            tool = sample['tools'][0]
            print(f"  Name: {tool['function']['name']}")
            print(f"  Description: {tool['function']['description']}")
        
        print("\n📞 Sample Tool Call:")
        if sample['tool_calls']:
            tool_call = sample['tool_calls'][0]
            print(f"  Function: {tool_call['function']['name']}")
            print(f"  Arguments: {tool_call['function']['arguments']}")


def main():
    """Main function to run the banking data generator"""
    print("🏦 Banking Customer Data Generator")
    print("=" * 40)
    print("📝 This script generates customer data and tools ONLY")
    print("🔍 Use 'banking_data_evaluator.py' for evaluation")
    print("=" * 40)
    
    try:
        # Step 1: Initialize Data Designer
        ndd, config_builder = initialize_data_designer()
        
        # Step 2: Add customer persona columns
        add_customer_persona_columns(config_builder)
        
        # Step 3: Add customer query generation
        add_customer_query_generation(config_builder)
        
        # Step 4: Add tool generation
        add_tool_generation(config_builder)
        
        # Step 5: Add tool call generation
        add_tool_call_generation(config_builder)
        
        # Step 6: Generate preview
        preview = generate_preview(ndd, config_builder)
        if preview is None:
            print("❌ Failed to generate preview. Exiting.")
            return
        
        # Step 7: Generate full dataset
        dataset, results = generate_full_dataset(ndd, config_builder, num_records=5)
        if dataset is None:
            print("❌ Failed to generate dataset. Exiting.")
            return
        
        # Step 8: Post-process the dataset
        print("\n🔧 Post-processing the dataset...")
        dataset = post_process_dataset(dataset)
        
        # Step 9: Export dataset
        export_data = export_dataset(dataset)
        
        # Step 10: Display sample data
        display_sample_data(export_data)
        
        print("\n🎉 Banking Data Generator completed successfully!")
        print("\n📁 Generated Files:")
        print(f"  - {export_data[0]['customer_profile']['name'] if export_data else 'N/A'}")
        print(f"  - {len(export_data)} customer records")
        print(f"  - {len(export_data[0]['tools']) if export_data else 0} banking tools")
        print(f"  - {len(export_data[0]['tool_calls']) if export_data else 0} expected tool calls")
        
        print("\n🔄 Next Steps:")
        print("1. Review the generated data in 'banking_agent_evaluation_data.json'")
        print("2. Run 'banking_data_evaluator.py' to evaluate the data quality")
        print("3. Use the data to test your banking AI agent")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
