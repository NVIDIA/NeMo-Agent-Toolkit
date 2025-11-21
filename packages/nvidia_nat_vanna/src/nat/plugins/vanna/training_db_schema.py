# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Manual training data and configuration for Vanna text-to-SQL.

This module provides default DDL statements, documentation examples,
question-SQL pairs, and prompt templates used to train and configure
the Vanna text-to-SQL model with database schema context.
"""

# yapf: disable
# ruff: noqa: E501

# DDL statements for training
# Define your database schema here to help the model understand table structures
VANNA_TRAINING_DDL: list[str] = [
    "CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100), created_at TIMESTAMP)",
    "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, product VARCHAR(100), amount DECIMAL(10,2), order_date DATE)",
    "CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), category VARCHAR(50), price DECIMAL(10,2))",
]

# Documentation for training
# Provide context and business logic about your tables and columns
VANNA_TRAINING_DOCUMENTATION: list[str] = [
    "The customers table contains all registered users. The created_at field shows registration date.",
    "Orders table tracks all purchases. The amount field is in USD.",
    "Products are organized by category (electronics, clothing, home, etc.).",
]

# Question-SQL examples for training
# Provide example question-SQL pairs to teach the model your query patterns
VANNA_TRAINING_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "How many customers do we have?",
        "sql": "SELECT COUNT(*) as customer_count FROM customers",
    },
    {
        "question": "What is the total revenue?",
        "sql": "SELECT SUM(amount) as total_revenue FROM orders",
    },
    {
        "question": "Who are the top 5 customers by spending?",
        "sql": "SELECT c.name, SUM(o.amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5",
    },
]

VANNA_ACTIVE_TABLES = ['catalog.schema.table_a', 'catalog.schema.table_b']

# Default prompts
VANNA_RESPONSE_GUIDELINES = """
Response Guidelines:
1. Carefully analyze the question to understand the user's intent, target columns, filters, and any aggregation or grouping requirements.
2. Output only JSON:
{
    "sql": "<valid SQL query>",
    "explanation": "<brief description>",
}
"""

VANNA_TRAINING_PROMPT = """
Response Guidelines:
1. Generate 20 natural language questions and their corresponding valid SQL queries.
2. Output JSON like: [{{"question": "...", "sql": "..."}}]
"""
