"""Database schema and promptfor Text2SQL."""

# DDL statements for training
# Define your database schema here to help the model understand table structures
TRAINING_DDL: list[str] = [
    "CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100), created_at TIMESTAMP)",
    "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, product VARCHAR(100), amount DECIMAL(10,2), order_date DATE)",
    "CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), category VARCHAR(50), price DECIMAL(10,2))",
]

# Documentation for training
# Provide context and business logic about your tables and columns
TRAINING_DOCUMENTATION: list[str] = [
    "The customers table contains all registered users. The created_at field shows registration date.",
    "Orders table tracks all purchases. The amount field is in USD.",
    "Products are organized by category (electronics, clothing, home, etc.).",
]

# Question-SQL examples for training
# Provide example question-SQL pairs to teach the model your query patterns
TRAINING_EXAMPLES: list[dict[str, str]] = [
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

# Remove from PR
ACTIVE_TABLES = [
    'catalog.schema.table_a',
    'catalog.schema.table_b'
]

# Default prompts
RESPONSE_GUIDELINES = """
Response Guidelines:
1. Carefully analyze the question to understand the user’s intent, target columns, filters, and any aggregation or grouping requirements.
2. Generate a valid SQL query and return it in JSON format as shown:

{
  "sql": "generated SQL query here",
  "explanation": "explanation of the SQL query"
}
"""

TRAINING_PROMPT = """
Response Guidelines:
1. Generate 20 natural language questions and their corresponding valid SQL queries.
2. Output JSON like: [{{"question": "...", "sql": "..."}}]
"""
