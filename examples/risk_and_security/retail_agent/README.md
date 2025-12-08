<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Retail Agent for Gardening Equipment

This example demonstrates an end-to-end agentic workflow for a retail customer service agent specialized in gardening equipment. The agent processes customer emails to handle product inquiries, order placement, and review submissions using the NeMo Agent toolkit.

## Table of Contents

- [Retail Agent for Gardening Equipment](#retail-agent-for-gardening-equipment)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
    - [Run the Workflow](#run-the-workflow)
  - [Agent Capabilities](#agent-capabilities)
  - [Example Usage](#example-usage)
    - [Product Inquiry](#product-inquiry)
    - [Review Submission](#review-submission)
    - [Order Placement](#order-placement)
    - [Order with Multiple Products](#order-with-multiple-products)
  - [Data Structure](#data-structure)
    - [Customer Information](#customer-information)
    - [Product Information](#product-information)
    - [Available Products](#available-products)
  - [Available Tools](#available-tools)
    - [Retail Tools Function Group](#retail-tools-function-group)
  - [Evaluation](#evaluation)
    - [Running the Evaluation](#running-the-evaluation)
    - [Evaluation Dataset](#evaluation-dataset)
    - [Evaluation Output](#evaluation-output)
  - [Notes](#notes)
  - [Architecture](#architecture)

---

## Key Features

- **Email Processing:** Processes customer emails to understand intent (product inquiry, review submission, or order placement).
- **Customer Management:** Retrieves customer information including purchase history, total orders, and past reviews.
- **Product Catalog:** Access to a comprehensive gardening equipment catalog with descriptions, pricing, stock levels, and reviews.
- **Order Processing:** Handles order placement with stock validation.
- **Review System:** Accepts and processes product reviews from existing customers.
- **ReAct Agent:** Uses iterative reasoning to determine appropriate actions based on email content.

---

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/risk_and_security/retail_agent
```

### Set Up API Keys

If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Run the Workflow

Return to your original terminal, and run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
nat run --config_file examples/risk_and_security/retail_agent/configs/config.yml --input "Email From: john.doe@email.com
Content: I'm interested in learning about your garden trowels. What options do you have available?"
```

The expected output of this run is shown below in the Examples seciont.

---

## Agent Capabilities

The retail agent can handle three main types of customer requests:

1. **Product Inquiries:** Customers can ask about specific products or request information about the entire catalog.
2. **Order Placement:** Customers can place orders for products with automatic stock validation and price calculation.
3. **Review Submission:** Existing customers can submit reviews and ratings for products they have purchased.

---

## Data Structure

### Customer Information

Customers are stored in `data/customers.json` with the following structure:

```json
{
  "id": "CUST001",
  "email": "john.doe@email.com",
  "name": "John Doe",
  "past_orders": [
    {
      "product_id": "PROD001",
      "product_name": "Premium Garden Trowel",
      "quantity": 2,
      "date": "2024-01-15",
      "total": 59.98
    }
  ],
  "total_orders": 3,
  "total_spent": 245.5,
  "past_reviews": [
    {
      "product_id": "PROD001",
      "product_name": "Premium Garden Trowel",
      "rating": 5,
      "review": "Excellent tool! Very durable and comfortable to use."
    }
  ]
}
```

### Product Information

Products are stored in `data/products.json` with the following structure:

```json
{
  "id": "PROD001",
  "name": "Premium Garden Trowel",
  "description": "Professional-grade stainless steel trowel with ergonomic soft-grip handle...",
  "price": 29.99,
  "stock": 45,
  "reviews": [
    {
      "customer_id": "CUST001",
      "customer_name": "John Doe",
      "rating": 5,
      "review": "Excellent tool! Very durable and comfortable to use."
    }
  ]
}
```

### Available Products

The catalog includes 10 gardening products:

- Premium Garden Trowel ($29.99)
- Professional Pruning Shears ($79.99)
- Ergonomic Watering Can ($45.99)
- Heavy-Duty Garden Hoe ($54.99)
- Digital Soil pH Tester ($139.53)
- Premium Garden Gloves ($29.99)
- Stainless Steel Hand Rake ($34.99)
- Garden Tool Set Organizer ($89.99)
- Telescoping Hedge Trimmer ($124.99)
- Compost Bin Starter Kit ($199.99)

---

## Available Tools

The retail agent has access to the following tools:

### Retail Tools Function Group

1. **get_customer_info** - Looks up customer by email address

   - Input: email (string)
   - Output: Customer object with id, name, past orders, total orders, total spent, and past reviews

2. **get_product_info** - Retrieves single product details

   - Input: product_id or product_name (string)
   - Output: Product object with id, name, description, price, stock, and reviews

3. **get_all_products** - Lists all products for comparison

   - Input: None
   - Output: List of all products with basic information

4. **write_review** - Mock function to add a review

   - Input: customer_email (string), product_name (string), rating (integer 1-5), review_text (string)
   - Output: Success message (mock - no actual database update)

5. **send_email** - Mock function to send email response

   - Input: recipient_email (string), content (string), cc (optional string)
   - Output: Success message with email details (mock - no actual email sent)

6. **update_customer_info** - Mock function to update customer order information
   - Input: customer_email (string), product_name (string), quantity (integer)
   - Output: Success message with updated order details (mock - no actual database update)

---

## Example Usage

### Product Inquiry

**Input:**

```bash
nat run --config_file examples/risk_and_security/retail_agent/configs/config.yml --input "Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?"
```

**Expected Output:**
The agent will use the `get_product_info` or `get_all_products` tool to retrieve information about garden trowels and respond with product details including description, price, and stock availability.

```text
Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 0
Number of Function Groups: 1
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 0
Number of TTC Strategies: 0
Number of Authentication Providers: 0
------------------------------
[AGENT]
Agent input: Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?
Agent's thoughts:
Thought: The customer is inquiring about garden trowels, so I should provide them with information about the available products.

Action: retail_tools.get_all_products
Action Input: None

------------------------------
[AGENT]
Calling tools: retail_tools.get_all_products
Tool's input: None
Tool's response:
[{'id': 'PROD001', 'name': 'Premium Garden Trowel', 'description': 'Professional-grade stainless steel trowel with ergonomic soft-grip handle. Perfect for transplanting, digging, and breaking up soil. Rust-resistant and built to last.', 'price': 29.99, 'stock': 45, 'average_rating': 5.0, 'review_count': 1}, {'id': 'PROD002', 'name': 'Professional Pruning Shears', 'description': 'High-carbon steel blades with precision-ground edges for clean cuts. Ergonomic design reduces hand fatigue. Ideal for trimming flowers, shrubs, and small branches up to 3/4 inch diameter.', 'price': 79.99, 'stock': 28, 'average_rating': 4.0, 'review_count': 1}, {'id': 'PROD003', 'name': 'Ergonomic Watering Can', 'description': '2-gallon capacity watering can with balanced design for easy pouring. Features a removable rose head for gentle watering. UV-resistant plastic construction.', 'price': 45.99, 'stock': 62, 'average_rating': 'No ratings yet', 'review_count': 0}, {'id': 'PROD004', 'name': 'Heavy-Duty Garden...(rest of response truncated)
------------------------------
[AGENT]
Agent input: Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?
Agent's thoughts:
Thought: I have the list of available products, and I can see that there are a few products related to gardening, such as trowels, pruning shears, and hoes. I should use the get_product_info tool to find a product that matches the customer's description of a garden trowel.

Action: retail_tools.get_product_info
Action Input: {"product_identifier": "garden trowel"}
------------------------------
[AGENT]
Calling tools: retail_tools.get_product_info
Tool's input: {'product_identifier': 'garden trowel'}
Tool's response:
[{'error': 'No product found with identifier: garden trowel'}]
------------------------------
[AGENT]
Agent input: Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?
Agent's thoughts:
Thought: The customer is inquiring about garden trowels, and I found a product named 'Premium Garden Trowel' in the list of all products. I should provide the customer with information about this product.

Action: retail_tools.get_product_info
Action Input: {"product_identifier": "Premium Garden Trowel"}
------------------------------
[AGENT]
Calling tools: retail_tools.get_product_info
Tool's input: {'product_identifier': 'Premium Garden Trowel'}
Tool's response:
[{'id': 'PROD001', 'name': 'Premium Garden Trowel', 'description': 'Professional-grade stainless steel trowel with ergonomic soft-grip handle. Perfect for transplanting, digging, and breaking up soil. Rust-resistant and built to last.', 'price': 29.99, 'stock': 45, 'reviews': [{'customer_id': 'CUST001', 'customer_name': 'John Doe', 'rating': 5, 'review': 'Excellent tool! Very durable and comfortable to use.'}]}]
------------------------------
[AGENT]
Agent input: Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?
Agent's thoughts:
Thought: I have found a product that matches the customer's description of a garden trowel. I should send an email to the customer with the product information and ask if they would like to place an order.

Action: retail_tools.send_email
Action Input: {"recipient_email": "david.brown@email.com", "content": "Dear Sir, \n\nWe have a Premium Garden Trowel (PROD001) available that might interest you. It is a professional-grade stainless steel trowel with an ergonomic soft-grip handle, perfect for transplanting, digging, and breaking up soil. The price is $29.99, and we currently have 45 in stock. If you would like to place an order, please let me know and I will be happy to assist you. \n\nBest regards, \n[Your Name]"}
------------------------------
[AGENT]
Calling tools: retail_tools.send_email
Tool's input: {'recipient_email': 'david.brown@email.com', 'content': 'Dear Sir, \n\nWe have a Premium Garden Trowel (PROD001) available that might interest you. It is a professional-grade stainless steel trowel with an ergonomic soft-grip handle, perfect for transplanting, digging, and breaking up soil. The price is $29.99, and we currently have 45 in stock. If you would like to place an order, please let me know and I will be happy to assist you. \n\nBest regards, \n[Your Name]'}
Tool's response:
[{'success': True, 'message': 'Email sent successfully', 'email_details': {'to': 'david.brown@email.com', 'cc': 'None', 'content': 'Dear Sir, \n\nWe have a Premium Garden Trowel (PROD001) available that might interest you. It is a professional-grade stainless steel trowel with an ergonomic soft-grip handle, perfect for transplanting, digging, and breaking up soil. The price is $29.99, and we currently have 45 in stock. If you would like to place an order, please let me know and I will be happy to assist you. \n\nBest regards, \n[Your Name]', 'timestamp': '2024-11-25T10:00:00Z'}, 'note': 'This is a mock operation - no actual email was sent.'}]
------------------------------
[AGENT]
Agent input: Email From: david.brown@email.com
Content: Hello, I'm interested in learning about your garden trowels. What do you have available?
Agent's thoughts:
Thought: I now know the final answer

Final Answer: The customer has been sent an email with the product information, and they will be contacted again if they respond with an interest in placing an order. The email details are as follows:
{'to': 'david.brown@email.com',
'cc': 'None',
'content': 'Dear Sir, \n\nWe have a Premium Garden Trowel (PROD001) available that might interest you. It is a professional-grade stainless steel trowel with an ergonomic soft-grip handle, perfect for transplanting, digging, and breaking up soil. The price is $29.99, and we currently have 45 in stock. If you would like to place an order, please let me know and I will be happy to assist you. \n\nBest regards, \n[Your Name]',
'timestamp': '2024-11-25T10:00:00Z'}
------------------------------
Workflow Result:
["The customer has been sent an email with the product information, and they will be contacted again if they respond with an interest in placing an order. The email details are as follows: \n{'to': 'david.brown@email.com', \n'cc': 'None', \n'content': 'Dear Sir, \\n\\nWe have a Premium Garden Trowel (PROD001) available that might interest you. It is a professional-grade stainless steel trowel with an ergonomic soft-grip handle, perfect for transplanting, digging, and breaking up soil. The price is $29.99, and we currently have 45 in stock. If you would like to place an order, please let me know and I will be happy to assist you. \\n\\nBest regards, \\n[Your Name]', \n'timestamp': '2024-11-25T10:00:00Z'}"]
--------------------------------------------------
```

---

### Review Submission

**Input:**

```bash
nat run --config_file examples/risk_and_security/retail_agent/configs/config.yml --input "Email From: john.doe@email.com
Content: I'd like to write a review for the Premium Garden Trowel I purchased. It's fantastic! I give it 5 stars. The stainless steel is very durable and the grip is comfortable."
```

**Expected Output:**
The agent will:

1. Look up the customer information using `get_customer_info`
2. Verify the product exists using `get_product_info`
3. Submit the review using `write_review`
4. Send a confirmation email using `send_email`

The output logs will be similar to these:

```text
[AGENT]
Agent input: Email From: john.doe@email.com
Content: I'd like to write a review for the Premium Garden Trowel I purchased. It's fantastic! I give it 5 stars. The stainless steel is very durable and the grip is comfortable.
Agent's thoughts:
Thought: The customer wants to write a review for the Premium Garden Trowel, so I should use the write_review tool to submit the review.

Action: retail_tools.write_review
Action Input: {"customer_email": "john.doe@email.com", "product_name": "Premium Garden Trowel", "rating": 5, "review_text": "The stainless steel is very durable and the grip is comfortable. It's fantastic!"}

------------------------------
[AGENT]
Calling tools: retail_tools.write_review
Tool's input: {'customer_email': 'john.doe@email.com', 'product_name': 'Premium Garden Trowel', 'rating': 5, 'review_text': "The stainless steel is very durable and the grip is comfortable. It's fantastic!"}
Tool's response:
[{'success': True, 'message': 'Review submitted successfully for Premium Garden Trowel', 'review': {'customer_name': 'John Doe', 'product_name': 'Premium Garden Trowel', 'rating': 5, 'review_text': "The stainless steel is very durable and the grip is comfortable. It's fantastic!"}, 'note': 'This is a mock operation - the review was not actually saved to the database.'}]
------------------------------
[AGENT]
Agent input: Email From: john.doe@email.com
Content: I'd like to write a review for the Premium Garden Trowel I purchased. It's fantastic! I give it 5 stars. The stainless steel is very durable and the grip is comfortable.
Agent's thoughts:
Thought: The review has been submitted successfully, now I should respond to the customer to confirm that their review has been received and thank them for taking the time to write it.

Action: retail_tools.send_email
Action Input: {"recipient_email": "john.doe@email.com", "content": "Dear Sir, thank you for taking the time to write a review for the Premium Garden Trowel. We appreciate your feedback and are glad to hear that you're enjoying the product. Your review has been submitted successfully.", "cc": ""}
------------------------------
[AGENT]
Calling tools: retail_tools.send_email
Tool's input: {'recipient_email': 'john.doe@email.com', 'content': "Dear Sir, thank you for taking the time to write a review for the Premium Garden Trowel. We appreciate your feedback and are glad to hear that you're enjoying the product. Your review has been submitted successfully.", 'cc': ''}
Tool's response:
[{'success': True, 'message': 'Email sent successfully', 'email_details': {'to': 'john.doe@email.com', 'cc': 'None', 'content': "Dear Sir, thank you for taking the time to write a review for the Premium Garden Trowel. We appreciate your feedback and are glad to hear that you're enjoying the product. Your review has been submitted successfully.", 'timestamp': '2024-11-25T10:00:00Z'}, 'note': 'This is a mock operation - no actual email was sent.'}]
------------------------------
[AGENT]
Agent input: Email From: john.doe@email.com
Content: I'd like to write a review for the Premium Garden Trowel I purchased. It's fantastic! I give it 5 stars. The stainless steel is very durable and the grip is comfortable.
Agent's thoughts:
Thought: I now know the final answer
------------------------------
Final Answer: The customer's review has been submitted successfully and an email has been sent to confirm this. The email details are as follows:
{'to': 'john.doe@email.com',
'cc': 'None',
'content': "Dear Sir, thank you for taking the time to write a review for the Premium Garden Trowel. We appreciate your feedback and are glad to hear that you're enjoying the product. Your review has been submitted successfully.",
'timestamp': '2024-11-25T10:00:00Z'}
Workflow Result:
['The customer\'s review has been submitted successfully and an email has been sent to confirm this. The email details are as follows: \n{\'to\': \'john.doe@email.com\', \n\'cc\': \'None\', \n\'content\': "Dear Sir, thank you for taking the time to write a review for the Premium Garden Trowel. We appreciate your feedback and are glad to hear that you\'re enjoying the product. Your review has been submitted successfully.", \n\'timestamp\': \'2024-11-25T10:00:00Z\'}']
```

---

### Order Placement

**Input:**

```bash
nat run --config_file examples/risk_and_security/retail_agent/configs/config.yml --input "Email From: sarah.smith@email.com
Content: I would like to order 2 Ergonomic Watering Cans. Can you process this order and let me know the total cost?"
```

**Expected Output:**
The agent will:

1. Look up customer information using `get_customer_info`
2. Check product details and stock using `get_product_info`
3. Update customer order information using `update_customer_info`
4. Send order confirmation using `send_email`

The output of the run:

```text
[AGENT]
Agent input: Email From: sarah.smith@email.com
Content: I would like to order 2 Ergonomic Watering Cans. Can you process this order and let me know the total cost?
Agent's thoughts:
Thought: The customer wants to order 2 Ergonomic Watering Cans. To confirm the order, I first need to check the product information, including stock and pricing.

Action: retail_tools.get_product_info
Action Input: {"product_identifier": "Ergonomic Watering Can"}

------------------------------
[AGENT]
Calling tools: retail_tools.get_product_info
Tool's input: {'product_identifier': 'Ergonomic Watering Can'}
Tool's response:
[{'id': 'PROD003', 'name': 'Ergonomic Watering Can', 'description': '2-gallon capacity watering can with balanced design for easy pouring. Features a removable rose head for gentle watering. UV-resistant plastic construction.', 'price': 45.99, 'stock': 62, 'reviews': []}]
------------------------------
[AGENT]
Agent input: Email From: sarah.smith@email.com
Content: I would like to order 2 Ergonomic Watering Cans. Can you process this order and let me know the total cost?
Agent's thoughts:
Thought: I have found the product information for the Ergonomic Watering Can, which has a product ID of PROD003, a price of $45.99, and is in stock. Since the customer wants to order 2 units, I will calculate the total cost.

The total cost will be 2 * $45.99 = $91.98. I will now confirm the order details with the customer and update their information.

Action: retail_tools.update_customer_info
Action Input: {"customer_email": "sarah.smith@email.com", "product_name": "PROD003", "quantity": 2}
------------------------------
[AGENT]
Calling tools: retail_tools.update_customer_info
Tool's input: {'customer_email': 'sarah.smith@email.com', 'product_name': 'PROD003', 'quantity': 2}
Tool's response:
[{'success': True, 'message': 'Order placed successfully for Sarah Smith', 'order_details': {'customer_name': 'Sarah Smith', 'customer_email': 'sarah.smith@email.com', 'product_name': 'Ergonomic Watering Can', 'product_id': 'PROD003', 'quantity': 2, 'unit_price': 45.99, 'total': 91.98, 'new_total_orders': 3, 'new_total_spent': 231.95}, 'note': 'This is a mock operation - the order was not actually saved to the database.'}]
------------------------------
[AGENT]
Agent input: Email From: sarah.smith@email.com
Content: I would like to order 2 Ergonomic Watering Cans. Can you process this order and let me know the total cost?
Agent's thoughts:
Thought: The customer's order has been successfully placed. I will now send a confirmation email to the customer with the order details.

Action: retail_tools.send_email
Action Input: {"recipient_email": "sarah.smith@email.com", "content": "Dear Madam, \n\nYour order for 2 Ergonomic Watering Cans (PROD003) has been successfully placed. The total cost is $91.98. \n\nThank you for shopping with us. \n\nBest regards, GreenThumb Gardening Equipment", "cc": ""}
------------------------------
[AGENT]
Calling tools: retail_tools.send_email
Tool's input: {'recipient_email': 'sarah.smith@email.com', 'content': 'Dear Madam, \n\nYour order for 2 Ergonomic Watering Cans (PROD003) has been successfully placed. The total cost is $91.98. \n\nThank you for shopping with us. \n\nBest regards, GreenThumb Gardening Equipment', 'cc': ''}
Tool's response:
[{'success': True, 'message': 'Email sent successfully', 'email_details': {'to': 'sarah.smith@email.com', 'cc': 'None', 'content': 'Dear Madam, \n\nYour order for 2 Ergonomic Watering Cans (PROD003) has been successfully placed. The total cost is $91.98. \n\nThank you for shopping with us. \n\nBest regards, GreenThumb Gardening Equipment', 'timestamp': '2024-11-25T10:00:00Z'}, 'note': 'This is a mock operation - no actual email was sent.'}]
------------------------------
[AGENT]
Agent input: Email From: sarah.smith@email.com
Content: I would like to order 2 Ergonomic Watering Cans. Can you process this order and let me know the total cost?
Agent's thoughts:
Thought: I now know the final answer
------------------------------
Final Answer: The customer, Sarah Smith, has been emailed with the order details. The email_details are as follows:
{'to': 'sarah.smith@email.com',
'cc': 'None',
'content': 'Dear Madam, \n\nYour order for 2 Ergonomic Watering Cans (PROD003) has been successfully placed. The total cost is $91.98. \n\nThank you for shopping with us. \n\nBest regards, GreenThumb Gardening Equipment',
'timestamp': '2024-11-25T10:00:00Z'}
Workflow Result:
["The customer, Sarah Smith, has been emailed with the order details. The email_details are as follows: \n{'to': 'sarah.smith@email.com', \n'cc': 'None', \n'content': 'Dear Madam, \\n\\nYour order for 2 Ergonomic Watering Cans (PROD003) has been successfully placed. The total cost is $91.98. \\n\\nThank you for shopping with us. \\n\\nBest regards, GreenThumb Gardening Equipment', \n'timestamp': '2024-11-25T10:00:00Z'}"]
--------------------------------------------------
```

---

## Evaluation

The retail agent includes an evaluation framework to assess the agent's performance across various customer service scenarios.

### Running the Evaluation

To run the evaluation, execute the following command from the root of the NeMo Agent toolkit repository:

```bash
nat eval --config_file examples/risk_and_security/retail_agent/configs/config-eval.yml
```

The evaluation uses the `tunable_rag_evaluator` to score the agent's responses against expected outcomes. The evaluator uses an LLM-as-a-judge approach to determine how well the agent handles each test case.

### Evaluation Dataset

The evaluation dataset (`data/evalset.json`) contains six test scenarios covering:

1. **Product Inquiry:** Customer asking about available garden trowels
2. **Review Submission:** Customer submitting a product review with rating
3. **Order Placement:** Customer placing an order with quantity and total cost request
4. **Unknown Customer Inquiry:** Handling inquiries from customers not in the system
5. **Implicit Product Reference:** Processing a review where the product is not explicitly named (requires the agent to look up purchase history)
6. **Combined Request:** Handling a review submission and order placement in a single email

Each test case includes:

- `question`: The customer email input
- `answer`: A description of the expected agent behavior and response

### Evaluation Output

The evaluation results are saved to `./.tmp/nat/examples/evaluation_and_profiling/nat_retail_agent/llama-33-70b/` and include:

- **Scores:** Each response is scored on a scale from 0.0 to 1.0
- **Profiler Metrics:** Base performance metrics for workflow execution

The scoring criteria evaluates whether:

- The agent sends an email to the correct recipient
- The agent uses appropriate greetings (Sir/Madam based on context)
- The response contains required information (product details, order confirmation, and so on)
- No inappropriate actions are taken (such as creating orders when only an inquiry was made)

---

## Notes

- **Mock Operations:** The `write_review`, `send_email`, and `update_customer_info` functions are mock operations for demonstration purposes. They do not persist data to the JSON files.
- **Demo Purpose:** This example is designed for demonstration and testing. In a production environment, you would integrate with actual databases and email services.
- **Future Enhancements:** This example serves as a foundation for adding red teaming and defense capabilities to test agent robustness and security.

---

## Architecture

This example uses:

- **ReAct Agent:** Provides iterative reasoning between tool calls to handle complex multi-step customer requests
- **Function Groups:** Organizes related retail tools together for better modularity
- **Custom Functions:** All retail tools are implemented as custom async functions
- **YAML Configuration:** Declarative workflow setup for easy customization
- **JSON Data:** Lightweight file-based "database" simulation
- **Plugin System:** Uses Python entry points for automatic component discovery
