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


# Personal Finance

<!-- Note: "Agno" is the official product name despite Vale spelling checker warnings -->
Built on [Agno](https://github.com/agno-agi/agno) and NeMo Agent toolkit, this workflow is a personal financial planner that generates personalized financial plans using NVIDIA NIM (can be customized to use OpenAI models). It automates the process of researching, planning, and creating tailored budgets, investment strategies, and savings goals, empowering you to take control of your financial future with ease.

This personal financial planner was revised based on the [Awesome-LLM-App](https://github.com/Shubhamsaboo/awesome-llm-apps) GitHub repo's [AI Personal Finance Planner](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/single_agent_apps/ai_personal_finance_agent) sample.


## Table of Contents

- [Personal Finance](#personal-finance)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
  - [Deployment-Oriented Setup](#deployment-oriented-setup)
    - [Build the Docker Image](#build-the-docker-image)
    - [Run the Docker Container](#run-the-docker-container)
    - [Test the API](#test-the-api)
    - [Expected API Output](#expected-api-output)


## Key Features

- **Agno Framework Integration:** Demonstrates seamless integration between the lightweight Agno multimodal agent library and NeMo Agent toolkit for building sophisticated agent workflows with minimal overhead.
- **Personal Financial Planning Workflow:** Creates personalized financial plans including budgets, investment strategies, and savings goals using NVIDIA NIM models with automated research and planning capabilities.
- **Multi-Framework Agent Architecture:** Shows how to combine Agno's lightning-fast, model-agnostic capabilities with NeMo Agent toolkit workflow management and tool integration system.
- **Automated Financial Research:** Integrates SERP API for real-time financial data gathering and market research to inform personalized financial planning recommendations.
- **Docker-Ready Deployment:** Provides complete containerization setup for deploying personal finance planning agents in production environments with API access.

### Agno

Agno is a lightweight library for building multimodal agents. Some of the key features of Agno include lightning fast, model agnostic, multimodal, multi agent, etc.  See Agno README [here](https://github.com/agno-agi/agno/blob/main/README.md) for more information about the library.


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/frameworks/agno_personal_finance
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
export OPENAI_API_KEY=<YOUR_API_KEY>
export SERP_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

```bash
aiq run --config_file examples/frameworks/agno_personal_finance/configs/config.yml --input "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."
```

**Expected Workflow Result**
```
### Personalized Financial Plan for Early Retirement at Age 60

#### Overview
You are currently 40 years old and working as a Machine Learning engineer at NVIDIA, with a goal to retire at age 60. This gives you 20 years to prepare for retirement. Below is a structured financial plan that includes budgeting, investment strategies, and savings strategies tailored to your situation.

---

### 1. Financial Goals
- **Retirement Age**: 60
- **Time Horizon**: 20 years
- **Desired Retirement Lifestyle**: Comfortable living, travel, and hobbies.

### 2. Current Financial Situation
- **Income**: As a Machine Learning engineer, your income is likely competitive within the tech industry. 
- **Expenses**: Assess your current monthly expenses to identify areas for savings.
- **Savings**: Evaluate your current savings and retirement accounts (e.g., 401(k), RRSP, etc.).

### 3. Suggested Budget
- **Monthly Income**: Calculate your net monthly income after taxes.
- **Expense Categories**:
  - **Housing**: 25-30% of income
  - **Utilities**: 5-10%
  - **Groceries**: 10-15%
  - **Transportation**: 10%
  - **Savings/Investments**: 20-30%
  - **Discretionary Spending**: 10-15%
  
**Example**: If your monthly income is $8,000:
- Housing: $2,000
- Utilities: $600
- Groceries: $1,000
- Transportation: $800
- Savings/Investments: $2,400
- Discretionary: $1,200

### 4. Investment Strategies
Given your background in technology, consider the following investment opportunities:

- **Tech Stocks**: Invest in high-performing tech stocks. For example, check out the [Best-Performing Tech Stocks for July 2025](https://www.nerdwallet.com/article/investing/best-performing-technology-stocks).
- **ETFs and Mutual Funds**: Diversify your portfolio with technology-focused ETFs or mutual funds. Refer to [Ways to Invest in Tech](https://www.investopedia.com/ways-to-invest-in-tech-11745768).
- **Retirement Accounts**: Maximize contributions to your 401(k) or RRSP, especially if your employer offers matching contributions.
- **Alternative Investments**: Explore opportunities in startups or angel investments in the tech sector.

### 5. Savings Strategies
To enhance your retirement savings, consider the following strategies:

- **Start Early**: The earlier you start saving, the more your money can grow. Aim to save at least 20-30% of your income.
- **Emergency Fund**: Maintain an emergency fund covering 3-6 months of living expenses.
- **Debt Management**: Pay off high-interest debts as soon as possible to free up more funds for savings.
- **Automate Savings**: Set up automatic transfers to your savings and investment accounts to ensure consistent contributions.
- **Review and Adjust**: Regularly review your financial plan and adjust your savings rate as your income grows.

### 6. Resources for Further Learning
- **Retirement Planning**: [How to Achieve Early Retirement in Canada](https://nesbittburns.bmo.com/surconmahoneywealthmanagement/blog/693121-How-to-Achieve-Early-Retirement-in-Canada-Proven-Strategies-for-Financial-Independence) provides practical strategies for financial independence.
- **Investment Insights**: [Technology Investments in 2025](https://wezom.com/blog/technology-investments-in-2025) offers insights into key investment areas in technology.
- **Savings Tips**: [10 Tips to Help You Boost Your Retirement Savings](https://www.merrilledge.com/article/10-tips-to-help-you-boost-your-retirement-savings-whatever-your-age-ose) provides actionable advice for enhancing your savings.

---

### Conclusion
By following this personalized financial plan, you can work towards achieving your goal of retiring at age 60. Regularly review your progress, adjust your strategies as needed, and stay informed about market trends and investment opportunities. With discipline and planning, you can secure a comfortable retirement.
```
---

## Deployment-Oriented Setup

For a production deployment, use Docker:

### Build the Docker Image

Prior to building the Docker image ensure that you have followed the steps in the [Installation and Setup](#installation-and-setup) section, and you are currently in the NeMo Agent toolkit virtual environment.

From the root directory of the `aiqtoolkit` repository, build the Docker image:

```bash
docker build --build-arg AIQ_VERSION=$(python -m setuptools_scm) -t agno_personal_finance -f examples/frameworks/agno_personal_finance/Dockerfile .
```

### Run the Docker Container
Deploy the container:

```bash
docker run -p 8000:8000 -e NVIDIA_API_KEY -e OPENAI_API_KEY -e SERP_API_KEY agno_personal_finance
```

### Test the API
Use the following curl command to test the deployed API:

```bash
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs": "My financial goal is to retire at age 60.  I am currently 40 years old, working as a Machine Learning engineer at NVIDIA."}'
```

### Expected API Output
The API response should look like this:

```json
{"value":"### Personalized Financial Plan for Early Retirement at Age 60\n\n#### **Current Situation**\n- **Age**: 40 years old\n- **Occupation**: Machine Learning Engineer at NVIDIA\n- **Goal**: Retire at age 60\n\n#### **Financial Goals**\n1. **Retirement Planning**: Accumulate sufficient wealth to maintain your lifestyle post-retirement.\n2. **Investment Growth**: Maximize returns through strategic investments.\n3. **Tax Efficiency**: Minimize tax liabilities to maximize savings.\n\n#### **Budgeting Strategy**\n- **Monthly Savings Goal**: Aim to save at least 20-25% of your monthly income. This can be adjusted based on your current expenses and lifestyle.\n- **Emergency Fund**: Maintain an emergency fund covering 6-12 months of living expenses to ensure financial security against unforeseen events.\n\n#### **Investment Plan**\n1. **Equity Compensation**: \n   - **Stock Options and RSUs**: As a tech professional, leverage your equity compensation. Consider diversifying these assets to reduce risk ([Retirement Roadmap](https://www.jyacwealth.com/blog/retirement-roadmap-advice-for-tech-industry-professionals)).\n   - **Maximize Stock Value**: Regularly review and adjust your stock portfolio to align with market conditions and personal risk tolerance.\n\n2. **Retirement Accounts**:\n   - **401(k) Contributions**: Maximize contributions to your 401(k) plan, especially if your employer offers matching contributions ([Investopedia](https://www.investopedia.com/retirement/top-retirement-savings-tips-55-to-64-year-olds/)).\n   - **IRA and Roth IRA**: Consider contributing to an IRA or Roth IRA for tax-advantaged growth ([Nasdaq](https://www.nasdaq.com/articles/want-retire-early-here-are-6-best-types-investments)).\n\n3. **Diversified Portfolio**:\n   - **Index Funds**: Invest in low-cost index funds for broad market exposure and long-term growth ([NerdWallet](https://www.nerdwallet.com/article/investing/early-retirement)).\n   - **Real Estate and Bonds**: Consider real estate investments and municipal bonds for additional income streams and diversification.\n\n#### **Savings Strategies**\n- **High-Income Earner Strategies**:\n  - **Tax-Deferred Accounts**: Utilize tax-deferred accounts to reduce taxable income.\n  - **Incorporation and Trusts**: Explore incorporation and family trusts for advanced tax strategies.\n\n#### **Tax Planning**\n- **RRSP Contributions**: If applicable, contribute to a Registered Retirement Savings Plan (RRSP) to lower taxable income.\n- **Charitable Donations**: Consider charitable donations for tax deductions and social impact.\n\n#### **Action Plan**\n1. **Review and Adjust**: Regularly review your financial plan and adjust based on changes in income, expenses, and market conditions.\n2. **Consult Professionals**: Engage with financial advisors to tailor strategies specific to your needs and to stay updated on tax laws and investment opportunities.\n3. **Education and Awareness**: Stay informed about financial trends and opportunities through continuous learning and professional advice.\n\nBy following this comprehensive financial plan, you can strategically work towards your goal of retiring at age 60 while ensuring financial security and growth."}
```
