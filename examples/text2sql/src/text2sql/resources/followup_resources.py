"""Follow-up question generation resources and templates.

This module contains the guidelines, use cases, and templates used for generating
contextually relevant follow-up questions in supply chain data analysis.
"""

from text2sql.utils.db_schema import TTYSC_TABLES

FOLLOWUP_GUIDELINES = """
You are a helpful assistant. Your task is to generate follow-up questions \
to help the user better understand their data.

The user previously asked the following `user_prompt`:
\"\"\"{user_prompt}\"\"\"

Here is the `table_preview` of the data returned (first few rows, no \
index):
{table_preview}

CRITICAL CONSTRAINTS:
- You may ONLY suggest follow-up questions that are EXACT variations of \
the supported use cases listed below
- DO NOT create questions involving time-based aggregations (e.g., \
"over the last 6 months", "average over time")
- DO NOT create questions asking for specific dates or date-based analysis \
(e.g., "which dates have largest gaps")
- DO NOT create questions asking for statistical analysis not explicitly \
shown in the use cases
- DO NOT create questions asking for calculations or metrics not present in \
the original use cases

SUPPORTED USE CASES ONLY:
{use_cases_text}

Please generate **four distinct follow-up questions** the user could \
logically ask next,
following these STRICT guidelines:

1.  **Questions 1 and 2** should be similar supported use cases that are \
related to the original `user_prompt`.
        - Find use cases from the SUPPORTED USE CASES list that are similar \
or related to the user's question
        - Use the EXACT wording from the supported use cases - do not modify \
them
        - Choose use cases that involve similar concepts, data types, or \
analysis patterns as the original question
        - Example: if `user_prompt` was "show latest material cost by CM for \
NVPN 316-0899-000"
          Question 1 could be "Show me the latest material cost by CM for a \
given NVPN" (similar supported use case; use an NVPN number)
          Question 2 could be "Show me the latest lead time by CM for a \
given NVPN" (related supported use case; use an NVPN number)

2.  **Question 3** should filter the current `table_preview` using \
patterns from supported use cases.
        - ONLY use filtering patterns that exist in the supported use cases
        - Example: if `table_preview` shows lead times, and a supported use \
case is "Components starting with PCB with lead time > 200"
          Question 3 could be "Show items with lead time > 100" (using the \
lead time filtering pattern)

3.  **Question 4** should be one of the supported use cases from the list \
above.
        - Pick a use case that is contextually relevant to the `user_prompt` \
or `table_preview`
        - Modify the wording to change it into a question or imperative \
format

4.  **VALIDATION RULES:**
    - Each question MUST follow the exact patterns and structures shown in \
the supported use cases
    - Each question MUST be achievable using only the data and operations \
shown in the use cases
    - Present follow-up questions as either questions (e.g., "What is the \
latest material cost by CM for NVPN X?") or imperatives (e.g., "Show \
latest material cost by CM for NVPN X")
    - If you cannot create a valid question following these constraints, use \
a generic supported use case instead
    - Number your questions from 1 to 4.
    - Do not include any commentary, preamble, or explanation. Just provide \
the questions.

FORBIDDEN QUESTION TYPES:
- Time-based aggregations (average over time, trends over specific periods)
- Date-specific analysis (which dates, specific time ranges)
- Complex statistical calculations not in use cases
- Questions that combine multiple metrics in ways not shown in use cases
"""

TABLE_USE_CASES = {
    TTYSC_TABLES.DEMAND_DLT: [
        "Top 20 NVPNs with highest shortages",
        "Inventory and supply trend by CM site or organization",
        "OH, supply, and demand during lead time by CM for a given NVPN",
        "Latest material cost by CM for a given NVPN",
        "Latest lead time by CM for a given NVPN",
        "Components starting with PCB with lead time > 200",
        "On-hand inventory trend for last 12 weeks",
        "Parts with supply gaps from a given SKU",
        "Top 10 items with longest lead time for a given SKU",
        "Supply and demand trend for a given NVPN",
        "Safety stock data for NVPN 316-0899-000",
        "Shortage including safety stock for NVPN 316-0899-000",
        "Excess including safety stock for NVPN 316-0899-000",
        "Supply gaps for NVPN with safety stocks",
        "Excess for NVPN with safety stocks",
        "Shortages within lead time for NVPN 316-0899-000",
        "Shortages within lead time for cm_site_name FBN_NBU",
        "Shortages within lead time for SKU 900-2G548-0001-000",
        "Supply gaps to cover safety stock demand for NVPN 316-0899-000",
        "Items with safety stocks and utilization percentage",
        "Items with safety stocks over utilized more than 50% in last 1 \
month",
        "Items with safety stocks under utilized more than 50% in last 1 \
month",
        "No of weeks safety stocks are over utilized in last 1 month",
        "No of weeks safety stocks are under utilized in last 1 month",
        "Items with safety stocks over utilized more than 50% in last \
quarter",
        "Items with safety stocks under utilized more than 50% in last \
quarter",
        "Open orders that are within lead time for NVPN 316-0899-000",
        "Open orders that are outside of lead time for NVPN 316-0899-000",
    ],
    TTYSC_TABLES.OPEN_ORDERS: [
        "Delivery schedules for on-orders for NVPN MLX000803 and CM FXM_NBU",
        "How many on-orders will be delivered in the current month for NVPN \
MLX000803 and CM FXM_NBU",
        "How many on-orders will be delivered in the next month for NVPN \
MLX000803 and CM FXM_NBU",
        "How many on-orders will be delivered in the current quarter for NVPN \
MLX000803 and CM FXM_NBU",
        "PO line-level information for NVPN MLX000803",
        "PO line-level information for NVPN MLX000803 and CM FXM_NBU",
        "Complete details open order information for NVPN MLX000803",
        "Commit and uncommit status of orders for NVPN MLX000803 and CM \
FXM_NBU",
        "Delivery schedules for manufacturer_pn PCR-S3TW100A-NVD",
        "PO details of OO created in April 2025 for NVPN 078.22513.M011 at \
WIST",
        "All OO to MURATA for NVPN 032-1096-000 at WIST",
        "Total OO, committed OO and uncommitted OO for NVPN \
681-24287-0012.A at WIST",
        "OO trend with committed and uncommitted for NVPN 032-1096-000 at \
WIST",
    ],
    TTYSC_TABLES.ESCALATION_DATA: [
        "Show me the all the L3 escalations",
        "Show me all the escalation for CM Site Wist",
        "Show me all the escalation for CM Site Wist and GPU",
        "How many days have these L3 escalations been open for NVPN \
MA011583",
        "When was the escalation opened for the NVPN MA011583",
        "What is the current status of the escalation for a NVPN IC001843",
    ],
    TTYSC_TABLES.MANU_PN: [
        "Show me the corresponding manufacturer parts for the NVPN \
195-3070-000 and cm_site_name FXLH",
        "Show me the mapped manufacturer parts for the NVPN MA011583",
        "Show me the corresponding NVPN for manufacturer part \
PCR-S3TW100A-NVD",
        "Show all the mapped AVL information for the NVPN 195-3070-000 and \
cm_site_name FXLH",
        "Show manufacturer item attributes of 032-1096-000 at WIST",
        "List all items with manufacturing name, MPN, unit cost, lead time \
from CM WIST",
        "Show all MPN and Manufacturer details with cost, lead time, split \
info for NVPN 316-0899-000",
        "Show latest unit cost and lead time of NVPN 316-0899-000 at CM \
FXLH",
    ],
    TTYSC_TABLES.PBR: [
        "Show me all the components without lead time for build id PB-60506",
        "Show me all the components without lead time for project id P3960",
        "Show all the red ones for build id 59153",
        "Show me all the red items for build id 60507 from April 10 to April \
17",
        "Give me a list of components with insufficient quantity for \
PB-55330",
        "Show all the red and blank ones for turnkeys items for sku \
699-12813-0000-501",
        "Show me for this nvpn 135-0473-000 when the shortage happens",
        "Show all the NVPN parts that are short within the next two weeks \
for supplier Murata Manufacturing",
        "Show items with shortages more than 50 units manufactured by Texas \
Instruments for SKU 699-13925-1099-TS3",
    ],
}
