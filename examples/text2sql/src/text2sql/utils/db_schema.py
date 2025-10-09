from enum import Enum


# TODO(apourhabib): Add all the tables here and to TABLES
class TTYSC_TABLES(str, Enum):
    """Enumeration of available tables in the Talk-to-Supply-Chain.

    This enum defines the table identifiers used throughout the system for
    supply chain data analysis and querying.

    Tables:
        PBR (pbrs_shortage_latest): The Prototype Build Request (PBR) System is used to
            manage the requests, materials planning, QT execution, allocation, and
            delivery logistics for all prototypes built at NVIDIA.

        DEMAND_DLT (demandleadtime): Supply gap analysis table that contains
            demand during lead time from CMs for supply chain analysis.

        OPEN_ORDERS (open_orders): Open orders and delivery schedule data for
            supply chain planning and tracking.

        ESCALATION_DATA (escalation_data): Escalation tracking data for
            supply chain issues and L3 escalations.

        MANU_PN (manu_pn): Manufacturer part number mapping.
    """

    DEMAND_DLT = "demandleadtime"
    PBR = "pbrs_shortage_latest"
    OPEN_ORDERS = "open_orders"
    ESCALATION_DATA = "escalation_data"
    MANU_PN = "manu_pn"


CONCEPTS = """
Carefully review the following NVIDIA specific terminology related to supply chain. If a query contains one or more of the following concepts, please ensure that the associated conditions are incorporated into the tools.
- If the question is for build id, use the column 'job' in the table.
- nvpn: Item, component, or part number.
- mfr: Manufacturer or supplier.
- total_shortage = The total shortage for an nvpn across all the build ids in a CM. In pbrs_shortage_latest table, if you sum up the shortage values for all the build ids for an NVPN in a CM, you get the total_shortage for that NVPN in that CM.
- accumulative_shortage = Another potential indicator of shortage/surplus. Interpretation might depend on calculation method. Compare with shortage and total_shortage. Generally this variable is not of main interest.
- mfr / mfr_pn / cmpn = Identify the supplier and specific part numbers (internal, manufacturer, potentially customer). Note that mfr_pn can contain multiple values and cmpn is often 'None' in the sample.
- nv_planner / materials_pm / cm_mpm / nv_mpm = Identify key personnel responsible for managing the component or providing status updates. These fields are often populated.
- project_code / project_desc = Provides context on which project or build the component is for.
- bom_usage / total_demand / net_demand / allocated / qty = Various quantity fields that relate to requirements, usage per unit, and allocation status.  Attrition in this sample appears to be a quantity.
- action_item / cm_comments / mpm_comments = Provide qualitative context on status and recovery actions. nvfeedback is moslty 'None' in data.
- wopldord = Appears to be a concatenated identifier, possibly hh_pn + job. Further investigation on its exact purpose might be needed.
- po / po_qty / total_po_qty = Provide information on purchase orders placed.
- material_status / action / color_code = Provide current status and required actions for the material.
- help_from_nv / help_from_us = Contain comments, status updates, or references to alternative parts rather than direct requests for assistance.
- The term 'project id' refers to the first part of 'project_desc' before the slash sign. For example, P4139 is the project id for project_desc 'P4139 / 01 / 17 TS2'.
- The column 'consign' indicates if an item is consiged or turnkey. The column color_code also include a category 'Consign'. But use the column consign =='Y' to filter the consigned items. (or consign !='Y' for turnkey items).
- In DEMAND_DLT table,
    - For lead time, always use lead_time_days unless specifically asked for component_lt or shipping_lt.
    - For demand, always use demand_during_lead_time unless asked for any other demand such as demand_26weeks.
    - For contract manufacturing, always use cm_site_name unless asked for organization.
- In PBR table, for demand always use total_demand.
- If asked for new items for PBR table, use the column color_code and look for empty or na values.
- For temporal questions in PBR (e.g., "next week", "this month", "in two weeks") without a specific date column mentioned, use 'priority_start_date' as the default temporal reference.
"""

TABLES = [
    {
        "name": TTYSC_TABLES.PBR.value,
        "description": "The Prototype Build Request (PBR) System is used to manage the requests, materials planning, QT execution, allocation, and delivery logistics for all prototypes built at NVIDIA. This table contains detailed updates about components and pbr shortage in the supply chain.",
        "schema": [
            {
                "field": "file_date",
                "type": "int",
                "description": "Date the data snapshot or report was generated (e.g., '2025-05-12'). Format YYYY-MM-DD.",
            },
            {
                "field": "action_item",
                "type": "nvarchar(256)",
                "description": "Description of the action required or being taken (e.g., 'None', 'BOM ALT 300-0657-000HF', 'BOM ALT 300-0906-000HF;300-1187-000HF').",
            },
            {
                "field": "allocated",
                "type": "int",
                "description": "Quantity of a component allocated for a specific purpose for a given build_id or job.",
            },
            {
                "field": "bom_usage",
                "type": "int",
                "description": "Quantity of the component used per unit of the parent assembly/BOM.",
            },
            {
                "field": "cm_site_name",
                "type": "nvarchar(256)",
                "description": "Code representing the specific contract manufacturing site (CM, or cm_site_name). e.g., 'FXQT', 'FXSJ', 'WIST', 'FXHC', 'BYD', 'INSJ', 'FXVN'. 'FXLH' is a site of the organization 'FOXCONN'.",
            },
            {
                "field": "consign",
                "type": "nvarchar(256)",
                "description": "Indicator for consign inventory. Can be 'Y' (consigned), or 'None' (potentially 'turnkey' or not specified).",
            },
            {
                "field": "description",
                "type": "nvarchar(256)",
                "description": "Text describing the material or component.",
            },
            {
                "field": "nvpn",
                "type": "nvarchar(256)",
                "description": "NVIDIA part number for the component (e.g., '320-0965-000', '174-0180-000'). Terms 'item', 'part', or 'part number' refers to NVPN.",
            },
            {
                "field": "nv_planner",
                "type": "nvarchar(256)",
                "description": "Name or ID of the Nvidia planner responsible for this component.",
            },
            {
                "field": "color_code",
                "type": "nvarchar(256)",
                "description": "Status or categorization indicator (e.g., 'Y', 'R', 'G', 'Consign'). (G=Green, R=Red, Y=Yellow, other codes like  D, K, H, A, T also present.). If empty it is a new part. The column 'color_code' also include a category 'Consign' but is not for consign check. Instead, use the column consign = 'Y' to filter the consigned items. If asked for new items or parts, use `(color_code IS NULL OR color_code = '')`",
            },
            {
                "field": "job",
                "type": "nvarchar(256)",
                "description": "Job number or identifier, also referred to as Build ID (e.g., 'PB-61738', 'DA61577', 'QS-57344'). It usually starts with 'PB'.",
            },
            {
                "field": "lt_weeks",
                "type": "int",
                "description": "Lead time (LT) in weeks. If lead time is 0, 99, 999, or missing it means the items does not have a lead time.",
            },
            {
                "field": "mfr",
                "type": "nvarchar(256)",
                "description": "Name(s) of the manufacturer(s) of the component. Can contain multiple names separated by commas or semicolons (e.g., 'COOLER,PEM,PENN,Penn Engineering &amp; Man...', 'FUJITSU,Fujitsu Limited'). Use `UPPER(mfr) LIKE '%MFR_NAME%'` to filter for specific manufacturers.",
            },
            {
                "field": "mfr_pn",
                "type": "nvarchar(256)",
                "description": "Manufacturer's part number(s) for the component. Can contain multiple PNs separated by commas or semicolons (e.g., 'CC-03743-01-GP3,SMTSO-M3-3ET,SMTSO-M3-3ET', 'MB85RC512TPNF-G-JNERE1,MB85RC512TPNF-G-JNERE1'). Use 'LIKE' operator to filter for specific patterns.",
            },
            {
                "field": "cmpn",
                "type": "nvarchar(256)",
                "description": "Component Part Number from Contract Manufacturer (CM). Often observed as 'None' in the data, but when present, can contain multiple PNs separated by commas (e.g., '13318529-00,13607684-00').",
            },
            {
                "field": "hh_pn",
                "type": "nvarchar(256)",
                "description": "Hon Hai (Foxconn) Part Number. Appears to be the nvpn or cmpn with an 'HF' suffix (e.g., '320-0965-000HF', '174-0180-000HF'). The HF means Halogen free or their designation that it's QT inventory versus NF for MP.",
            },
            {
                "field": "material_status",
                "type": "nvarchar(256)",
                "description": "Text description of the material's status (e.g., 'Transfer from QT Inventory', 'Gated on NV Approval (R)', 'Will purchase from disty (R)').",
            },
            {
                "field": "nv_bom",
                "type": "nvarchar(256)",
                "description": "Nvidia BOM identifier (e.g., '699-12425-1099-TS5', '699-13921-1099-TS3'). This column is the same as SKU or NV SKU.",
            },
            {
                "field": "po_qty",
                "type": "int",
                "description": "Corresponding Purchase Order quantity/quantities.",
            },
            {
                "field": "project_desc",
                "type": "nvarchar(256)",
                "description": "Description of the project or build phase (e.g., 'E2425 / 0000 / 30 TS5', 'E3921 / 1099 / 13 TS3').",
            },
            {
                "field": "priority_start_date",
                "type": "nvarchar(256)",
                "description": "Date indicating when the material is needed or prioritized (e.g., '2025-06-10'). Format YYYY-MM-DD. It indicates the start of the project cycle. PSD refers to priority_start_date.",
            },
            {
                "field": "shortage",
                "type": "int",
                "description": "Shortage for the specific line item/build. Positive means there is a shortage in pbrs_shortage_latest table. This is the difference between 'total_demand' and 'allocated'. Use this for assessing criticality.",
            },
            {
                "field": "supplieretadate",
                "type": "nvarchar(256)",
                "description": "Estimated Time of Arrival date provided by the supplier. Format YYYY-MM-DD. Often 'NaT' (missing). It's also called Material ETA. Crucial for planning when available, but frequently missing. Analysis may need to focus on items with an ETA or rely on lt_weeks / priority_start_date.",
            },
            {
                "field": "total_demand",
                "type": "int",
                "description": "Calculated overall required quantity for the nvpn across all builds/jobs for a given CM.",
            },
            {
                "field": "total_po_qty",
                "type": "int",
                "description": "Total quantity ordered via Purchase Orders (PO).",
            },
            {
                "field": "total_shortage",
                "type": "int",
                "description": "Calculated overall shortage quantity for the nvpn across all builds/jobs for a given CM.",
            },
            {
                "field": "rev",
                "type": "nvarchar(256)",
                "description": "Revision level of the part or BOM (e.g., 'G', 'D', 'C', 'K', 'F').",
            },
            {
                "field": "priority_start_date",
                "type": "nvarchar(256)",
                "description": "Date indicating when the material is needed or prioritized (e.g., '2025-06-10'). Format YYYY-MM-DD. It indicates the start of the project cycle, the urgency, or need date for the component.",
            },
            {
                "field": "da_released_date",
                "type": "nvarchar(256)",
                "description": "Date related to 'DA' field, possibly Design/Document Approval release date? Mostly observed as 'NaT' (Not a Time) in the data, suggesting it's often missing or not applicable.",
            },
            {
                "field": "licenseno",
                "type": "nvarchar(256)",
                "description": "License number, potentially for export/import. Can contain numeric values (e.g., '1317', '3304').",
            },
        ],
    },
    {
        "name": TTYSC_TABLES.DEMAND_DLT.value,
        "description": "The Demand Lead Time (DLT) table contains supply gap analysis data from CMs for supply chain analysis. This table contains demand during lead time for supply chain analysis.",
        "schema": [
            {
                "field": "date",
                "type": "nvarchar(256)",
                "description": "Date of the data snapshot or report (e.g., '2025-05-12'). Format YYYY-MM-DD.",
            },
            {
                "field": "cm_site_name",
                "type": "nvarchar(256)",
                "description": "Code representing the specific contract manufacturing site (CM, or cm_site_name). e.g., 'FXQT', 'FXSJ', 'WIST', 'FXHC', 'BYD', 'INSJ', 'FXVN'. 'FXLH' is a site of the organization 'FOXCONN'.",
            },
            {
                "field": "component_lt",
                "type": "int",
                "description": "Component lead time (e.g., 1000).",
            },
            {
                "field": "demand_26weeks",
                "type": "int",
                "description": "Demand quantity in next 26 weeks (e.g., 1000).",
            },
            {
                "field": "excess_incl_safety_stock",
                "type": "int",
                "description": "Excess quantity including safety stock.",
            },
            {
                "field": "gpu_nbu",
                "type": "nvarchar(256)",
                "description": "GPU or NBU (e.g., 'GPU', 'NBU').",
            },
            {
                "field": "lead_time_days",
                "type": "int",
                "description": "Lead time in days (e.g., 1000).",
            },
            {
                "field": "material_cost",
                "type": "double",
                "description": "Unit price/cost of item. (e.g., 0.02485).",
            },
            {
                "field": "nettable_oh_inventory",
                "type": "int",
                "description": "Nettable on-hand inventory (e.g., 1000).",
            },
            {
                "field": "nv_sku",
                "type": "nvarchar(256)",
                "description": "NVIDIA SKU (e.g., '900-2G108-2500-000').",
            },
            {
                "field": "nvpn",
                "type": "nvarchar(256)",
                "description": "NVIDIA part number of the component (e.g., '316-0899-000').",
            },
            {
                "field": "organization",
                "type": "nvarchar(256)",
                "description": "The name of the Contract Manufacturer (e.g., 'FOXCONN', 'BYD').",
            },
            {
                "field": "qt_mp",
                "type": "nvarchar(256)",
                "description": "Quick Turn or Mass Production (e.g., 'QT', 'MP').",
            },
            {
                "field": "supply",
                "type": "int",
                "description": "Supply quantity (e.g., 1000).",
            },
            {
                "field": "demand_during_lead_time",
                "type": "int",
                "description": "Demand quantity during lead time (e.g., 1000). It equals aggregated demand during Component LT (component_lt) + Shipping LT (shipping_lt)",
            },
            {
                "field": "shortage",
                "type": "int",
                "description": "Shortage quantity. Positive means there is a shortage in demandleadtime table.",
            },
            {
                "field": "shortage_incl_lead_time",
                "type": "int",
                "description": "The formula is `(DDLT) - (OH+OO with LT)` which means it's demand during lead time minus On-hand plus On-orders that will be delivered during the lead time.",
            },
            {
                "field": "excess",
                "type": "int",
                "description": "Positive means there is excess in demandleadtime table.",
            },
            {
                "field": "safety_stock",
                "type": "int",
                "description": "Safety stock quantity.",
            },
            {
                "field": "supply_depletion_date",
                "type": "nvarchar(256)",
                "description": "Supply depletion date (e.g., '2025-06-10'). Format YYYY-MM-DD.",
            },
            {
                "field": "tier_a_c",
                "type": "nvarchar(256)",
                "description": "Tier A or C (e.g., 'A', 'C').",
            },
            {
                "field": "consigned",
                "type": "nvarchar(256)",
                "description": "Consigned or not (e.g., 'Yes', 'No').",
            },
            {
                "field": "on_order_remaining",
                "type": "int",
                "description": "On-order remaining quantity.",
            },
            {
                "field": "on_order_remaining_total",
                "type": "int",
                "description": "On-order remaining total quantity.",
            },
        ],
    },
]

PBR_EXAMPLES = [
    {
        "Query": "Show all consigned parts for build id PB-61738 that are green.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE job = 'PB-61738' AND color_code = 'G' AND consign = 'Y'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show all turnkey items for build id PB-61738 that are red.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE job = 'PB-61738' AND color_code = 'R' AND ((consign <> 'Y') OR (consign IS NULL))",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "What are the top 3 sites with the most components for project E2425?",
        "SQL": "SELECT cm_site_name, COUNT(DISTINCT nvpn) AS component_count FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE REGEXP_SUBSTR(project_desc, '^[^ ]+') = 'E2425' GROUP BY cm_site_name ORDER BY component_count DESC LIMIT 3",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me all parts from Murata manufacturer that have no lead time information.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(mfr) LIKE '%MURATA%' AND (lt_weeks IS NULL OR lt_weeks = 0 OR lt_weeks = 99 OR lt_weeks = 999)",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Which components for SKU 900-2G108-2500-000 have zero allocated quantity?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nv_bom = '900-2G108-2500-000' AND allocated = 0",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me components by planner for all Intel parts with positive demand.",
        "SQL": "SELECT nv_planner, COUNT(DISTINCT nvpn) AS part_count, SUM(total_demand) AS total_demand_sum FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(mfr) LIKE '%INTEL%' AND total_demand > 0 GROUP BY nv_planner ORDER BY total_demand_sum DESC",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me all the turnkey parts that have shortages with supplier ETA dates this month.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE shortage > 0 AND ((consign <> 'Y') OR (consign IS NULL)) AND supplieretadate BETWEEN DATE_TRUNC('MONTH', CURRENT_DATE) AND LAST_DAY(CURRENT_DATE)",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "List all components for project P4139 where the description contains 'Conn'.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE REGEXP_SUBSTR(project_desc, '^[^ ]+') = 'P4139' AND UPPER(description) LIKE '%CONN%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me components that have Hon Hai part numbers.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE hh_pn IS NOT NULL AND hh_pn != '' AND hh_pn != 'None'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me top ten shortages for the PBR builds in two weeks.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE shortage > 0 and priority_start_date BETWEEN CURRENT_DATE AND DATEADD(WEEK, 2, CURRENT_DATE) ORDER BY shortage DESC LIMIT 10",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Which revision levels are used for NVPN 316-0899-000?",
        "SQL": "SELECT rev, COUNT(*) AS usage_count, cm_site_name FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn = '316-0899-000' AND rev IS NOT NULL AND rev != '' GROUP BY rev, cm_site_name ORDER BY usage_count DESC",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Display all non-consigned items for build id 64591 that are marked as red.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE job = 'PB-64591' AND color_code = 'R' AND ((consign <> 'Y') OR (consign IS NULL))",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "List the top 5 vendors contribute most to material shortage in project id X9900?",
        "SQL": "SELECT mfr, SUM(shortage) AS total_shortage_quantity FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE REGEXP_SUBSTR(project_desc, '^[^ ]+') = 'X9900' AND shortage > 0 GROUP BY mfr ORDER BY total_shortage_quantity DESC LIMIT 5",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Calculate  total shortage by CM site for broadcom supplied components.",
        "SQL": "SELECT cm_site_name, SUM(shortage) AS aggregate_shortage FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(mfr) LIKE '%BROADCOM%' AND shortage > 0 GROUP BY cm_site_name",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "In which builds and locations are parts like 168-XXXX-000 utilized?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn LIKE '168-%-000'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Locate items where manufacturer part number contains QRS789.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(mfr_pn) LIKE '%QRS789%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show new items at FXHC.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE (color_code IS NULL OR color_code = '') AND cm_site_name = 'FXHC'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Which items under mthomas planner have shortage more than 8000?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(nv_planner) = 'MTHOMAS' AND shortage > 8000",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Find components mentioning design alternatives in comments for build id 77123.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(action_item) LIKE '%DESIGN ALT%' AND job = 'PB-77123'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show shortage inventory for red items at BYD.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE color_code = 'R' AND cm_site_name = 'BYD' AND shortage > 0",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Provide shortage analysis by location for all QUALCOMM components.",
        "SQL": "SELECT cm_site_name, SUM(shortage) AS total_shortage_quantity FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(mfr) LIKE '%QUALCOMM%' AND shortage > 0 GROUP BY cm_site_name",
        "metadata": {"analysis": "pbr"},
    },
    {
        # TODO(jiaxiangr): fix after shortage is set to positive
        "Query": "List individual shortage records for all red items at BYD facility.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE color_code = 'R' AND cm_site_name = 'BYD' AND shortage > 0",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "What builds and sites utilize component 422-7766-000?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn = '422-7766-000'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Search for parts matching supplier part number JKL456MNO789.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE mfr_pn LIKE '%JKL456MNO789%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show recently introduced items at WISTRON CM site.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE (color_code IS NULL OR color_code = '') AND cm_site_name = 'WISTRON'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Find items with bom usage above 1500 at INSJ facility.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE bom_usage > 1500 AND cm_site_name = 'INSJ' ORDER BY bom_usage DESC",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Locate items with manufacturing change notes for build 99887.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(action_item) LIKE '%MFG CHANGE%' AND job = 'PB-99887'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "What sites utilize components that start with 115?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn LIKE '115%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        # TODO(jiaxiangr): need normalize `order by` manner
        "Query": "Show components with  status 'Awaiting Engineering Review'.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(material_status) LIKE '%AWAITING ENGINEERING REVIEW%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "List parts with a material status of 'Stock enough'.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(material_status) LIKE '%STOCK ENOUGH%'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Display new components introduced at BYD.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE (color_code IS NULL OR color_code = '') AND cm_site_name = 'BYD'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Which parts does nv planner rchen manage with shortage over 12000?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(nv_planner) = 'RCHEN' AND shortage > 12000",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Find parts with supplier updates noted for build id PB-77665.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE UPPER(action_item) LIKE '%SUPPLIER UPDATE%' AND job = 'PB-77665'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Compare total requirements for item 088-5511-000 across manufacturing sites.",
        "SQL": "SELECT cm_site_name, SUM(total_demand) AS total_demand_quantity FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn = '088-5511-000' GROUP BY cm_site_name",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "How does total demand for sku 622-33445-8800-TS1 compare between INSJ and FXHC locations?",
        "SQL": "SELECT cm_site_name, SUM(total_demand) AS total_demand_quantity FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nv_bom = '622-33445-8800-TS1' AND cm_site_name IN ('INSJ', 'FXHC') GROUP BY cm_site_name",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show all the parts that start with 115 and used in build id 65582?",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn LIKE '115%' AND job = 'PB-65582'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show all the parts with lead time longer than 5 weeks at FXHC",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE cm_site_name = 'FXHC' AND lt_weeks > 5",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show in which projects and builds this item 115-1124-000 is used.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE nvpn = '115-1124-000'",
        "metadata": {"analysis": "pbr"},
    },
    {
        "Query": "Show me all the consigend shortages for the following two weeks at FXSJ.",
        "SQL": "SELECT cm_site_name, nvpn, nv_bom, job, description, project_desc, consign, mfr, mfr_pn, shortage, total_shortage, priority_start_date, lt_weeks, total_demand, allocated, color_code, total_po_qty FROM hive_metastore.silver_global_supply.pbrs_shortage_latest WHERE cm_site_name = 'FXSJ' AND consign = 'Y' AND shortage > 0 AND priority_start_date BETWEEN CURRENT_DATE AND DATEADD(WEEK, 2, CURRENT_DATE)",
        "metadata": {"analysis": "pbr"},
    },
]

DEMAND_DLT_EXAMPLES = [
    {
        "Query": "Show me the latest 25 NVPNs with the highest excess inventory quantities?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, excess FROM hive_metastore.gold_global_supply.demandleadtime)) SELECT date, cm_site_name, nvpn, qt_mp, excess FROM OrderedData WHERE rnk = 1 AND excess > 0 ORDER BY excess DESC LIMIT 25",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display latest supply and safety stock information for NVPN 078-22513-M011, segmented by organization.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY organization ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, organization, cm_site_name, nvpn, qt_mp, supply, safety_stock, nettable_oh_inventory FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '078-22513-M011')) SELECT date, organization, cm_site_name, qt_mp, SUM(supply) AS supply, SUM(safety_stock) AS safety_stock, SUM(nettable_oh_inventory) AS nettable_oh_inventory FROM OrderedData WHERE rnk = 1 GROUP BY date, organization, cm_site_name, qt_mp",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "What is the latest component lead time by organization for NVPN 032-1096-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY organization ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, organization, nvpn, qt_mp, lead_time_days FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '032-1096-000')) SELECT date, organization, nvpn, qt_mp, lead_time_days FROM OrderedData WHERE rnk = 1;",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show me the most recent material costs for Mass Production items above $10?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, material_cost FROM hive_metastore.gold_global_supply.demandleadtime WHERE qt_mp = 'MP')) SELECT date, cm_site_name, nvpn, qt_mp, printf('$%,.5f', material_cost) AS material_cost FROM OrderedData WHERE rnk = 1 AND material_cost > 10.0 ORDER BY material_cost DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "List all QT components with on-order remaining quantities under SKU 900-2G200-4500-000.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, nv_sku, on_order_remaining, on_order_remaining_total FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G200-4500-000' AND qt_mp = 'QT' AND on_order_remaining > 0)) SELECT date, cm_site_name, nvpn, qt_mp, on_order_remaining, on_order_remaining_total FROM OrderedData WHERE rnk = 1 ORDER BY on_order_remaining DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show me all components starting with '032' that have component lead time greater than 50 days.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM hive_metastore.gold_global_supply.demandleadtime)) SELECT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM OrderedData WHERE rnk = 1 AND lead_time_days > 50 AND nvpn LIKE '032%'",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display the top 12 components with the shortest lead times for Quick Turn items.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM hive_metastore.gold_global_supply.demandleadtime WHERE qt_mp = 'QT')) SELECT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM OrderedData WHERE rnk = 1 LIMIT 12",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show latest demand in next 26 weeks for NVPN 681-24287-0012.A by CM site.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, demand_26weeks, demand_during_lead_time FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '681-24287-0012.A')) SELECT date, cm_site_name, nvpn, qt_mp, demand_26weeks, demand_during_lead_time FROM OrderedData WHERE rnk = 1 ORDER BY demand_26weeks DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "List all consigned components with excess including safety stock.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, consigned, excess_incl_safety_stock, safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE consigned = 'Yes' AND excess_incl_safety_stock > 0)) SELECT date, cm_site_name, nvpn, qt_mp, excess_incl_safety_stock, safety_stock FROM OrderedData WHERE rnk = 1 ORDER BY excess_incl_safety_stock DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show components starting with '195' that have nettable inventory above 1000 units.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory FROM hive_metastore.gold_global_supply.demandleadtime)) SELECT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory FROM OrderedData WHERE rnk = 1 AND nettable_oh_inventory > 1000 AND nvpn LIKE '195%' ORDER BY nettable_oh_inventory DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display components by tier classification with supply depletion dates.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, tier_a_c, supply_depletion_date FROM hive_metastore.gold_global_supply.demandleadtime WHERE supply_depletion_date IS NOT NULL)) SELECT date, cm_site_name, nvpn, qt_mp, tier_a_c, supply_depletion_date FROM OrderedData WHERE rnk = 1 ORDER BY supply_depletion_date ASC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show latest shortage data for FOXCONN organization components.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY organization ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, organization, cm_site_name, nvpn, qt_mp, shortage, supply, demand_during_lead_time FROM hive_metastore.gold_global_supply.demandleadtime WHERE organization = 'FOXCONN' AND shortage > 0)) SELECT date, organization, cm_site_name, nvpn, qt_mp, shortage, supply, demand_during_lead_time FROM OrderedData WHERE rnk = 1 ORDER BY shortage DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Find components with shortage including lead time between 100 and 1000 units.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage_incl_lead_time FROM hive_metastore.gold_global_supply.demandleadtime WHERE shortage_incl_lead_time BETWEEN 100 AND 1000)) SELECT date, cm_site_name, nvpn, qt_mp, shortage_incl_lead_time FROM OrderedData WHERE rnk = 1 ORDER BY shortage_incl_lead_time DESC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show components with supply depletion dates in the next quarter.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, supply_depletion_date FROM hive_metastore.gold_global_supply.demandleadtime WHERE supply_depletion_date BETWEEN CURRENT_DATE AND DATEADD(MONTH, 3, CURRENT_DATE))) SELECT date, cm_site_name, nvpn, qt_mp, supply_depletion_date FROM OrderedData WHERE rnk = 1 ORDER BY supply_depletion_date ASC",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display current stock levels and availability for component 455-7788-000, aggregated by facility.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, nettable_oh_inventory FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '455-7788-000')) SELECT date, cm_site_name, qt_mp, SUM(supply) AS supply, SUM(demand_during_lead_time) AS demand_during_lead_time, SUM(nettable_oh_inventory) AS nettable_oh_inventory FROM OrderedData WHERE rnk = 1 GROUP BY date, cm_site_name, qt_mp",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Identify supply shortfalls for product family 800-3K999-1500-000.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, demand_during_lead_time, supply, shortage FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '800-3K999-1500-000' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, shortage FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Find 12 parts with most critical supply deficits right now.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, shortage FROM hive_metastore.gold_global_supply.demandleadtime)) SELECT date, cm_site_name, nvpn, qt_mp, shortage FROM OrderedData WHERE rnk = 1 AND shortage > 0 ORDER BY shortage DESC LIMIT 12",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "What are current procurement cycle times by location for item 588-9944-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '588-9944-000')) SELECT date, cm_site_name, nvpn, qt_mp, lead_time_days FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Provide shortages including safety stock, for 316-0899-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "For component 316-0899-000, get the shortages including safety stock?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display shortages with safety stock for item 316-0899-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "What is the safety stock for NVPN 316-0819-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0819-000')) SELECT date, cm_site_name, nvpn, qt_mp, safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show the latest safety stock for 316-0819-000 for different sites?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0819-000')) SELECT date, cm_site_name, nvpn, qt_mp, safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Get Safety Stock data for 316-0819-000.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0819-000')) SELECT date, cm_site_name, nvpn, qt_mp, safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show shortages and safety stocks.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE safety_stock > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "For different sites, get shortages with safety stock.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE safety_stock > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show supply gaps and safety stocks for different NVPNs and CMs.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE safety_stock > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Which parts for 900-2G548-0001-000 currently have supply gaps?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, demand_during_lead_time, supply, shortage FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, shortage FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show supply gaps for components belongs to 900-2G548-0001-000 and FXHC",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, demand_during_lead_time, supply, shortage FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000' AND cm_site_name  = 'FXHC' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, shortage FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Get shortage items for 900-2G548-0001-000, only more than 10000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, demand_during_lead_time, supply, shortage FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000' AND shortage > 1000)) SELECT date, cm_site_name, nvpn, qt_mp, supply, demand_during_lead_time, shortage FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show me the supply and demand trend for 315-1157-000 and FXHC",
        "SQL": "WITH DeduplicatedData AS (SELECT DISTINCT date, supply, demand_during_lead_time FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '315-1157-000' and cm_site_name = 'FXHC') SELECT date, SUM(supply) AS supply, SUM(demand_during_lead_time) AS demand_during_lead_time FROM DeduplicatedData GROUP BY date ORDER BY date",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Plot supply vs demand over time for NVPN 315-1157-000 and SKU 690-2G535-0220-T00",
        "SQL": "WITH DeduplicatedData AS (SELECT DISTINCT date, supply, demand_during_lead_time FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '315-1157-000' AND nv_sku = '690-2G535-0220-T00') SELECT date, SUM(supply) AS supply, SUM(demand_during_lead_time) AS demand_during_lead_time FROM DeduplicatedData GROUP BY date ORDER BY date",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "supply trend for NVPN 315-1157-000 for the past 16 weeks.",
        "SQL": "WITH DeduplicatedData AS (SELECT DISTINCT date, supply FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '315-1157-000') SELECT date, SUM(supply) AS supply FROM DeduplicatedData WHERE date BETWEEN DATEADD(WEEK, -16, CURRENT_DATE) AND CURRENT_DATE GROUP BY date ORDER BY date",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "What are the lead-time gaps for SKU 900-2G548-0001-000?",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000')) SELECT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show lead-time shortages for 900-2G548-0001-000.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000')) SELECT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Get shortages within lead time for 900-2G548-0001-000 by ietms and sites.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nv_sku = '900-2G548-0001-000')) SELECT date, cm_site_name, nv_sku, nvpn, qt_mp, shortage, safety_stock, shortage_incl_lead_time_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Get shortages for 316-0899-000 after applying safety stock.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show supply gaps for 316-0899-000 by CM site, including safety stock.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display latest supply gaps with safety stock for 316-0899-000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000' AND shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, shortage, safety_stock, shortage_incl_safety_stock FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show open POs that are within lead time for 316-0819-000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "For item 316-0819-000, get OO within lead time",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show open orders within lead time and shortage > 0 for 316-0899-000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000' and shortage > 0)) SELECT date, cm_site_name, nvpn, qt_mp, on_order_remaining FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show open POs outside lead time for NVPN 316-0899-000",
        "SQL": "WITH OrderedData AS (SELECT *, (on_order_remaining_total - on_order_remaining) AS on_order_outside_lead_time, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining_total, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, on_order_outside_lead_time FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Get open orders beyond LT for 316-0899-000 for MP",
        "SQL": "WITH OrderedData AS (SELECT *, (on_order_remaining_total - on_order_remaining) AS on_order_outside_lead_time, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining_total, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000' and qt_mp = 'MP')) SELECT date, cm_site_name, nvpn, qt_mp, on_order_outside_lead_time FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display OO arriving after the LT window for 316-0899-000 for different sites",
        "SQL": "WITH OrderedData AS (SELECT *, (on_order_remaining_total - on_order_remaining) AS on_order_outside_lead_time, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, on_order_remaining_total, on_order_remaining FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '316-0899-000')) SELECT date, cm_site_name, nvpn, qt_mp, on_order_outside_lead_time FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "When does the supply is getting consumed for item 220-0227-000.",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '220-0227-000')) SELECT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Show for differnet sites, until when the supply will last for NVPN 220-0227-000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '220-0227-000')) SELECT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
    {
        "Query": "Display earliest on hand and on order usage for 220-0227-000",
        "SQL": "WITH OrderedData AS (SELECT *, RANK() OVER (PARTITION BY cm_site_name ORDER BY date DESC) AS rnk FROM (SELECT DISTINCT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM hive_metastore.gold_global_supply.demandleadtime WHERE nvpn = '220-0227-000')) SELECT date, cm_site_name, nvpn, qt_mp, nettable_oh_inventory, on_order_remaining, demand_during_lead_time, supply_depletion_date FROM OrderedData WHERE rnk = 1",
        "metadata": {"analysis": "supply_gap"},
    },
]

GUIDELINES = """
Please follow the following guidelines strictly.
    1.	Generate Valid SQL:
         • Ensure SQL syntax is correct for Databricks.
         • Ensure that the column name exists in the table it is referenced from.
         • For queries involving joins carefully review that the correct aliases are used when referencing columns.
         • Columns used in `ORDER BY` or `GROUP BY` should also appear in the columns in the SELECT clause.
         • The schema is case-sensitive. When handling quoted text, apply case conversion as follows:
            - If the quoted text contains only lowercase characters, use LOWER() for case conversion and matching.
            - For all other cases (mixed or uppercase), use UPPER() instead.
         • If the query includes words such as "containing", "like", "mentioning", or "including", translate it using a LIKE '%text_to_match%' clause with the appropriate case conversion (UPPER() or LOWER()).
            - Example:
                - "action items mentioning 'ALT'" → "UPPER(action_item) LIKE '%ALT%'
                - "action items mentioning 'alternative'" → "LOWER(action_item) LIKE '%alternative%'".
	2.	Retrieve columns required to directly answer the query:
         • Add columns only if they are explicitly mentioned in the query.
            - Do not add extra columns unless the query absolutely mentioned it.
         • Map query values to columns based on schema descriptions and predefined mappings:
            Examples:
                - 'parts' or 'components' or 'items' → nvpn (NVIDIA part number)
                - 'manufacturing sites' or 'CM sites' → cm_site_name
                - 'inventory shortages' or 'supply gaps' → shortage
                - 'excess inventory' → excess
                - 'on-hand inventory' or 'in stock' → nettable_oh_inventory
                - 'Quick Turn' → qt_mp = 'QT'
                - 'Mass Production' → qt_mp = 'MP'
                - 'consigned stock' → consigned = 'consigned'
                - 'turnkey stock' → consigned = 'turnkey'
         • Use appropriate date filtering for time-based queries:
            - Syntax: DATEADD(unit, value, CURRENT_DATE)
            - 15 days from today: DATEADD(DAY, 15, CURRENT_DATE)
            - 6 months from today: DATEADD(MONTH, 6, CURRENT_DATE)
            - 2 weeks ago: DATEADD(WEEK, -2, CURRENT_DATE)
         • For PBR table temporal queries: If no specific date column is mentioned in the query, use 'priority_start_date' as the default date column for temporal filtering (e.g., "next week", "in two weeks", "this month", etc.).
         • For trend related queries, use DISTINCT on date and the columns that are mentioned in the query. Use `ORDER BY date` for user readability.
"""


RESPONSE_GUIDELINES = """
===Response Guidelines & Few-Shot Examples\n
1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n
2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n
3. If the provided context is insufficient, please explain why it can't be generated. \n
4. Please use the most relevant table(s). \n
"""

INSTRUCTION_PROMPT = """
You are a helpful assistant which takes as input a natural language query and generates SQL in the Databricks format.
This SQL can be used to query Nvidia's supply chain database.

Think step by step. Execute the following steps in order to generate SQL compatible with Databricks.
1. Review concepts related to Nvidia Supply Chain DB under CONCEPTS section.
2. Carefully review the guidelines under GUIDELINES section. Ensure that all guidelines are followed.

===CONCEPTS:
{CONCEPTS}

===GUIDELINES:
{GUIDELINES}

Please generate only the SQL and no other output.
"""

# Format the prompt template with actual values
INSTRUCTION_PROMPT = INSTRUCTION_PROMPT.format(CONCEPTS=CONCEPTS, GUIDELINES=GUIDELINES)


def generate_table_description(table: dict) -> str:
    """Generate a description of a table based on its schema."""
    description = f"{table['name']} - {table['description']}\n"

    # Iterate over the fields in the schema
    for field in table["schema"]:
        description += (
            f"    • {field['field']} ({field['type']}): {field['description']}\n"
        )

    return description.strip()
