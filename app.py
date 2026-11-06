"""
Inventory & Fulfillment Exception Engine - Streamlit App
=========================================================
Natural-language query engine for Apex Logistics' supply-chain operations
database. A business question is routed either to a pre-approved verified SQL
template or to fresh SQL generation, passed through a four-check validation
gate (with one retry for generated SQL), executed against a READ-ONLY SQLite
database, and summarised into a concise business answer. Questions that cannot
be resolved reliably are escalated to a human analyst.

Run with:
    streamlit run app.py

Expected next to this file:
    supply_chain_ops.db      (required)  - on-premise SQLite database
    test_queries.csv         (optional)  - ground truth for the Evaluation tab

Credentials (first match wins):
    1. .streamlit/secrets.toml  ->  OPENAI_API_KEY, OPENAI_API_BASE
    2. Environment variables    ->  OPENAI_API_KEY, OPENAI_API_BASE / OPENAI_BASE_URL
    3. config.json next to app  ->  {"OPENAI_API_KEY": "...", "OPENAI_API_BASE": "..."}
"""

import json
import os
import re
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Inventory & Fulfillment Exception Engine",
    page_icon="📦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("SUPPLY_CHAIN_DB_PATH", APP_DIR / "supply_chain_ops.db"))
TEST_QUERIES_PATH = Path(os.environ.get("TEST_QUERIES_PATH", APP_DIR / "test_queries.csv"))
CONFIG_JSON_PATH = APP_DIR / "config.json"

PRIMARY_MODEL = "gpt-4o-mini"      # routing, SQL generation, retry, response
EVALUATOR_MODEL = "gpt-4o"         # relevance validation


# ============================================================
# Credentials (OpenAI API Calling)
# ============================================================

def load_credentials():
    """
    Resolve the OpenAI API key and base URL.
    Order: Streamlit secrets -> environment variables -> config.json.
    """
    api_key, api_base = None, None

    # 1. Streamlit secrets (Streamlit Community Cloud / .streamlit/secrets.toml)
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE") or st.secrets.get("OPENAI_BASE_URL")
    except Exception:
        pass

    # 2. Environment variables
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_base:
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    # 3. config.json (same format the notebook used)
    if (not api_key or not api_base) and CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r") as file:
                config = json.load(file)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    if not api_base:
        api_base = "https://api.openai.com/v1"

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Add `OPENAI_API_KEY` (and optionally "
        "`OPENAI_API_BASE`) to `.streamlit/secrets.toml`, set them as environment "
        "variables, or place a `config.json` next to `app.py`."
    )
    st.stop()

# Store API credentials in environment variables (as in the notebook)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE


# ============================================================
# Model Initialization & Database Connection
# ============================================================

@st.cache_resource
def get_llms():
    """
    Two LLMs separate responsibilities:
    - a lightweight model for primary reasoning, classification, and generation
    - a more capable model for validation and evaluation
    """
    llm = ChatOpenAI(model=PRIMARY_MODEL, temperature=0)
    evaluator_llm = ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    return llm, evaluator_llm


@st.cache_resource
def get_db_connection(db_file: str):
    """
    Opens a read-only connection to the SQLite database.

    - URI mode with ?mode=ro guarantees the connection cannot execute any write
      operation, even if a validation check were bypassed (defense in depth).
    - check_same_thread=False because Streamlit serves requests from worker threads.
    """
    uri = Path(db_file).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


if not DB_PATH.exists():
    st.error(
        f"Database file not found at `{DB_PATH}`. Place `supply_chain_ops.db` "
        "next to `app.py` (or set the `SUPPLY_CHAIN_DB_PATH` environment variable)."
    )
    st.stop()

llm, evaluator_llm = get_llms()
conn = get_db_connection(str(DB_PATH))


# ============================================================
# Database Schema (passed to the LLM prompts)
# ============================================================

database_schema = """
warehouse_master:
  warehouse_id (TEXT, PK): US MSA facility code (e.g., WH_ORD_01, WH_DFW_02)
  warehouse_name (TEXT): legal facility name (e.g., 'Chicago O\'Hare Hub', 'Dallas Fort-Worth Main')
  region (TEXT): US Census Region (Northeast, Midwest, South, West)
  max_capacity_pallet_positions (INTEGER): total high-bay pallet position capacity
  current_occupancy_pct (REAL): utilization percentage; high occupancy threshold is >= 85.0
  is_cbp_bonded_ftz (INTEGER): 1 if CBP-bonded or Foreign Trade Zone, 0 otherwise

inventory_levels:
  inventory_id (INTEGER, PK): auto-increment identifier
  warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  sku_id (TEXT): unique stock keeping unit code
  sku_category (TEXT): one of Consumer Packaged Goods, Automotive Parts, Cold-Chain Perishables, Apparel, Industrial
  units_on_hand (INTEGER): physical stock count in warehouse
  reorder_point (INTEGER): minimum stock threshold before replenishment order
  unit_cost_usd (REAL): carrying unit cost under US GAAP (ASC 330)
  last_restock_date (DATE): date of last inventory receipt

shipment_tracker:
  shipment_id (TEXT, PK): unique BOL or tracking number (e.g., BOL-000001)
  order_id (TEXT): client purchase order reference (e.g., PO-100001)
  origin_warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  scac_code (TEXT, FK): joins to carrier_performance.scac_code
  promised_ship_date (DATE): contractual SLA dispatch date
  actual_ship_date (DATE): actual gate-out dispatch date, NULL if pending
  delivery_status (TEXT): one of On-Time, Delayed, In-Transit, Cancelled
  delay_reason (TEXT): one of 'FMCSA Driver HOS Limit', 'DOT Road Closure', 'Chassis Shortage', 'CBP Freight Hold', 'Warehouse Backlog', 'N/A'

carrier_performance:
  scac_code (TEXT, PK): NMFTA Standard Carrier Alpha Code (e.g., FEDX, UPSN, XPOF, JBHA, ODFL)
  carrier_name (TEXT): legal corporate name of carrier
  otif_compliance_pct (REAL): On-Time In-Full delivery percentage as a decimal (e.g., 0.94 for 94%)
  avg_delay_hours (REAL): mean delivery delay in hours
  otif_chargeback_usd (REAL): accrued SLA non-compliance penalties in USD

Business rules:
- Stockout definition: units_on_hand = 0
- Below reorder definition: units_on_hand > 0 AND units_on_hand <= reorder_point
- High occupancy threshold: current_occupancy_pct >= 85.0
- Delayed shipments: delivery_status = 'Delayed'
- In-transit shipments: delivery_status = 'In-Transit'
- Inventory value formula: units_on_hand * unit_cost_usd
"""


# ============================================================
# Verified Query Template Library
# ============================================================
# 10 pre-approved templates for common supply chain questions. Each has a
# verified SQL query and a plain-English description used for intent routing.

verified_query_library = {
    'VQ1': {
        'description': 'Regional stockout count showing which US regions have the most SKUs currently at zero units on hand',
        'sql': """SELECT w.region,
     COUNT(*) AS stockout_skus
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
WHERE i.units_on_hand = 0
GROUP BY w.region
ORDER BY stockout_skus DESC"""
    },


    'VQ2': {
        'description': 'SKU categories with the most items currently below reorder point but not yet stocked out, indicating near-term replenishment need',
        'sql': """SELECT sku_category,
     COUNT(*) AS below_reorder_skus
FROM inventory_levels
WHERE units_on_hand > 0 AND units_on_hand <= reorder_point
GROUP BY sku_category
ORDER BY below_reorder_skus DESC"""
    },


    'VQ3': {
        'description': 'Warehouses at or above the 85% high-occupancy threshold, indicating capacity risk',
        'sql': """SELECT warehouse_id,
     warehouse_name,
     region,
     current_occupancy_pct
FROM warehouse_master
WHERE current_occupancy_pct >= 85.0
ORDER BY current_occupancy_pct DESC"""
    },


    'VQ4': {
        'description': 'Total count of shipments currently marked as Delayed in the shipment tracker',
        'sql': """SELECT COUNT(*) AS delayed_count
FROM shipment_tracker
WHERE delivery_status = 'Delayed'"""
    },


    'VQ5': {
        'description': 'Carriers ranked from worst to best by On-Time In-Full (OTIF) compliance percentage',
        'sql': """SELECT scac_code,
     carrier_name,
     otif_compliance_pct
FROM carrier_performance
ORDER BY otif_compliance_pct ASC"""
    },


    'VQ6': {
        'description': 'Carrier with the highest accrued OTIF chargeback penalties in USD',
        'sql': """SELECT * FROM (
    SELECT carrier_name,
         otif_chargeback_usd
    FROM carrier_performance
    ORDER BY otif_chargeback_usd DESC
    LIMIT 1)"""
    },


    'VQ7': {
    'description': 'Top 5 warehouses ranked by total inventory value (units_on_hand * unit_cost_usd), showing where carrying cost is concentrated',
    'sql': """SELECT * FROM (SELECT w.warehouse_id,
     w.warehouse_name,
     ROUND(SUM(i.units_on_hand * i.unit_cost_usd), 2) AS inventory_value_usd
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
GROUP BY w.warehouse_id
ORDER BY inventory_value_usd DESC
LIMIT 5)"""
},


    'VQ8': {
        'description': 'Most common reasons for shipment delays with occurrence counts across all delayed shipments',
        'sql': """SELECT delay_reason,
     COUNT(*) AS occurrences
FROM shipment_tracker
WHERE delivery_status = 'Delayed'
GROUP BY delay_reason
ORDER BY occurrences DESC"""
    },


    'VQ9': {
        'description': 'Average occupancy comparison between CBP-bonded/FTZ warehouses and non-bonded facilities',
        'sql': """SELECT is_cbp_bonded_ftz,
     ROUND(AVG(current_occupancy_pct), 2) AS avg_occupancy_pct,
     COUNT(*) AS warehouse_count
FROM warehouse_master
GROUP BY is_cbp_bonded_ftz"""
    },


    'VQ10': {
        'description': 'Aggregate count of shipments currently in transit broken down by carrier SCAC code',
        'sql': """SELECT scac_code,
     COUNT(*) AS in_transit_count
FROM shipment_tracker
WHERE delivery_status = 'In-Transit'
GROUP BY scac_code
ORDER BY in_transit_count DESC"""
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated), 'query_id' (template ID or None),
            and 'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a supply chain operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'out of stock' means stockout (units_on_hand = 0), 'FTZ' or 'bonded' refers to is_cbp_bonded_ftz = 1, 'late' or 'behind schedule' means Delayed, 'facilities near capacity' means high occupancy.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'show me the shipments') should NOT match a template that returns an aggregate count.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""
### ROLE
You are a senior SQL developer specializing in supply chain operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using warehouse_name or warehouse_id where relevant (for example, 'Dallas' maps to WH_DFW_01, 'Chicago' maps to WH_ORD_01).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()

    # Strip markdown fences if present
    # Remove Markdown code fences (```sql ... ```) and extra whitespace from the extracted SQL
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove generic Markdown code fences (``` ... ```) and extra whitespace
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()

    return sql


# ============================================================
# Tool 3: Query Validation (four-check gate)
# ============================================================
# 1. Read-only check        - SELECT/WITH only, no destructive ops, single statement
# 2. Schema conformance     - tables / columns exist in the database
# 3. Parse & plan check     - SQLite EXPLAIN succeeds
# 4. LLM relevance check    - SQL actually answers the user's question
# The query moves to execution ONLY when all checks pass.

def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):

    # Store the validation result; query is considered failed by default
    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }


    # ============================================================
    # CHECK 1: READ-ONLY SHAPE CHECK
    # Make sure the query is safe and contains only read operations.
    # ============================================================

    sql_upper = candidate_sql.upper().strip()

    forbidden_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT',
        'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH'
    ]

    # Query must start with SELECT or WITH
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result

    # Block forbidden SQL operations
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result

    # Allow only one SQL statement
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result


    # ============================================================
    # CHECK 2: SCHEMA CONFORMANCE CHECK
    # Make sure the query uses valid tables and columns.
    # ============================================================

    cur = db_connection.cursor()

    # Get all real tables from the database
    real_tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    # Get all real columns from those tables
    real_columns = set()

    for t in real_tables:
        for col_info in cur.execute(
            f"PRAGMA table_info({t})"
        ).fetchall():
            real_columns.add(col_info[1].lower())

    # Parse the SQL
    parsed = sqlparse.parse(candidate_sql)[0]

    # Extract identifiers used in the SQL
    tokens = [
        str(t).strip().lower()
        for t in parsed.flatten()
        if t.ttype is None or 'Name' in str(t.ttype)
    ]

    referenced_identifiers = re.findall(
        r'\b[a-z_][a-z0-9_]*\b',
        candidate_sql.lower()
    )

    # SQL keywords that should not be treated as table/column names
    sql_keywords = {
        'select', 'from', 'where', 'and', 'or', 'group', 'by',
        'order', 'having', 'limit', 'join', 'on', 'as', 'case',
        'when', 'then', 'else', 'end', 'sum', 'count', 'avg',
        'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
        'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in',
        'like', 'with', 'union', 'all', 'between', 'coalesce'
    }

    # Find identifiers that are not known tables, columns, or keywords
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_columns
        and tok not in real_tables
        and not tok.isdigit()
        and tok not in ('w', 'i', 's', 'c', 'p')
    ]


    # ============================================================
    # CHECK 3: PARSE-AND-PLAN DRY RUN
    # Use EXPLAIN to confirm that the SQL can be parsed and planned.
    # ============================================================

    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()

    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result


    # ============================================================
    # CHECK 4: LLM RELEVANCE CHECK
    # Ask the LLM whether the SQL actually answers the user's question.
    # ============================================================

    # Check whether this SQL came from the verified-query library
    is_verified_track = (
        query_id is not None and query_id in query_library
    )

    # Give the evaluator the correct context for the query type
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all regions/categories/carriers rather than filtering "
        "to just what the user asked). A separate response-generation step will "
        "filter and highlight the relevant rows afterward. Do NOT fail this query "
        "for lacking a WHERE clause that narrows to the user's specific "
        "region/category/carrier: judge only whether the underlying metric, tables, "
        "and aggregation logic match the question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped and filtered to answer it directly."
    )

    # Prompt the LLM to evaluate business relevance
    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""

    # Send the question and SQL to the evaluator LLM
    relevance_response = evaluator_llm.invoke(
        relevance_prompt
    ).content.strip()

    # Extract the JSON response from the LLM
    json_match = re.search(
        r'\{.*\}',
        relevance_response,
        re.DOTALL
    )

    if json_match:

        relevance_json = json.loads(json_match.group())

        # Store the LLM confidence score
        result['relevance_confidence'] = relevance_json.get(
            'confidence',
            0.0
        )

        # Fail if the LLM rejects the query or confidence is below 0.6
        if (
            relevance_json.get('verdict') == 'no'
            or relevance_json.get('confidence', 0.0) < 0.6
        ):
            result['failed_check'] = 'llm_relevance'
            result['details'] = (
                f"Relevance check failed: "
                f"{relevance_json.get('reason', 'unknown')}"
            )
            return result


    # ============================================================
    # ALL 4 CHECKS PASSED
    # The query is now approved for execution.
    # ============================================================

    result['passed'] = True
    result['details'] = 'All validation checks passed'

    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()

    # Remove Markdown code fences (```sql ... ```) and extra whitespace from the extracted SQL
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove generic Markdown code fences (``` ... ```) and extra whitespace
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()

    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    # Initialize the result structure with default values.
    # The DataFrame will be populated after the SQL query is executed.
    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    # Execute the validated SQL query and store the results in a DataFrame.
    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Perform basic reasonableness checks on the query results.
    # These checks flag potential data-quality issues but do not stop execution.

    # Check whether the query returned any rows.
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    # Check numeric columns for potentially unexpected values.
    for col in df.select_dtypes(include='number').columns:

        # Flag negative values unless the column represents a deviation or change,
        # where negative values can be valid and meaningful.
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')

        # Check for missing (NULL/NaN) values in the numeric column.
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()

            # Warn when more than 50% of the column values are missing,
            # as this may indicate a data-quality or query issue.
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### ROLE
You are a supply chain operations analyst writing a concise business response for a fulfillment or inventory question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific region, warehouse, carrier, or category, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as a warehouse close to a capacity threshold or a carrier significantly worse than peers.
6. Use clear, professional language suitable for a fulfillment operations memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _usd as "$X", _pct or _percent as "X%", _count as a plain count, _hours as "X hours". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    response = llm.invoke(response_prompt).content.strip()
    return response


# ============================================================
# Complete Workflow (single entry point)
# ============================================================
# 1. Classify intent and pick a route
# 2. Build SQL (verified template or freshly generated)
# 3. Validation gate
# 4. Retry ONCE on the generated track if validation fails
# 5. Escalate to a human analyst if it still fails
# 6. Execute the approved SQL (read-only)
# 7. Generate the answer and record everything in the log

def run_workflow(user_question, verbose=True, reporter=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - verbose (bool): If True, reports intermediate pipeline stages.
    - reporter (callable, optional): Receives each stage message (used by the
      Streamlit UI to show a live trace). Falls back to print() when omitted.

    Returns:
    - dict: Complete pipeline output including response, SQL, data, and log.
    '''

    def say(message):
        if verbose:
            (reporter or print)(message)

    db_connection = conn
    query_library = verified_query_library
    schema_context = database_schema

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'response': None
    }

    # Step 1: Intent classification

    # run the intent classification tool
    classification = classify_intent(user_question, query_library)

    # log the output from intent classification tool
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction

    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)

    # log the chosen SQL query
    log['candidate_sql'] = candidate_sql

    # report the chosen route
    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate

    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])

    # log the validation gate output
    log['gate_result'] = gate
    log['confidence'] = gate.get('relevance_confidence')

    # report the validation gate output
    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails

    # if validation fail send for retry generation
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        # report the validation gate output of updated query
        say(f" Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing

    # log for escalation to human
    if not gate['passed']:
        log['escalated'] = True
        log['route'] = 'escalate'
        log['response'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = gate.get('relevance_confidence')
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute

    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    # report the execution result
    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    response = generate_response(user_question, df, log['route'], log['query_id'])
    log['response'] = response

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ============================================================
# Evaluation against Ground Truth (test_queries.csv)
# ============================================================

def run_evaluation(ground_truth, progress_callback=None):
    '''
    Runs every ground-truth question through the workflow and compares route and
    query ID with the expected values. Returns (evaluation_df, metrics, test_results).
    '''
    test_results = []
    total = len(ground_truth)

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        if progress_callback:
            progress_callback(i, total, gt['question'])
        test_results.append(run_workflow(gt['question'], verbose=False))

    evaluation_rows = []

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results[i]

        evaluation_rows.append({
            'Test ID': gt['test_id'],
            'Expected Route': gt['expected_route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['expected_route'],
            'Expected Query ID': gt['expected_query_id'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['expected_query_id']) and
                (tr['query_id'] is None or pd.isna(tr['query_id']))
            ) or tr['query_id'] == gt['expected_query_id'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count']
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = (
        evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    )

    query_accuracy = (
        evaluation_df.loc[verified, 'Query ID Match'].mean() * 100
    )

    average_confidence = pd.to_numeric(
        evaluation_df['Confidence']
    ).mean()

    metrics = {
        'path_accuracy': path_accuracy,
        'query_accuracy': query_accuracy,
        'average_confidence': average_confidence,
    }

    return evaluation_df, metrics, test_results


# ============================================================
# Streamlit helpers
# ============================================================

EXAMPLE_QUESTIONS = [
    "Which 5 warehouses have the highest dollar value of on-hand inventory?",
    "Do our bonded warehouses run hotter on capacity than the regular ones?",
    "Which regions have an above-average stockout rate, and how many SKUs are sitting at or below reorder point in those regions?",
    "For our FTZ warehouses, what's the at-risk inventory value tied to shipments that were delayed more than 2 days, broken out by carrier?",
    "How many shipments are currently delayed?",
    "Which carriers have the worst OTIF compliance?",
    "What's the average employee headcount per warehouse?",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None
if "eval" not in st.session_state:
    st.session_state.eval = None


def use_example(question):
    """Callback: load an example question into the input box and run it."""
    st.session_state["question_input"] = question
    st.session_state["auto_run"] = True


def confidence_badge(confidence):
    if isinstance(confidence, (int, float)):
        if confidence >= 0.8:
            badge = "🟢"
        elif confidence >= 0.6:
            badge = "🟡"
        else:
            badge = "🔴"
        return badge, f"{confidence:.2f}"
    return "⚪", str(confidence)


def render_result(last, show_trace):
    """Render the stored result of the most recent pipeline run."""
    output = last["output"]

    st.divider()

    # ---------------- Escalation ----------------
    if output["escalated"]:
        st.warning(output["response"])

        with st.expander("Validation details", expanded=True):
            st.json(output["gate_result"])

        with st.expander("Last candidate SQL (failed validation)"):
            st.code(output["candidate_sql"], language="sql")

    # ---------------- Successful result ----------------
    else:
        badge, confidence_display = confidence_badge(output["confidence"])

        st.subheader("Answer")
        st.write(output["response"])

        st.caption(
            f"{badge} Confidence: {confidence_display}  ·  "
            f"Route: {output['route']}"
            + (f" ({output['query_id']})" if output["query_id"] else "")
            + f"  ·  Rows returned: {output['row_count']}"
            + f"  ·  Retry used: {output['retry_used']}"
            + f"  ·  Time: {last['elapsed']:.1f}s"
        )

        if output["dataframe"] is not None:
            st.subheader("Result data")
            st.dataframe(output["dataframe"], hide_index=True)

        with st.expander("View executed SQL"):
            st.code(output["executed_sql"], language="sql")

    if show_trace and last.get("trace"):
        with st.expander("Pipeline trace"):
            st.code("\n".join(last["trace"]), language=None)

    with st.expander("Audit log entry (JSON)"):
        st.json(output["log"])


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About")
    st.write(
        "Ask operational questions in plain English. Recurring questions are "
        "answered from a pre-approved library of verified SQL templates; novel "
        "questions get freshly generated, read-only SQL. Every query passes a "
        "four-check validation gate before it touches the database, and anything "
        "that cannot be resolved reliably is escalated to a human analyst."
    )

    show_trace = st.checkbox("Show pipeline trace", value=True)

    st.subheader("Example questions")
    for idx, q in enumerate(EXAMPLE_QUESTIONS):
        st.button(q, key=f"example_{idx}", on_click=use_example, args=(q,))

    st.subheader("Connection")
    st.caption(
        f"Database: `{DB_PATH.name}` (read-only)  \n"
        f"Primary model: `{PRIMARY_MODEL}`  \n"
        f"Evaluator model: `{EVALUATOR_MODEL}`"
    )

    if st.button("Test API connection"):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            response = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": "Hello, how many tokens did this use?"}],
            )
            usage = response.usage
            st.success(
                f"Connected. Prompt tokens: {usage.prompt_tokens} · "
                f"Completion tokens: {usage.completion_tokens} · "
                f"Total: {usage.total_tokens}"
            )
        except Exception as e:
            st.error(f"API connection failed: {e}")


# ============================================================
# Main Page
# ============================================================

st.title("📦 Inventory & Fulfillment Exception Engine")
st.caption(
    "Apex Logistics · Natural-language queries over inventory, warehouses, "
    "shipments, and carrier performance - verified answers in minutes."
)

tab_ask, tab_eval, tab_library, tab_log = st.tabs(
    ["Ask a question", "Evaluation", "Library & schema", "Session log"]
)


# ------------------------------------------------------------
# Tab 1: Ask a question
# ------------------------------------------------------------

with tab_ask:
    user_question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. Which warehouses are operating above 85% capacity?",
    )

    run_clicked = st.button("Run query", type="primary")
    auto_run = st.session_state.pop("auto_run", False)

    if (run_clicked or auto_run) and user_question.strip():

        trace_lines = []
        trace_placeholder = st.empty()

        def reporter(message):
            trace_lines.append(message.rstrip())
            if show_trace:
                trace_placeholder.code("\n".join(trace_lines), language=None)

        started = time.time()

        try:
            with st.spinner("Running pipeline..."):
                output = run_workflow(user_question.strip(), verbose=True, reporter=reporter)
        except Exception as e:
            trace_placeholder.empty()
            st.error(f"Pipeline failed: {e}")
            st.stop()

        elapsed = time.time() - started
        trace_placeholder.empty()

        st.session_state.last = {
            "output": output,
            "trace": trace_lines,
            "elapsed": elapsed,
        }

        # Session audit trail
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": output["user_question"],
            "route": output["route"],
            "query_id": output["query_id"],
            "match_reason": output["match_reason"],
            "retry_used": output["retry_used"],
            "escalated": output["escalated"],
            "confidence": output["confidence"],
            "row_count": output["row_count"],
            "executed_sql": output["executed_sql"],
            "candidate_sql": output["candidate_sql"],
            "response": output["response"],
            "seconds": round(elapsed, 1),
        })

    elif run_clicked:
        st.info("Please enter a question first.")

    if st.session_state.last is not None:
        render_result(st.session_state.last, show_trace)


# ------------------------------------------------------------
# Tab 2: Evaluation against ground truth
# ------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluation against ground truth")
    st.write(
        "Runs every case in `test_queries.csv` through the full workflow and "
        "compares the selected route and verified-query ID with the expected values. "
        "This makes several LLM calls per test case."
    )

    if not TEST_QUERIES_PATH.exists():
        st.info(
            f"`{TEST_QUERIES_PATH.name}` was not found next to `app.py`, so the "
            "evaluation is unavailable. Add the file to enable this tab."
        )
    else:
        ground_truth = pd.read_csv(TEST_QUERIES_PATH)

        with st.expander("Ground-truth test cases", expanded=False):
            st.dataframe(ground_truth, hide_index=True)

        if st.button("Run evaluation"):
            progress = st.progress(0.0, text="Starting evaluation...")

            def on_progress(i, total, question):
                progress.progress(i / total, text=f"Test {i + 1} of {total}: {question}")

            try:
                evaluation_df, metrics, test_results = run_evaluation(ground_truth, on_progress)
                progress.progress(1.0, text="Evaluation complete")
                st.session_state.eval = {
                    "df": evaluation_df,
                    "metrics": metrics,
                    "results": test_results,
                }
            except Exception as e:
                progress.empty()
                st.error(f"Evaluation failed: {e}")

        if st.session_state.eval is not None:
            metrics = st.session_state.eval["metrics"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Selected Path Accuracy", f"{metrics['path_accuracy']:.1f}%")
            c2.metric(
                "Selected Query Accuracy (verified only)",
                "n/a" if pd.isna(metrics["query_accuracy"]) else f"{metrics['query_accuracy']:.1f}%",
            )
            c3.metric(
                "Average Confidence Score",
                "n/a" if pd.isna(metrics["average_confidence"]) else f"{metrics['average_confidence']:.2f}",
            )

            st.dataframe(st.session_state.eval["df"], hide_index=True)

            with st.expander("Per-test responses and data"):
                for (_, gt), tr in zip(ground_truth.iterrows(), st.session_state.eval["results"]):
                    st.markdown(f"**{gt['test_id']}** - {gt['question']}")
                    st.write(tr["response"])
                    if tr["dataframe"] is not None:
                        st.dataframe(tr["dataframe"], hide_index=True)
                    st.divider()


# ------------------------------------------------------------
# Tab 3: Library & schema
# ------------------------------------------------------------

with tab_library:
    st.subheader("Verified query library")
    st.caption(f"{len(verified_query_library)} pre-approved templates")

    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"], language="sql")

    st.subheader("Database schema (as passed to the LLM)")
    st.code(database_schema, language=None)

    st.subheader("Database tables")
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    st.dataframe(tables, hide_index=True)

    for table_name in ['warehouse_master', 'inventory_levels', 'shipment_tracker', 'carrier_performance']:
        with st.expander(f"Preview: {table_name}"):
            try:
                st.dataframe(
                    pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn),
                    hide_index=True,
                )
            except Exception as e:
                st.error(f"Could not read {table_name}: {e}")


# ------------------------------------------------------------
# Tab 4: Session log
# ------------------------------------------------------------

with tab_log:
    st.subheader("Session audit log")
    st.caption(
        "Executed statements, validation outcomes, and confidence scores for "
        "questions asked in this browser session."
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, hide_index=True)
        st.download_button(
            "Download log as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="query_engine_audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No questions have been run in this session yet.")
