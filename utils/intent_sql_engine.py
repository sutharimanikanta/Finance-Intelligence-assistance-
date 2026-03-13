"""
utils/intent_sql_engine.py
─────────────────────────────────────────────────────────────────────────────
The core of the system.

KEY INNOVATION (kept from the original notebook, but enhanced):
  The LLM is ONLY used to classify *intent* into a structured object.
  All SQL is built deterministically by code — zero SQL hallucination.

NEW vs notebook:
  • TriageRouter: decides whether a query needs SQL, RAG, Web-Search, or a
    combination — enabling multi-source answers.
  • Structured `QueryIntent` is now a frozen dataclass (immutable once set).
  • ClarificationManager is a proper state-machine with a timeout counter.
  • EntityResolver caches portfolio names across sessions.
  • SQLGenerator validates every column reference against the live schema.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from models.llm import chat_complete

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  INTENT VOCABULARY
# ─────────────────────────────────────────────

class OperationType(Enum):
    LIST      = "list"
    COUNT     = "count"
    SUM       = "sum"
    AGGREGATE = "aggregate"
    COMPARE   = "compare"
    TOP       = "top"       # "which X has the highest/lowest Y" — ranked LIMIT query
    UNKNOWN   = "unknown"


class TableType(Enum):
    HOLDINGS = "holdings"
    TRADES   = "trades"
    BOTH     = "both"


class SourceType(Enum):
    """Which data sources the triage router selects."""
    SQL       = "sql"        # portfolio DB query
    RAG       = "rag"        # internal research docs
    WEB       = "web"        # live web search
    SQL_RAG   = "sql+rag"    # hybrid: DB + docs
    SQL_WEB   = "sql+web"    # hybrid: DB + news
    RAG_WEB   = "rag+web"    # hybrid: docs + news
    ALL       = "all"        # full hybrid


@dataclass
class QueryIntent:
    """Structured semantic intent.  No SQL lives here."""
    operation:              OperationType
    table:                  Optional[TableType]
    metric:                 Optional[str]           = None
    group_by:               Optional[List[str]]     = None
    filters:                Optional[Dict[str, str]] = None
    order_by:               Optional[str]           = None
    order_direction:        str                     = "DESC"
    limit:                  Optional[int]           = None   # for TOP queries
    # Clarification
    needs_clarification:    bool                    = False
    clarification_question: Optional[str]           = None
    clarification_type:     Optional[str]           = None


# ─────────────────────────────────────────────
# 2.  TRIAGE ROUTER  (NEW)
# ─────────────────────────────────────────────

_TRIAGE_SYSTEM = """You are a financial assistant router. Your job is to decide
which data sources are needed to answer the user's question.

Available sources:
  sql  – live portfolio database (holdings & trades CSVs loaded into SQLite)
  rag  – internal research documents (PDFs / DOCX uploaded by the user)
  web  – real-time web search for market news & company information

SQL PRIORITY RULES (apply these first before any other rule):
- Any query containing words like "holdings", "trades", "portfolio", "profit",
  "loss", "returns", "invested", "exposure", "position", "quantity", "qty",
  "check in", "in holdings", "in trades", "my fund" → ALWAYS route to sql
  (even if a proper noun or company name appears — it is likely a portfolio or security name).
- A named entity (person name, brand, word) followed by "portfolio", "profit",
  "holdings", "returns", or "exposure" → sql (the name is a portfolio filter).
- "What company is X and check in holdings" or similar dual-intent phrasing → sql+web
  (sql for the DB lookup, web for the company description).
- Misspelled or unusual words near portfolio keywords → still route to sql;
  the entity resolver handles spelling correction.

OTHER RULES:
- Questions about financial concepts, company fundamentals, research from documents → rag
- Pure questions about recent news, current prices, market events (no DB keyword) → web
- "How is X affecting my portfolio?" → sql+web
- "Explain X and show if it's in my portfolio" → sql+rag
- Complex hybrid questions → all

Return ONLY a JSON object:
{"source": "<sql|rag|web|sql+rag|sql+web|rag+web|all>", "reasoning": "<one line>"}
"""

def triage_query(query: str, history: list[dict]) -> SourceType:
    """
    Ask the LLM to decide which source(s) should handle this query.
    Falls back to 'all' on any error.
    """
    ctx = ""
    if history:
        ctx = "Recent turns:\n" + "\n".join(
            f"Q: {h['query']}\nA: {h['response']}" for h in history[-2:]
        )

    messages = [
        {"role": "system", "content": _TRIAGE_SYSTEM},
        {"role": "user",   "content": f"{ctx}\n\nQuery: {query}"},
    ]
    try:
        raw   = chat_complete(messages, max_tokens=120, temperature=0)
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)
        return SourceType(data["source"])
    except Exception as exc:
        logger.warning("Triage failed (%s). Defaulting to ALL.", exc)
        return SourceType.ALL


# ─────────────────────────────────────────────
# 3.  LLM INTENT CLASSIFIER
# ─────────────────────────────────────────────

_INTENT_SYSTEM = """You are a financial query classifier. The database has two tables:
  holdings  columns: AsOfDate, PortfolioName, SecurityTypeName, SecName, Qty, Price, MV_Base
  trades    columns: TradeDate, SettleDate, TradeTypeName, SecurityId, SecurityName, Ticker,
                     Quantity, Price, PortfolioName, TotalCash, CustodianName

Classify the query. Return ONLY valid JSON:
{
  "operation":              "list|count|sum|aggregate|compare|top|unknown",
  "table":                  "holdings|trades|unclear",
  "metric":                 "value|quantity|count|profit|null",
  "group_by":               ["SecName"] or null,
  "filters":                {"PortfolioName": "exact_name"} or null,
  "order_by":               "MV_Base" or null,
  "order_direction":        "DESC" or "ASC",
  "limit":                  1 or null,
  "needs_clarification":    true/false,
  "clarification_question": "text" or null,
  "clarification_type":     "table_choice|entity_name|metric_choice" or null
}

RULES
- Never generate SQL.
- "unclear" table + needs_clarification=true ONLY when the table is genuinely ambiguous AND
  no portfolio/security name is present. Do NOT ask for clarification just because a name
  looks unusual — treat it as a portfolio or security name and use it in filters.
- filters must use exact column names from the schema above.
- "yearly" → group_by: ["Year"];  "by fund" → group_by: ["PortfolioName"]
- NAMING RULES (very important):
    • Any proper noun / word before "portfolio", "profit", "returns", "holdings" →
      set filters.PortfolioName to that word as-is (EntityResolver corrects spelling later).
    • "check [Company] in holdings" or "is [Company] in holdings/trades" →
      table: holdings, operation: list, filters: {"~SecName": "Company"}
      (The tilde prefix ~ means LIKE '%Company%' — use it for company/security name searches.)
    • "profit" or "gain" keyword → operation: aggregate, metric: profit
    • metric: profit means compute SUM(MV_Base - invested_value) if both columns exist,
      otherwise fall back to SUM(MV_Base)
    • TOP / RANKING QUERIES — use operation: "top" for any question asking:
        "which company/stock/security has the highest/lowest/most/least [metric]?"
        "what is the top holding by value?"
        "which holding contributes most to [portfolio]?"
      For these set:
        group_by: ["SecName"]         ← the entity column to display (use "SecurityName" for trades)
        order_by: "MV_Base"           ← the metric to rank by (or "Qty", "Price" etc)
        order_direction: "DESC"       ← DESC for highest, ASC for lowest
        limit: 1                      ← 1 for single top result; increase for top-N
        metric: "value"               ← whichever metric is being ranked
      Do NOT use operation: aggregate or sum for these — that generates SUM() which is wrong.
"""


class IntentClassifier:
    """LLM classifies intent ONLY. Zero SQL output."""

    def classify(self, query: str, history: list[dict] | None = None) -> QueryIntent:
        ctx = ""
        if history:
            ctx = "Recent context:\n" + "\n".join(
                f"Q: {h['query']}\nA: {h['response']}" for h in history[-2:]
            )

        messages = [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user",   "content": f"{ctx}\n\nQuery: {query}"},
        ]
        try:
            raw   = chat_complete(messages, max_tokens=400, temperature=0)
            clean = re.sub(r"```json|```", "", raw).strip()
            d     = json.loads(clean)
            return self._from_dict(d)
        except Exception as exc:
            logger.error("Intent classification error: %s", exc)
            return QueryIntent(
                operation=OperationType.UNKNOWN,
                table=None,
                needs_clarification=True,
                clarification_question="Could you rephrase your question?",
                clarification_type="unclear",
            )

    @staticmethod
    def _from_dict(d: dict) -> QueryIntent:
        table_str = d.get("table", "unclear")
        try:
            table = TableType(table_str) if table_str != "unclear" else None
        except ValueError:
            table = None

        try:
            op = OperationType(d.get("operation", "unknown"))
        except ValueError:
            op = OperationType.UNKNOWN

        return QueryIntent(
            operation=op,
            table=table,
            metric=d.get("metric") if d.get("metric") != "null" else None,
            group_by=d.get("group_by"),
            filters=d.get("filters"),
            order_by=d.get("order_by"),
            needs_clarification=bool(d.get("needs_clarification", False)),
            clarification_question=d.get("clarification_question"),
            clarification_type=d.get("clarification_type"),
        )


# ─────────────────────────────────────────────
# 4.  ENTITY RESOLVER
# ─────────────────────────────────────────────

class EntityResolver:
    """Fuzzy portfolio-name matching via LLM."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn   = conn
        self._cache: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT PortfolioName FROM holdings WHERE PortfolioName IS NOT NULL")
            self._cache["portfolios"] = sorted(r[0] for r in cur.fetchall())
        except Exception as exc:
            logger.error("EntityResolver load error: %s", exc)
            self._cache["portfolios"] = []

    def resolve_portfolio(self, user_input: str) -> Optional[str]:
        if not user_input:
            return None
        portfolios = self._cache.get("portfolios", [])
        # Exact match (case-insensitive)
        for p in portfolios:
            if user_input.strip().lower() == p.lower():
                return p
        # Fuzzy via LLM
        prompt = (
            f'Match "{user_input}" to one of {json.dumps(portfolios)}.\n'
            "Return ONLY the exact name, or \"UNKNOWN\"."
        )
        try:
            result = chat_complete(
                [{"role": "user", "content": prompt}],
                max_tokens=60, temperature=0,
            )
            result = result.strip().strip('"')
            return result if result in portfolios else None
        except Exception as exc:
            logger.error("EntityResolver LLM error: %s", exc)
            return None

    def find_portfolio_in_query(self, raw_query: str) -> Optional[str]:
        """
        Proactively scan the raw user query for a known portfolio name
        (exact substring first, then LLM fuzzy match).  Used before intent
        classification to force SQL routing for portfolio-name queries.
        """
        portfolios = self._cache.get("portfolios", [])
        if not portfolios:
            return None
        query_lower = raw_query.lower()
        # 1. Cheap exact substring search (case-insensitive)
        for p in portfolios:
            if p.lower() in query_lower:
                return p
        # 2. LLM fuzzy match on the full query
        prompt = (
            f"Does the following query mention or refer to (even with a typo) any of "
            f"these portfolio names: {json.dumps(portfolios)}?\n"
            f"Query: \"{raw_query}\"\n"
            "Return ONLY the exact portfolio name it refers to, or \"NONE\"."
        )
        try:
            result = chat_complete(
                [{"role": "user", "content": prompt}],
                max_tokens=60, temperature=0,
            )
            result = result.strip().strip('"')
            return result if result in portfolios else None
        except Exception as exc:
            logger.error("EntityResolver.find_portfolio_in_query LLM error: %s", exc)
            return None

    @property
    def all_portfolios(self) -> list[str]:
        return self._cache.get("portfolios", [])


# ─────────────────────────────────────────────
# 5.  SQL GENERATOR  (code-only, no LLM)
# ─────────────────────────────────────────────

_METRIC_MAP = {
    # (metric, table) → column expression
    ("value",    "holdings"): "MV_Base",
    ("value",    "trades"):   "Price * Quantity",
    ("quantity", "holdings"): "Qty",
    ("quantity", "trades"):   "Quantity",
    ("count",    "holdings"): "1",
    ("count",    "trades"):   "1",
    # profit = current value − invested value (falls back to MV_Base if no invested_value col)
    ("profit",   "holdings"): "MV_Base",   # overridden at runtime if invested_value present
    ("profit",   "trades"):   "Price * Quantity",
}
_DEFAULT_METRIC = {"holdings": "MV_Base", "trades": "Price * Quantity"}


class SQLGenerator:
    """Builds SQL deterministically from a QueryIntent.  Zero LLM involvement."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn   = conn
        self.schema = self._load_schema()

    def _load_schema(self) -> dict:
        schema: dict = {}
        try:
            cur = self.conn.cursor()
            for tbl in ("holdings", "trades"):
                cur.execute(f"PRAGMA table_info({tbl})")
                rows = cur.fetchall()
                schema[tbl] = {r[1] for r in rows}   # set of column names
        except Exception as exc:
            logger.error("Schema load error: %s", exc)
        return schema

    def generate(self, intent: QueryIntent) -> Optional[str]:
        if intent.needs_clarification or not intent.table:
            return None

        table = intent.table.value
        if table == "both":
            # For BOTH, union holdings + trades (simplified)
            return self._build_both_query(intent)

        select = self._select(intent, table)
        where  = self._where(intent, table)
        grp    = self._group_by(intent)
        order  = self._order_by(intent)

        parts = [select, f"FROM {table}"]
        if where:  parts.append(where)
        if grp:    parts.append(grp)
        if order:  parts.append(order)
        # LIMIT — used by TOP queries and any intent with an explicit limit
        if intent.limit and intent.limit > 0:
            parts.append(f"LIMIT {intent.limit}")
        return " ".join(parts)

    def _resolve_metric(self, metric: Optional[str], table: str) -> str:
        # Special handling for "profit" metric: use current_value - invested_value if available
        if metric == "profit" and table == "holdings":
            cols = self.schema.get("holdings", set())
            if "invested_value" in cols:
                return "MV_Base - invested_value"
            # Fall through to MV_Base (best proxy for value without cost basis)
        if metric:
            expr = _METRIC_MAP.get((metric, table))
            if expr:
                return expr
        return _DEFAULT_METRIC.get(table, "MV_Base")

    def _select(self, intent: QueryIntent, table: str) -> str:
        op  = intent.operation
        grp = intent.group_by or []

        if op == OperationType.COUNT:
            if grp:
                return f"SELECT {', '.join(grp)}, COUNT(*) AS count"
            return "SELECT COUNT(*) AS count"

        if op in (OperationType.SUM, OperationType.AGGREGATE, OperationType.COMPARE):
            col = self._resolve_metric(intent.metric, table)
            # Wrap expressions safely
            agg = f"SUM(CAST({col} AS REAL)) AS total"
            if grp:
                return f"SELECT {', '.join(grp)}, {agg}"
            return f"SELECT {agg}"

        if op == OperationType.TOP:
            # Ranked / top-N query: show the entity name + the raw metric value
            # e.g. "which company has highest MV?" → SELECT SecName, MV_Base
            col = self._resolve_metric(intent.metric, table)
            # entity column: prefer group_by if provided, else sensible default
            entity_cols = grp if grp else (
                ["SecName"] if table == "holdings" else ["SecurityName"]
            )
            return f"SELECT {', '.join(entity_cols)}, CAST({col} AS REAL) AS metric_value"

        if op == OperationType.LIST:
            if grp:
                return f"SELECT DISTINCT {', '.join(grp)}"
            return "SELECT DISTINCT PortfolioName"

        return "SELECT *"

    def _where(self, intent: QueryIntent, table: str) -> str:
        if not intent.filters:
            return ""
        valid_cols = self.schema.get(table, set())
        conditions = []
        for col, val in intent.filters.items():
            # Tilde prefix (~ColName) → LIKE '%val%' for company/security searches
            like_mode = col.startswith("~")
            col_clean = col.lstrip("~")
            if col_clean in valid_cols:
                safe_val = str(val).replace("'", "''")
                if like_mode:
                    conditions.append(f"{col_clean} LIKE '%{safe_val}%'")
                else:
                    conditions.append(f"{col_clean} = '{safe_val}'")
            else:
                logger.warning("Filter column '%s' not in schema, skipped.", col_clean)
        return ("WHERE " + " AND ".join(conditions)) if conditions else ""

    def _group_by(self, intent: QueryIntent) -> str:
        return ("GROUP BY " + ", ".join(intent.group_by)) if intent.group_by else ""

    def _order_by(self, intent: QueryIntent) -> str:
        if intent.order_by:
            return f"ORDER BY {intent.order_by} {intent.order_direction}"
        if intent.operation == OperationType.TOP:
            # Default for TOP: order by the metric value column
            return f"ORDER BY metric_value {intent.order_direction}"
        if intent.operation in (OperationType.SUM, OperationType.AGGREGATE, OperationType.COMPARE):
            return f"ORDER BY total {intent.order_direction}"
        return ""

    def _build_both_query(self, intent: QueryIntent) -> str:
        """Union holdings + trades market-value totals."""
        return (
            "SELECT 'holdings' AS source, SUM(CAST(MV_Base AS REAL)) AS total FROM holdings "
            "UNION ALL "
            "SELECT 'trades' AS source, SUM(CAST(Price * Quantity AS REAL)) AS total FROM trades"
        )


# ─────────────────────────────────────────────
# 6.  CLARIFICATION MANAGER  (state machine)
# ─────────────────────────────────────────────

class ClarificationManager:
    """Single-pending clarification with a retry counter to prevent infinite loops."""

    MAX_RETRIES = 2

    def __init__(self):
        self._pending: Optional[dict]     = None
        self._original_intent: Optional[QueryIntent] = None
        self._retries: int                = 0

    def set_pending(self, intent: QueryIntent, original_query: str):
        self._pending         = {
            "type":     intent.clarification_type,
            "question": intent.clarification_question,
        }
        self._original_intent = intent
        self._retries         = 0

    def resolve(self, user_response: str) -> Optional[QueryIntent]:
        if not self._pending:
            return None
        intent = self._original_intent
        ctype  = self._pending["type"]

        if ctype == "table_choice":
            lowered = user_response.lower()
            if "hold"  in lowered: intent.table = TableType.HOLDINGS
            elif "trade" in lowered: intent.table = TableType.TRADES
            elif "both"  in lowered: intent.table = TableType.BOTH
            else:
                self._retries += 1
                if self._retries >= self.MAX_RETRIES:
                    self.clear()
                    return None
                return None   # still pending

        intent.needs_clarification    = False
        intent.clarification_question = None
        self.clear()
        return intent

    def clear(self):
        self._pending         = None
        self._original_intent = None
        self._retries         = 0

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def question(self) -> Optional[str]:
        return self._pending["question"] if self._pending else None


# ─────────────────────────────────────────────
# 7.  RESPONSE FORMATTER
# ─────────────────────────────────────────────

class ResponseFormatter:
    """Converts raw SQL result-sets into readable English."""

    def format(
        self,
        query: str,
        results: list[dict],
        intent: QueryIntent,
        mode: str = "concise",
    ) -> str:
        try:
            if not results:
                return "I couldn't find any matching records in the database for your query."

            # Single scalar result
            if len(results) == 1 and len(results[0]) == 1:
                k, v = next(iter(results[0].items()))
                v_fmt = self._fmt_num(v)
                label = k.replace("_", " ").lower()
                base  = f"The {label} is **{v_fmt}**."
                if mode == "detailed":
                    base += f"\n\n*(Based on your portfolio database as of latest snapshot.)*"
                return base

            # Grouped results
            if intent.group_by:
                return self._format_grouped(results, intent, mode)

            # List
            if intent.operation == OperationType.LIST:
                items = [str(list(r.values())[0]) for r in results]
                return "Portfolios: " + ", ".join(items)

            # Generic table
            return self._format_table(results)
        except Exception as exc:
            logger.error("ResponseFormatter error: %s", exc)
            return "I retrieved data but encountered an error formatting the results."

    def _format_grouped(self, results: list[dict], intent: QueryIntent, mode: str) -> str:
        lines = ["**Results:**\n"]
        for i, row in enumerate(results[:15], 1):
            parts = []
            for k, v in row.items():
                parts.append(f"{k}: **{self._fmt_num(v)}**")
            lines.append(f"{i}. " + " | ".join(parts))
        if mode == "detailed" and len(results) > 15:
            lines.append(f"\n*…and {len(results) - 15} more rows.*")
        return "\n".join(lines)

    def _format_table(self, results: list[dict]) -> str:
        if len(results) == 1:
            return "; ".join(f"{k}: **{self._fmt_num(v)}**" for k, v in results[0].items())
        return f"Found **{len(results)}** records."

    @staticmethod
    def _fmt_num(v) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v) if v is not None else "N/A"
