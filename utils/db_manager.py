"""
utils/db_manager.py
Loads CSV datasets into SQLite and executes queries.
"""

import logging
import sqlite3
from typing import Optional, Tuple

import pandas as pd

from config.config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton-style SQLite manager for holdings & trades."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    # ── setup ────────────────────────────────────────────────────────────────

    def init_from_csvs(self, holdings_path: str, trades_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load two CSVs, create SQLite DB, return the DataFrames for preview."""
        try:
            holdings_df = pd.read_csv(holdings_path)
            trades_df   = pd.read_csv(trades_path)
            holdings_df.columns = holdings_df.columns.str.strip()
            trades_df.columns   = trades_df.columns.str.strip()

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            holdings_df.to_sql("holdings", self.conn, if_exists="replace", index=False)
            trades_df.to_sql("trades",   self.conn, if_exists="replace", index=False)
            self.conn.commit()
            logger.info(
                "DB initialised: %d holdings, %d trades",
                len(holdings_df), len(trades_df),
            )
            return holdings_df, trades_df
        except Exception as exc:
            logger.error("DB init error: %s", exc)
            raise

    def connect(self):
        """Open connection to an existing DB."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except Exception as exc:
            logger.error("DB connect error: %s", exc)
            raise

    # ── query ────────────────────────────────────────────────────────────────

    def execute(self, sql: str) -> Tuple[bool, list[dict], Optional[str]]:
        """
        Execute a SELECT statement.

        Returns:
            (success, rows_as_dicts, error_message)
        """
        if self.conn is None:
            return False, [], "Database not connected."
        try:
            cur     = self.conn.cursor()
            cur.execute(sql)
            cols    = [d[0] for d in cur.description]
            rows    = [dict(zip(cols, row)) for row in cur.fetchall()]
            return True, rows, None
        except Exception as exc:
            logger.error("Query execution error: %s | SQL: %s", exc, sql)
            return False, [], str(exc)

    def table_info(self, table: str) -> list[str]:
        """Return column names for a table."""
        if self.conn is None:
            return []
        try:
            cur = self.conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            return [row[1] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("table_info error: %s", exc)
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("DB connection closed.")
