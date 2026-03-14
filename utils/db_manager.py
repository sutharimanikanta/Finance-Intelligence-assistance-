"""
utils/db_manager.py
Loads CSV datasets into SQLite and executes queries.
"""

import logging
import sqlite3

import pandas as pd

from config.config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = None

    def init_from_csvs(self, holdings_path, trades_path):
        holdings_df = pd.read_csv(holdings_path)
        trades_df = pd.read_csv(trades_path)
        holdings_df.columns = holdings_df.columns.str.strip()
        trades_df.columns = trades_df.columns.str.strip()

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        holdings_df.to_sql("holdings", self.conn, if_exists="replace", index=False)
        trades_df.to_sql("trades", self.conn, if_exists="replace", index=False)
        self.conn.commit()
        logger.info("DB initialised: %d holdings, %d trades", len(holdings_df), len(trades_df))
        return holdings_df, trades_df

    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def execute(self, sql):
        if self.conn is None:
            return False, [], "Database not connected."
        try:
            cur = self.conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return True, rows, None
        except Exception as exc:
            logger.error("Query error: %s | SQL: %s", exc, sql)
            return False, [], str(exc)

    def table_info(self, table):
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