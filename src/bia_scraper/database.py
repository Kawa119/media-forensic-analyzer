import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from .models import Company

DB_PATH = Path("yell_scraper.db")


class Database:
    """Simple SQLite wrapper for persisting scraped companies and progress."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=30
        )
        # Enable write-ahead logging for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.Lock()
        self._ensure_schema()

    @contextmanager
    def _cursor(self):
        """Thread-safe cursor context manager."""
        with self.lock:
            cur = self.conn.cursor()
            try:
                yield cur
                self.conn.commit()
            except sqlite3.OperationalError:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    def _ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    identifier TEXT PRIMARY KEY,
                    name TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    website TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS progress (
                    last_id INTEGER
                )
                """
            )
            # initialize progress if empty
            cur.execute("SELECT COUNT(*) FROM progress")
            if cur.fetchone()[0] == 0:
                cur.execute("INSERT INTO progress(last_id) VALUES (0)")

    def save_company(self, company: Company) -> None:
        if not company.is_valid():
            return
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO companies
                (identifier, name, phone, email, address, website)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    company.identifier,
                    company.name,
                    company.phone,
                    company.email,
                    company.address,
                    company.website,
                ),
            )

    def update_progress(self, last_id: int) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE progress SET last_id=?", (last_id,))

    def get_last_id(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT last_id FROM progress")
            row = cur.fetchone()
            return row[0] if row else 0

    def all_companies(self) -> Iterable[Company]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM companies")
            for row in cur:
                yield Company(*row)

    def close(self) -> None:
        self.conn.close()

