import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional
from .models import Company

DB_PATH = Path('yell_scraper.db')


class Database:
    """Simple SQLite wrapper for persisting scraped companies and progress."""

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._ensure_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            c = conn.cursor()
            c.execute(
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
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS progress (
                    last_id INTEGER
                )
                """
            )
            # initialize progress if empty
            c.execute("SELECT COUNT(*) FROM progress")
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO progress(last_id) VALUES (0)")

    def save_company(self, company: Company) -> None:
        if not company.is_valid():
            return
        with self.connect() as conn:
            conn.execute(
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
        with self.connect() as conn:
            conn.execute("UPDATE progress SET last_id=?", (last_id,))

    def get_last_id(self) -> int:
        with self.connect() as conn:
            cur = conn.execute("SELECT last_id FROM progress")
            row = cur.fetchone()
            return row[0] if row else 0

    def all_companies(self) -> Iterable[Company]:
        with self.connect() as conn:
            cur = conn.execute("SELECT * FROM companies")
            for row in cur:
                yield Company(*row)
