"""
Session Memory — SQLite-backed conversation store.

Why SQLite over in-memory:
  Survives server restarts. Multi-turn follow-up resolution needs prior turns
  to still exist after a uvicorn reload during development. Zero infra overhead.
  The right tradeoff for a demo that must prove follow-up handling works end-to-end.

Schema:
  sessions(session_id TEXT PK, user_id TEXT, created_at TEXT)
  turns(id INTEGER PK, session_id TEXT FK, role TEXT, content TEXT, created_at TEXT)
"""
import os
import json
import logging
import sqlite3
from typing import List, Dict
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

# Colors (same pattern as deal_agent_framework.py)
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

DB_PATH = os.getenv("MEMORY_DB_PATH", "valura_memory.db")
MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "6"))  # last N turns fed to classifier


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
    """)
    conn.commit()
    conn.close()


class SessionMemory:
    """
    Read and write conversation turns for a single session.
    One instance per request — cheap to construct.
    """

    def __init__(self, session_id: str, user_id: str = "anonymous"):
        self.session_id = session_id
        self.user_id = user_id
        self._ensure_session()

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Session Memory] " + message + RESET
        logging.info(text)

    def _ensure_session(self) -> None:
        conn = _get_connection()
        existing = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, created_at) VALUES (?, ?, ?)",
                (self.session_id, self.user_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            self.log(f"New session created: {self.session_id}")
        conn.close()

    def get_recent_turns(self, n: int = MAX_CONTEXT_TURNS) -> List[Dict[str, str]]:
        """Return the last N turns as [{"role": ..., "content": ...}, ...]."""
        conn = _get_connection()
        rows = conn.execute(
            """
            SELECT role, content FROM turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.session_id, n),
        ).fetchall()
        conn.close()
        # Reverse so oldest turn is first (chronological order for LLM context)
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def save_turn(self, role: str, content: str) -> None:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    def get_all_turns(self) -> List[Dict[str, str]]:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()
        conn.close()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    @classmethod
    def reset_session(cls, session_id: str) -> None:
        """Delete all turns for a session. Useful in tests."""
        conn = _get_connection()
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    init_db()
    mem = SessionMemory(session_id="test-session-001", user_id="user_001")
    mem.save_turn("user", "Tell me about Microsoft")
    mem.save_turn("assistant", "Microsoft is a large-cap technology company...")
    mem.save_turn("user", "What about Apple?")
    print(json.dumps(mem.get_recent_turns(), indent=2))
