import json
import sqlite3
from typing import Any

from aroflow.application.port import ResultStore


class SQLiteResultStore(ResultStore):
    """SQLite-based result store for workflow step results."""

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the SQLite result store.

        :param db_path: Path to SQLite database file (defaults to in-memory)
        :type db_path: str
        """
        self.db_path = db_path
        self._conn = None
        self._init_database()

    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def _init_database(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_results (
                workflow_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                result TEXT NOT NULL,
                PRIMARY KEY (workflow_id, step_id)
            )
        """)
        conn.commit()

    def set(self, key: str, value: Any):
        """
        Store a value with the given key.

        :param key: The key to store the value under (format: workflow_id.step_id)
        :type key: str
        :param value: The value to store
        :type value: Any
        """
        # Parse key to extract workflow_id and step_id
        if "." in key:
            workflow_id, step_id = key.split(".", 1)
        else:
            # Fallback for simple keys
            workflow_id = "default"
            step_id = key

        # Serialize the value to JSON
        result_json = json.dumps(value, default=str)

        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO workflow_results (workflow_id, step_id, result) VALUES (?, ?, ?)",
            (workflow_id, step_id, result_json)
        )
        conn.commit()

    def get(self, key: str) -> Any:
        """
        Retrieve a value by key.

        :param key: The key to retrieve the value for (format: workflow_id.step_id)
        :type key: str
        :returns: The stored value
        :rtype: Any
        :raises KeyError: If the key is not found
        """
        # Parse key to extract workflow_id and step_id
        if "." in key:
            workflow_id, step_id = key.split(".", 1)
        else:
            # Fallback for simple keys
            workflow_id = "default"
            step_id = key

        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT result FROM workflow_results WHERE workflow_id = ? AND step_id = ?",
            (workflow_id, step_id)
        )
        row = cursor.fetchone()

        if row is None:
            raise KeyError(f"Key '{key}' not found")

        # Deserialize from JSON
        return json.loads(row[0])

    def __del__(self):
        """Close the database connection on cleanup."""
        if self._conn:
            self._conn.close()