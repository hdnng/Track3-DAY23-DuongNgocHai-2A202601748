"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "outputs/checkpoints.sqlite"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            conn_str = database_url or "postgresql://postgres:postgres@localhost:5432/langgraph_lab"
            return PostgresSaver.from_conn_string(conn_str)
        except Exception as exc:
            msg = "Postgres checkpointer requires PostgreSQL and langgraph-checkpoint-postgres"
            raise NotImplementedError(msg) from exc
    raise ValueError(f"Unknown checkpointer kind: {kind}")


