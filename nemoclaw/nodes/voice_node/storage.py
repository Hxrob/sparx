"""
Persistent storage for SparX voice node.

SQLite — no server, no dependencies, survives restarts.
DB file: voice_node/sparx.db (next to server.py)

Tables:
  sessions  — one row per browser session, tracks call count + last seen
  calls     — one row per /transcribe request, full structured record

Usage:
    from storage import Storage
    db = Storage()                        # call once at startup
    db.save_call(session_id, payload)     # after every /transcribe
    db.session_history(session_id)        # last N calls for a session
    db.address_history(address)           # all calls mentioning an address
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger("storage")

_DEFAULT_DB = Path(__file__).parent / "sparx.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    call_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    ts                  TEXT NOT NULL,

    -- ASR
    transcript          TEXT,
    language_code       TEXT,
    language_name       TEXT,
    translated          INTEGER NOT NULL DEFAULT 0,

    -- Emotion
    emotion             TEXT,
    distress_score      REAL,
    emotion_markers     TEXT,       -- JSON list
    is_crisis           INTEGER NOT NULL DEFAULT 0,

    -- Direction engine
    decision            TEXT,
    categories          TEXT,       -- JSON list
    question            TEXT,
    direction_response  TEXT,
    confidence          REAL,

    -- NemoClaw resolution
    resolution_source   TEXT,
    resolution_response TEXT,
    form_name           TEXT,
    form_url            TEXT,

    -- Displacement
    address             TEXT,
    displacement_risk   TEXT,
    displacement_score  REAL,
    displacement_alert  TEXT,
    violations_count    INTEGER,
    evictions_count     INTEGER,
    corporate_owner     INTEGER,

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_calls_session  ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_ts       ON calls(ts);
CREATE INDEX IF NOT EXISTS idx_calls_address  ON calls(address);
CREATE INDEX IF NOT EXISTS idx_calls_decision ON calls(decision);
CREATE INDEX IF NOT EXISTS idx_calls_emotion  ON calls(emotion);
"""


class Storage:
    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._path = str(db_path)
        self._init_db()
        LOGGER.info("Storage ready at %s", self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._tx() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_call(self, session_id: str | None, payload: dict) -> int:
        """
        Persist one /transcribe result.
        Returns the new call row id.

        payload is the full dict returned by the /transcribe endpoint.
        """
        now = datetime.now(timezone.utc).isoformat()

        direction    = payload.get("direction", {})
        resolution   = payload.get("resolution", {})
        lang         = payload.get("language", {})
        emotion      = payload.get("emotion", {})
        displacement = payload.get("displacement", {})
        signals      = displacement.get("signals", {})

        form_finder  = resolution.get("form_finder") or {}

        with self._tx() as conn:
            # Upsert session
            if session_id:
                conn.execute(
                    """
                    INSERT INTO sessions(session_id, first_seen, last_seen, call_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_seen  = excluded.last_seen,
                        call_count = call_count + 1
                    """,
                    (session_id, now, now),
                )

            # Insert call record
            cur = conn.execute(
                """
                INSERT INTO calls (
                    session_id, ts,
                    transcript, language_code, language_name, translated,
                    emotion, distress_score, emotion_markers, is_crisis,
                    decision, categories, question, direction_response, confidence,
                    resolution_source, resolution_response, form_name, form_url,
                    address, displacement_risk, displacement_score, displacement_alert,
                    violations_count, evictions_count, corporate_owner
                ) VALUES (
                    ?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,?
                )
                """,
                (
                    session_id, now,
                    payload.get("transcription"),
                    lang.get("code"),
                    lang.get("detected"),
                    int(lang.get("translated", False)),

                    emotion.get("emotion"),
                    emotion.get("distress_score"),
                    json.dumps(emotion.get("markers", [])),
                    int(emotion.get("is_crisis", False)),

                    direction.get("decision"),
                    json.dumps(direction.get("categories", [])),
                    direction.get("question"),
                    direction.get("response"),
                    direction.get("confidence"),

                    resolution.get("source"),
                    resolution.get("response"),
                    form_finder.get("form_name"),
                    form_finder.get("form_url"),

                    displacement.get("address"),
                    displacement.get("risk_level"),
                    displacement.get("risk_score"),
                    displacement.get("alert"),
                    signals.get("violations", 0),
                    signals.get("evictions", 0),
                    int(signals.get("corporate_owner", False)),
                ),
            )
            row_id = cur.lastrowid

        LOGGER.info("Saved call id=%d session=%s decision=%s",
                    row_id, session_id or "anon", direction.get("decision"))
        return row_id

    def session_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """Return the last `limit` calls for a session, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, transcript, emotion, distress_score, decision,
                       question, direction_response, resolution_source,
                       form_name, address, displacement_risk
                FROM calls
                WHERE session_id = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def address_history(self, address: str, limit: int = 50) -> list[dict]:
        """Return all calls that mention a given address."""
        like = f"%{address.lower()}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, transcript, decision, question,
                       displacement_risk, displacement_score,
                       violations_count, evictions_count
                FROM calls
                WHERE lower(address) LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_crises(self, hours: int = 24) -> list[dict]:
        """Return all crisis-level calls in the last N hours."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, session_id, transcript, emotion,
                       distress_score, direction_response
                FROM calls
                WHERE is_crisis = 1
                  AND ts >= datetime('now', ?)
                ORDER BY ts DESC
                """,
                (f"-{hours} hours",),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Quick summary stats — useful for a future admin dashboard."""
        with self._connect() as conn:
            total_calls    = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            crisis_calls   = conn.execute("SELECT COUNT(*) FROM calls WHERE is_crisis=1").fetchone()[0]
            top_decision   = conn.execute(
                "SELECT decision, COUNT(*) c FROM calls GROUP BY decision ORDER BY c DESC LIMIT 1"
            ).fetchone()
            high_disp      = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE displacement_risk IN ('high','critical')"
            ).fetchone()[0]

        return {
            "total_calls":    total_calls,
            "total_sessions": total_sessions,
            "crisis_calls":   crisis_calls,
            "top_decision":   dict(top_decision) if top_decision else {},
            "high_displacement_calls": high_disp,
        }
