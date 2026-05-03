"""
Central audit repository (SQLite).
Stores the full session: metadata, transcript, entities, face data,
risk decisions, LLM output, offers, and a final hash for tamper-evidence.
"""
import json
import os
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / os.getenv(
    "AUDIT_DB", "audit_repo/loan_wizard.db"
)


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id     TEXT PRIMARY KEY,
            created_at     TEXT,
            updated_at     TEXT,
            status         TEXT,
            geo_lat        REAL,
            geo_lng        REAL,
            geo_accuracy   REAL,
            ip             TEXT,
            user_agent     TEXT,
            consent_ts     TEXT,
            consent_text   TEXT,
            entities_json  TEXT,
            face_json      TEXT,
            transcript_json TEXT,
            policy_json    TEXT,
            propensity_json TEXT,
            llm_json       TEXT,
            offers_json    TEXT,
            decision       TEXT,
            integrity_hash TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            ts         TEXT,
            event      TEXT,
            payload    TEXT
        )
        """)


def create_session(session_id: str, geo: dict | None, device: dict | None,
                   ip: str | None, user_agent: str | None):
    init_db()
    now = datetime.utcnow().isoformat() + "Z"
    geo = geo or {}
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, created_at, updated_at, status, geo_lat, geo_lng,
             geo_accuracy, ip, user_agent)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            session_id, now, now, "started",
            geo.get("lat"), geo.get("lng"), geo.get("accuracy"),
            ip, user_agent,
        ))
    log_event(session_id, "session.created", {
        "geo": geo, "device": device, "ip": ip, "user_agent": user_agent,
    })


def log_event(session_id: str, event: str, payload: dict | None = None):
    init_db()
    ts = datetime.utcnow().isoformat() + "Z"
    with _conn() as c:
        c.execute(
            "INSERT INTO audit_events (session_id, ts, event, payload) VALUES (?,?,?,?)",
            (session_id, ts, event, json.dumps(payload or {}, default=str)),
        )


def finalize_session(session_id: str, transcript: list, entities: dict,
                     face: dict, policy: dict, propensity: dict,
                     llm: dict, offers: list, decision: str):
    """Writes the analytical result and computes an integrity hash."""
    init_db()
    payload = {
        "session_id": session_id,
        "transcript": transcript,
        "entities": entities,
        "face": face,
        "policy": policy,
        "propensity": propensity,
        "llm": llm,
        "offers": offers,
        "decision": decision,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    consent_ts = next(
        (t["ts"] for t in transcript
         if t.get("speaker") == "customer" and t.get("is_consent")), None
    )
    consent_text = next(
        (t["text"] for t in transcript
         if t.get("speaker") == "customer" and t.get("is_consent")), None
    )
    now = datetime.utcnow().isoformat() + "Z"
    with _conn() as c:
        c.execute("""
            UPDATE sessions SET
              updated_at=?, status=?, consent_ts=?, consent_text=?,
              entities_json=?, face_json=?, transcript_json=?,
              policy_json=?, propensity_json=?, llm_json=?,
              offers_json=?, decision=?, integrity_hash=?
            WHERE session_id=?
        """, (
            now, "completed", consent_ts, consent_text,
            json.dumps(entities, default=str),
            json.dumps(face, default=str),
            json.dumps(transcript, default=str),
            json.dumps(policy, default=str),
            json.dumps(propensity, default=str),
            json.dumps(llm, default=str),
            json.dumps(offers, default=str),
            decision, digest, session_id,
        ))
    log_event(session_id, "session.finalized",
              {"decision": decision, "integrity_hash": digest})
    return digest


def get_session(session_id: str) -> dict | None:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        s = dict(row)
        events = c.execute(
            "SELECT * FROM audit_events WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        s["events"] = [dict(e) for e in events]
    for k in ("entities_json", "face_json", "transcript_json", "policy_json",
              "propensity_json", "llm_json", "offers_json"):
        if s.get(k):
            try:
                s[k.replace("_json", "")] = json.loads(s[k])
            except Exception:
                s[k.replace("_json", "")] = None
    return s


def list_sessions(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute("""
            SELECT session_id, created_at, status, decision,
                   geo_lat, geo_lng, integrity_hash
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]
