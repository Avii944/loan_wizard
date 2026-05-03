"""
Loan Wizard — Flask prototype
=============================

Implements the 9-step Agentic AI Video-Call Loan Origination flow:

  1. Customer entry via campaign link
  2. Video + audio + geo + session metadata capture
  3. Speech-to-text + verbal consent capture
  4. CV-based age estimation + liveness
  5. Auto-fill from extracted entities
  6. Risk + policy evaluation (deterministic gates + ML-style propensity)
  7. LLM intelligence layer (advisory, never overrides policy)
  8. Personalized offer generation
  9. Central audit + integrity hash

Run: python app.py  (then open http://localhost:5000)
"""
import os
import uuid
import json
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, abort
)
from dotenv import load_dotenv

load_dotenv()

from services import nlp, risk_engine, llm, offer_generator, audit

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "loan-wizard-dev-secret")

# In-memory transient session state (cleared on restart; persistent data is in audit DB)
SESSIONS: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Page routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing - simulated campaign link page."""
    return render_template("entry.html")


@app.route("/start")
def start_session():
    """
    Step 1: Customer enters via campaign link.
    Generates a session id and routes to the pre-call permissions page.
    """
    sid = "LW-" + uuid.uuid4().hex[:8].upper()
    SESSIONS[sid] = {
        "id": sid,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "started",
        "transcript": [],
        "entities": {},
        "face": {},
        "geo": None,
        "device": None,
    }
    return redirect(url_for("precall", sid=sid))


@app.route("/precall/<sid>")
def precall(sid):
    if sid not in SESSIONS:
        abort(404)
    return render_template("precall.html", sid=sid)


@app.route("/session/<sid>")
def video_session(sid):
    if sid not in SESSIONS:
        abort(404)
    return render_template("session.html", sid=sid, llm_status=llm.llm_status())


@app.route("/offers/<sid>")
def offers_page(sid):
    s = audit.get_session(sid)
    if not s:
        abort(404)
    return render_template("offers.html", session=s)


@app.route("/audit")
def audit_dashboard():
    sessions = audit.list_sessions(limit=100)
    return render_template("audit.html", sessions=sessions, detail=None)


@app.route("/audit/<sid>")
def audit_detail(sid):
    s = audit.get_session(sid)
    if not s:
        abort(404)
    return render_template("audit.html", sessions=audit.list_sessions(50), detail=s)


# ──────────────────────────────────────────────────────────────────────────────
# JSON API
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/session/<sid>/metadata", methods=["POST"])
def api_metadata(sid):
    """Step 2: persist geo, device, IP, and session metadata at session start."""
    if sid not in SESSIONS:
        abort(404)
    body = request.get_json(force=True, silent=True) or {}
    geo = body.get("geo") or {}
    device = body.get("device") or {}
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    user_agent = request.headers.get("User-Agent", "")

    SESSIONS[sid]["geo"] = geo
    SESSIONS[sid]["device"] = device
    SESSIONS[sid]["ip"] = ip
    SESSIONS[sid]["user_agent"] = user_agent
    SESSIONS[sid]["status"] = "metadata_captured"

    audit.create_session(sid, geo, device, ip, user_agent)
    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/session/<sid>/transcript", methods=["POST"])
def api_transcript(sid):
    """Step 3: append a transcript line (speaker + text + ts + flags)."""
    if sid not in SESSIONS:
        abort(404)
    body = request.get_json(force=True, silent=True) or {}
    line = {
        "speaker": body.get("speaker", "customer"),
        "text": body.get("text", ""),
        "ts": body.get("ts") or datetime.utcnow().isoformat() + "Z",
        "is_consent": bool(body.get("is_consent", False)),
        "question_id": body.get("question_id"),
    }
    SESSIONS[sid]["transcript"].append(line)
    audit.log_event(sid, "transcript.line", line)
    return jsonify({"ok": True, "lines": len(SESSIONS[sid]["transcript"])})


@app.route("/api/session/<sid>/face", methods=["POST"])
def api_face(sid):
    """Step 4: record CV outputs (age, gender, liveness)."""
    if sid not in SESSIONS:
        abort(404)
    body = request.get_json(force=True, silent=True) or {}
    face = {
        "age_estimated": body.get("age_estimated"),
        "age_declared": body.get("age_declared"),
        "gender": body.get("gender"),
        "expression": body.get("expression"),
        "liveness_passed": body.get("liveness_passed", True),
        "liveness_challenges": body.get("liveness_challenges", []),
        "models_loaded": body.get("models_loaded", True),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    SESSIONS[sid]["face"] = face
    audit.log_event(sid, "face.captured", face)
    return jsonify({"ok": True})


@app.route("/api/session/<sid>/finalize", methods=["POST"])
def api_finalize(sid):
    """
    Final pipeline: NLP extract → bureau lookup → policy gates → propensity
    → LLM advisory → offer generation → audit log + integrity hash.
    """
    if sid not in SESSIONS:
        abort(404)
    sess = SESSIONS[sid]
    body = request.get_json(force=True, silent=True) or {}

    # 5. Auto-fill / NLP extraction (merges any client-provided overrides)
    extracted = nlp.extract_all(sess["transcript"])
    overrides = body.get("entities_override") or {}
    entities = {**extracted, **{k: v for k, v in overrides.items() if v not in (None, "")}}

    audit.log_event(sid, "nlp.extracted", entities)

    # Merge declared age (from NLP / user override) into the face dict so the
    # policy engine can cross-check it against the CV-estimated age and use it
    # as a fallback when face detection didn't yield an age.
    face = dict(sess.get("face") or {})
    if entities.get("age_declared") and not face.get("age_declared"):
        try:
            face["age_declared"] = int(entities["age_declared"])
        except (TypeError, ValueError):
            pass

    # 6. Risk + policy
    bureau_profile_hint = body.get("bureau_profile") or "_default"
    bureau = risk_engine.lookup_bureau(bureau_profile_hint)
    policy = risk_engine.evaluate_policy(entities, face,
                                         sess.get("geo") or {}, bureau)
    propensity = risk_engine.propensity_score(entities, face, bureau)
    audit.log_event(sid, "policy.evaluated", policy)
    audit.log_event(sid, "propensity.scored", propensity)

    # 7. LLM advisory
    llm_out = llm.analyze_transcript(sess["transcript"], entities)
    audit.log_event(sid, "llm.analyzed", llm_out)

    # 8. Offers — only if policy passes
    offers = []
    decision = "approved"
    if not policy["eligible"]:
        decision = "rejected"
    else:
        offers = offer_generator.build_offers(entities, propensity, llm_out.get("persona"))
        if not offers:
            decision = "no_offer_fits_dti_cap"
    audit.log_event(sid, "offers.generated", {"count": len(offers), "decision": decision})

    # 9. Audit finalize with integrity hash
    integrity_hash = audit.finalize_session(
        sid, sess["transcript"], entities, sess.get("face") or {},
        policy, propensity, llm_out, offers, decision,
    )

    SESSIONS[sid].update({
        "status": "completed",
        "entities": entities,
        "policy": policy,
        "propensity": propensity,
        "llm": llm_out,
        "offers": offers,
        "decision": decision,
        "bureau": bureau,
        "integrity_hash": integrity_hash,
    })
    return jsonify({
        "ok": True, "session_id": sid, "decision": decision,
        "policy": policy, "propensity": propensity, "llm": llm_out,
        "offers": offers, "entities": entities, "bureau": bureau,
        "integrity_hash": integrity_hash,
        "redirect": url_for("offers_page", sid=sid),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "llm": llm.llm_status(),
        "ts": datetime.utcnow().isoformat() + "Z",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Template filters
# ──────────────────────────────────────────────────────────────────────────────

@app.template_filter("inr")
def inr(v):
    try:
        return f"₹{int(v):,}"
    except (TypeError, ValueError):
        return "—"


@app.template_filter("prettyjson")
def prettyjson(v):
    if v is None:
        return ""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return v
    return json.dumps(v, indent=2, default=str)


# Initialize the audit DB at import time so it works under both `python app.py`
# (dev) and `gunicorn app:app` (production on Render/Railway/etc.).
audit.init_db()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"\n  Loan Wizard — Flask prototype")
    print(f"  → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
