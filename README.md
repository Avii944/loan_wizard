# Loan Wizard

End-to-end working prototype of the Agentic AI Video-Call Loan Origination System
described in Poonawalla Fincorp Problem Statement 3.

A single live video call captures, verifies, scores, and generates a personalized
loan offer in under three minutes — implementing every functional requirement
in the problem statement.

---
#🚀 Live Demo

Experience the working prototype of the application here:
👉 https://web-production-6e0fb.up.railway.app/

## What it implements (mapped to the problem statement)

| § | Requirement | Where it lives |
|---|---|---|
| 2.1.1 | Customer entry via campaign link | `templates/entry.html` → `/start` route |
| 2.1.2 | Video + audio + geo + session metadata | `templates/precall.html`, `static/js/session.js` (WebRTC) |
| 2.1.3 | Speech-to-text + verbal consent capture | `static/js/session.js` (Web Speech API) |
| 2.1.4 | CV-based age estimation + liveness | `static/js/session.js` (face-api.js, blink challenge) |
| 2.1.5 | Auto-fill / NLP entity extraction | `services/nlp.py` (regex + heuristics) |
| 2.1.6 | Risk + policy evaluation | `services/risk_engine.py` (deterministic gates + propensity score) |
| 2.1.7 | LLM intelligence layer (advisory only) | `services/llm.py` (Groq llama-3.3-70b + deterministic fallback) |
| 2.1.8 | Offer generation | `services/offer_generator.py` (3 ranked variants, EMI calc) |
| 2.1.9 | Central audit + integrity hash | `services/audit.py` (SQLite + SHA-256) |

Every judging criterion (3.1 – 3.5) is addressed in the architecture and
visible in the audit dashboard at `/audit`.

---

## Quick start

### 1. Install
```bash
cd loan_app
pip install -r requirements.txt
cp .env.example .env
```

### 2. (Optional) Add a Groq key
The system has a built-in deterministic fallback so it runs with zero API keys.
To use the real LLM (Llama 3.3 70B), get a free key at
[console.groq.com](https://console.groq.com/keys) and put it in `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

### 3. Run
```bash
python app.py
```
Open **http://localhost:5000** in **Chrome or Edge** (Web Speech API is required
for live STT).

---

## Demo flow

1. **Home** — click "Start my loan call".
2. **Pre-call** — grant camera, microphone, and geo-location.
3. **Live session**:
   - Click the mic, answer each of 6 questions naturally, click *Next*.
   - The right pane shows the 8-agent pipeline lighting up live.
   - Live entities update in the bottom-right card as you speak.
   - Blink at least twice — that's the liveness challenge.
4. **Offers** — three personalized loan offers with rationale.
5. **Audit** — visit `/audit` to see the entire session record, including the
   SHA-256 integrity hash and every event timestamp.

---

## Architecture

```
Browser                              Flask backend                    Storage
─────────                            ─────────────                    ───────
WebRTC video                         /api/session/<sid>/metadata  ─►
Web Speech API STT  ─── transcript ─► /api/session/<sid>/transcript ─►
face-api.js CV     ─── face data  ─► /api/session/<sid>/face       ─►  SQLite
                                                                       audit_repo/
                                     /api/session/<sid>/finalize          loan_wizard.db
                                       │
                                       ├─► services/nlp.py            ◄── audit events
                                       ├─► services/risk_engine.py        with timestamps
                                       ├─► services/llm.py
                                       └─► services/offer_generator.py
                                       └─► services/audit.py (+ SHA-256 hash)
```

**Dual-engine principle.** The deterministic policy engine has the final word
on eligibility. The LLM is *advisory* — it can shape ranking and persona, but
it cannot approve a loan that fails a policy gate. Every contribution to the
propensity score is itself recorded in the audit log, so any decision is
explainable.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Flask 3 (single-file `app.py` + `services/`) |
| Frontend | Vanilla JS + face-api.js (CDN) |
| STT | Web Speech API (browser-native) |
| CV | face-api.js (TinyFaceDetector + AgeGender + Landmark68 + Expression nets) |
| LLM | Groq `llama-3.3-70b-versatile` (with deterministic fallback) |
| Storage | SQLite (`audit_repo/loan_wizard.db`) |
| Compliance hash | SHA-256 over the full finalized session payload |

---

## File layout

```
loan_app/
├── app.py                      # All routes + pipeline orchestration
├── requirements.txt
├── .env.example
├── README.md
├── services/
│   ├── nlp.py                  # Entity extraction (income, employer, etc.)
│   ├── risk_engine.py          # Policy gates + propensity scoring
│   ├── llm.py                  # Groq + deterministic fallback
│   ├── offer_generator.py      # EMI calculation, persona-aware ranking
│   └── audit.py                # SQLite repository + SHA-256 integrity
├── data/
│   ├── policy_rules.json       # Eligibility thresholds
│   └── bureau_mock.json        # Mock CIBIL/CRIF profiles
├── static/
│   ├── css/style.css
│   └── js/session.js
├── templates/
│   ├── base.html
│   ├── entry.html              # Step 1: campaign landing
│   ├── precall.html            # Step 2: permissions
│   ├── session.html            # Steps 3-5: live call
│   ├── offers.html             # Steps 6-8: offer screen
│   └── audit.html              # Step 9: full audit dashboard
└── audit_repo/
    └── loan_wizard.db          # Auto-created on first run
```

---

## Browser support

| Browser | STT | Face-api | Verdict |
|---|---|---|---|
| Chrome 110+ | ✅ | ✅ | Full support |
| Edge 110+ | ✅ | ✅ | Full support |
| Firefox | ❌ (no Web Speech API) | ✅ | Limited — STT will fail |
| Safari | ⚠️ (partial) | ✅ | Limited |

For demos, **use Chrome.**

---

## Compliance notes

- **RBI V-CIP**: live video, geo capture, liveness verification, examiner-style
  conversation flow, full session recording in audit log.
- **DPDP Act 2023**: explicit verbal consent (timestamped, transcript-linked),
  data minimization, retention controlled by audit DB.
- **Tamper evidence**: SHA-256 over the finalized session means any byte
  modification changes the hash — useful for downstream regulatory queries.

---

## Customization

- **Eligibility rules**: edit `data/policy_rules.json` — no code changes needed.
- **Bureau profiles**: edit `data/bureau_mock.json` to simulate different
  applicants. Pass `bureau_profile` in finalize body, or call
  `risk_engine.lookup_bureau()` from your own integration code.
- **LLM provider**: swap Groq for OpenAI / Anthropic by editing `services/llm.py`.
- **Question script**: edit `QUESTIONS` in `static/js/session.js`.

---

## Author

Prepared for Poonawalla Fincorp Problem Statement 3 — Agentic AI Video-Call
Onboarding.
