"""
LLM intelligence layer.

Reads the full transcript and outputs:
- persona classification (risk-averse / growth-oriented / etc.)
- intent + sentiment
- soft signals (hesitation, inconsistency)
- advisory only — never overrides the policy engine.

Uses Groq (llama-3.3-70b) when GROQ_API_KEY is set; otherwise a deterministic
template-driven fallback so the prototype runs anywhere with zero setup.
"""
import os
import json
import re

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
_groq = None

if GROQ_KEY:
    try:
        from groq import Groq
        _groq = Groq(api_key=GROQ_KEY)
        print("[llm] Groq client initialized — using llama-3.3-70b-versatile")
    except Exception as e:
        print(f"[llm] Groq init failed: {e!r} — using fallback")
else:
    print("[llm] GROQ_API_KEY not set — using deterministic fallback")


SYSTEM_PROMPT = """You are an analyst for Poonawalla Fincorp's Loan Wizard.
Read a customer's video-call transcript and emit a JSON object describing
the customer's persona, intent, sentiment, and any soft risk signals.

Strict rules:
- Output VALID JSON only, no prose, no code fences.
- Your output is advisory. The deterministic policy engine, not you, decides eligibility.
- Use the exact schema given in the user message.
"""


def analyze_transcript(transcript_lines: list[dict], entities: dict) -> dict:
    """
    transcript_lines: [{speaker, text, ts}, ...]
    entities: dict of structured fields already extracted by the NLP layer
    """
    if _groq:
        try:
            return _groq_analyze(transcript_lines, entities)
        except Exception as e:
            print(f"[llm] Groq call failed: {e!r} — falling back")
    return _fallback_analyze(transcript_lines, entities)


def _groq_analyze(transcript_lines: list[dict], entities: dict) -> dict:
    transcript_text = "\n".join(
        f"{t.get('speaker','?').upper()}: {t.get('text','')}" for t in transcript_lines
    )
    user_msg = f"""TRANSCRIPT:
{transcript_text}

EXTRACTED ENTITIES:
{json.dumps(entities, indent=2, default=str)}

Return JSON with EXACTLY this schema:
{{
  "persona": "<short label, e.g. 'Risk-averse salaried professional'>",
  "persona_reasoning": "<one sentence>",
  "intent": "<what the customer actually wants — short phrase>",
  "sentiment": "<positive | neutral | cautious | negative>",
  "confidence": <0.0 to 1.0>,
  "soft_flags": ["<any hesitation, inconsistency, or unusual signal>", "..."],
  "ranking_preference": "<balanced | lowest_emi | shortest_tenure | maximum_amount>",
  "summary": "<two-sentence summary for the audit log>"
}}"""

    resp = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    parsed = json.loads(txt)
    parsed["_source"] = "groq:llama-3.3-70b-versatile"
    return parsed


def _fallback_analyze(transcript_lines: list[dict], entities: dict) -> dict:
    """Deterministic fallback so the demo works without an API key."""
    customer_text = " ".join(
        t.get("text", "") for t in transcript_lines if t.get("speaker") == "customer"
    ).lower()

    soft_flags = []

    # Sentiment heuristics
    cautious_terms = ["worried", "afraid", "scared", "tight", "unsure",
                      "not sure", "minimum", "just enough", "concerned"]
    growth_terms = ["maximum", "as much as", "highest", "expand", "growth",
                    "scale", "biggest", "more"]
    cautious_hits = sum(1 for k in cautious_terms if k in customer_text)
    growth_hits = sum(1 for k in growth_terms if k in customer_text)

    if cautious_hits > growth_hits:
        sentiment = "cautious"
        ranking = "lowest_emi"
        persona_label = "Risk-averse"
    elif growth_hits > cautious_hits:
        sentiment = "positive"
        ranking = "maximum_amount"
        persona_label = "Growth-oriented"
    else:
        sentiment = "neutral"
        ranking = "balanced"
        persona_label = "Balanced"

    employer = entities.get("employer") or ""
    if "self" in employer.lower() or "freelan" in employer.lower():
        persona = f"{persona_label} self-employed applicant"
    elif employer:
        persona = f"{persona_label} salaried professional"
    else:
        persona = f"{persona_label} applicant"

    # Soft flag: short responses
    customer_chars = entities.get("raw_customer_text_chars") or 0
    if customer_chars < 80:
        soft_flags.append("Very short customer responses — limited intent signal.")

    # Soft flag: missing tenure when amount given
    if entities.get("loan_amount_requested_inr") and not entities.get("tenure_months"):
        soft_flags.append("Customer requested an amount but did not specify tenure.")

    # Soft flag: income vs amount sanity
    income = entities.get("monthly_income_inr") or 0
    amt = entities.get("loan_amount_requested_inr") or 0
    if income and amt and amt > income * 40:
        soft_flags.append(
            f"Requested amount ₹{amt:,} is >40x monthly income — unusually high."
        )

    intent_purpose = entities.get("loan_purpose") or "personal use"
    intent = f"Personal loan for {intent_purpose}"
    if amt:
        intent += f", around ₹{amt:,}"

    summary = (
        f"Applicant presents as a {persona.lower()}. "
        f"Sentiment {sentiment}; ranking preference {ranking}. "
        f"{len(soft_flags)} soft flag(s) detected."
    )

    return {
        "persona": persona,
        "persona_reasoning": (
            f"Customer text contained {cautious_hits} cautious cue(s) and "
            f"{growth_hits} growth cue(s); employer signal '{employer or 'none'}'."
        ),
        "intent": intent,
        "sentiment": sentiment,
        "confidence": 0.65,
        "soft_flags": soft_flags,
        "ranking_preference": ranking,
        "summary": summary,
        "_source": "fallback:deterministic",
    }


def llm_status() -> dict:
    return {
        "provider": "groq" if _groq else "fallback",
        "model": "llama-3.3-70b-versatile" if _groq else "deterministic",
    }
