"""
Deterministic risk + policy engine.
Implements the dual-engine principle: rules decide eligibility, scores influence offers.
"""
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "policy_rules.json", "r") as f:
    POLICY = json.load(f)
with open(DATA_DIR / "bureau_mock.json", "r") as f:
    BUREAU = json.load(f)


def lookup_bureau(profile_hint: str | None = None) -> dict:
    """
    In production this would call CRIF/CIBIL/Experian APIs by PAN/Aadhaar.
    Here we simulate by selecting a profile from the mock dataset.
    """
    if profile_hint and profile_hint in BUREAU:
        return dict(BUREAU[profile_hint])
    return dict(BUREAU["_default"])


def evaluate_policy(entities: dict, face: dict, geo: dict, bureau: dict) -> dict:
    """
    Hard eligibility gates. Returns {eligible: bool, violations: [str], warnings: [str]}.
    """
    violations: list[str] = []
    warnings: list[str] = []

    # Age check (uses CV-estimated age first, falls back to declared age)
    age = face.get("age_estimated") or face.get("age_declared")
    if age is None:
        warnings.append("Age could not be verified — manual review required.")
    else:
        try:
            age_int = int(age)
            if age_int < POLICY["min_age"]:
                violations.append(
                    f"Estimated age {age_int} is below minimum age {POLICY['min_age']}."
                )
            elif age_int > POLICY["max_age"]:
                violations.append(
                    f"Estimated age {age_int} exceeds maximum age {POLICY['max_age']}."
                )
        except (TypeError, ValueError):
            warnings.append("Age value could not be parsed.")

    # Age mismatch fraud signal (CV vs declared)
    if face.get("age_estimated") and face.get("age_declared"):
        try:
            diff = abs(int(face["age_estimated"]) - int(face["age_declared"]))
            if diff > 5:
                violations.append(
                    f"Age mismatch: CV estimated {face['age_estimated']}, "
                    f"declared {face['age_declared']} (diff {diff} years > 5 threshold)."
                )
        except (TypeError, ValueError):
            pass

    # Liveness
    if face.get("liveness_passed") is False:
        violations.append("Liveness check failed — possible spoofing attempt.")

    # Income gate
    income = entities.get("monthly_income_inr") or 0
    if income < POLICY["min_monthly_income_inr"]:
        violations.append(
            f"Declared income ₹{income:,} below minimum ₹{POLICY['min_monthly_income_inr']:,}."
        )

    # Geo check
    if geo and (geo.get("lat") is not None) and (geo.get("lng") is not None):
        bbox = POLICY["allowed_country_bbox"]
        if not (
            bbox["lat_min"] <= geo["lat"] <= bbox["lat_max"]
            and bbox["lng_min"] <= geo["lng"] <= bbox["lng_max"]
        ):
            violations.append(
                f"Geo-location ({geo['lat']:.3f}, {geo['lng']:.3f}) outside permitted region."
            )
    else:
        warnings.append("Geo-location not captured — flagged for review.")

    # Excluded PIN
    pin = (entities.get("pincode") or "").strip()
    if pin and any(pin.startswith(p) for p in POLICY["excluded_pin_prefixes"]):
        violations.append(f"PIN code {pin} is in excluded service area.")

    # Bureau gates
    if bureau.get("credit_score", 0) < POLICY["min_credit_score"]:
        violations.append(
            f"Bureau credit score {bureau.get('credit_score')} below minimum {POLICY['min_credit_score']}."
        )
    if bureau.get("dpd_30plus_last_12mo", 0) > POLICY["max_dpd_30plus_last_12mo"]:
        violations.append(
            f"Bureau shows {bureau['dpd_30plus_last_12mo']} DPD-30+ events in last 12 months "
            f"(max {POLICY['max_dpd_30plus_last_12mo']})."
        )

    # Consent
    if not entities.get("consent_given"):
        violations.append("Verbal consent not detected in transcript.")

    return {
        "eligible": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }


def propensity_score(entities: dict, face: dict, bureau: dict) -> dict:
    """
    ML-style score 0-100. Higher = better. Built from interpretable signals
    so each contribution is auditable. In production you'd swap this for a real model.
    """
    contributions = []
    score = 50.0

    # Bureau (40 pts)
    cs = bureau.get("credit_score", 0)
    if cs:
        bureau_pts = max(-10, min(40, (cs - 600) / 5))
        score += bureau_pts
        contributions.append(("bureau_credit_score", round(bureau_pts, 2),
                              f"Bureau score {cs}"))

    # Income (15 pts)
    inc = entities.get("monthly_income_inr") or 0
    if inc >= 200000:
        delta = 15
    elif inc >= 100000:
        delta = 12
    elif inc >= 50000:
        delta = 8
    elif inc >= 25000:
        delta = 4
    else:
        delta = 0
    score += delta
    contributions.append(("declared_income", delta, f"Income ₹{inc:,}/mo"))

    # DPD penalty (-15)
    dpd = bureau.get("dpd_30plus_last_12mo", 0)
    if dpd:
        delta = -min(15, dpd * 5)
        score += delta
        contributions.append(("bureau_dpd", delta, f"{dpd} DPD-30+ events"))

    # Credit utilization
    util = bureau.get("credit_utilization_pct", 0)
    if util >= 80:
        delta = -8
    elif util >= 50:
        delta = -3
    elif util <= 30:
        delta = 4
    else:
        delta = 0
    if delta:
        score += delta
        contributions.append(("credit_utilization", delta, f"Utilization {util}%"))

    # Age sweet spot
    age = face.get("age_estimated") or face.get("age_declared")
    if age and 25 <= age <= 55:
        score += 4
        contributions.append(("age_band", 4, f"Age {age} in prime range"))

    # Employer signal
    employer = (entities.get("employer") or "").lower()
    stable_keywords = ("government", "bank", "tcs", "infosys", "wipro",
                       "accenture", "engineer", "doctor", "manager")
    if any(k in employer for k in stable_keywords):
        score += 4
        contributions.append(("employer_signal", 4, "Stable employer/role keyword"))

    # Liveness bonus
    if face.get("liveness_passed"):
        score += 2
        contributions.append(("liveness_passed", 2, "Liveness check passed"))

    score = max(20, min(95, round(score)))

    if score >= 75:
        band = "low_risk"
    elif score >= 55:
        band = "moderate_risk"
    else:
        band = "high_risk"

    return {
        "score": score,
        "band": band,
        "contributions": [
            {"feature": f, "delta": d, "explanation": e} for (f, d, e) in contributions
        ],
    }


def get_policy() -> dict:
    """Expose policy rules to other modules (offer generator etc.)."""
    return POLICY
