"""
Generate up to 3 ranked, personalized loan offers.
Honors policy gates, uses the propensity score for pricing,
and uses the LLM persona to rank ordering.
"""
import math
from .risk_engine import get_policy

POLICY = get_policy()


def emi(principal: float, annual_rate_pct: float, months: int) -> int:
    monthly = annual_rate_pct / 12 / 100
    if monthly == 0:
        return round(principal / months)
    return round(
        principal * monthly * (1 + monthly) ** months
        / ((1 + monthly) ** months - 1)
    )


def _round_to(amount: int, step: int = 5000) -> int:
    return max(step, int(round(amount / step) * step))


def build_offers(entities: dict, propensity: dict, persona: str | None) -> list[dict]:
    """
    Returns a list of up to 3 offers, ranked. The first one is the recommended.
    Each offer carries a plain-language rationale and the inputs that drove it.
    """
    income = entities.get("monthly_income_inr") or 25000
    band = propensity.get("band", "moderate_risk")
    rate_band = POLICY["interest_rate_bands"][band]
    base_rate = (rate_band["min"] + rate_band["max"]) / 2

    # Max principal from EMI/income cap
    max_emi = income * POLICY["max_emi_to_income_ratio"]
    # Ratio caps eligible amount as multiple of monthly income
    eligible_principal = min(
        income * POLICY["max_loan_to_income_ratio"],
        # back-solve from max_emi at base_rate, 36 months — safe ceiling
        _principal_from_emi(max_emi, base_rate, 36) * 1.0,
    )

    # Three structured variants
    variants = [
        # Balanced offer
        {
            "principal": _round_to(eligible_principal * 0.70),
            "tenure_months": 36,
            "rate_pct": round(base_rate, 2),
            "label": "Balanced — recommended",
        },
        # Short tenure / cheaper
        {
            "principal": _round_to(eligible_principal * 0.50),
            "tenure_months": 24,
            "rate_pct": round(rate_band["min"], 2),
            "label": "Lower interest, faster payoff",
        },
        # Maximum amount, longer tenure
        {
            "principal": _round_to(eligible_principal),
            "tenure_months": 48,
            "rate_pct": round(rate_band["max"], 2),
            "label": "Maximum amount, lower EMI",
        },
    ]

    offers = []
    for v in variants:
        emi_amount = emi(v["principal"], v["rate_pct"], v["tenure_months"])
        emi_to_income = emi_amount / income if income else 1
        # Skip if EMI exceeds the policy DTI cap
        if emi_to_income > POLICY["max_emi_to_income_ratio"]:
            continue
        offers.append({
            "label": v["label"],
            "principal_inr": v["principal"],
            "tenure_months": v["tenure_months"],
            "interest_rate_pct": v["rate_pct"],
            "emi_inr": emi_amount,
            "emi_to_income_pct": round(emi_to_income * 100, 1),
            "total_payable_inr": emi_amount * v["tenure_months"],
            "total_interest_inr": emi_amount * v["tenure_months"] - v["principal"],
            "rationale": _rationale(v, emi_amount, income, propensity, entities, persona),
        })

    # Re-rank based on persona
    offers = _rank_by_persona(offers, persona)
    if offers:
        offers[0]["recommended"] = True
        for o in offers[1:]:
            o["recommended"] = False
    return offers


def _principal_from_emi(target_emi: float, annual_rate_pct: float, months: int) -> float:
    monthly = annual_rate_pct / 12 / 100
    if monthly == 0:
        return target_emi * months
    return target_emi * ((1 + monthly) ** months - 1) / (monthly * (1 + monthly) ** months)


def _rationale(v: dict, emi_amount: int, income: int,
               propensity: dict, entities: dict, persona: str | None) -> str:
    purpose = entities.get("loan_purpose") or "personal needs"
    bits = [
        f"Principal of ₹{v['principal']:,} at {v['rate_pct']}% p.a. over "
        f"{v['tenure_months']} months yields an EMI of ₹{emi_amount:,}.",
        f"This is {round(emi_amount/income*100)}% of your declared monthly income — "
        f"within Poonawalla's {int(POLICY['max_emi_to_income_ratio']*100)}% DTI limit.",
        f"Pricing falls in the '{propensity['band'].replace('_',' ')}' band based on your "
        f"propensity score of {propensity['score']}/100.",
    ]
    if persona:
        bits.append(f"Aligned with your profile signal: {persona}.")
    if purpose:
        bits.append(f"Suited for stated purpose: {purpose}.")
    return " ".join(bits)


def _rank_by_persona(offers: list[dict], persona: str | None) -> list[dict]:
    if not offers or not persona:
        return offers
    p = persona.lower()
    if "risk-averse" in p or "conservative" in p or "concerned" in p:
        # Promote shortest-tenure / lowest-rate offer
        offers.sort(key=lambda o: (o["tenure_months"], o["interest_rate_pct"]))
    elif "growth" in p or "ambitious" in p or "maximum" in p:
        # Promote largest principal
        offers.sort(key=lambda o: -o["principal_inr"])
    # else: keep balanced first
    return offers
