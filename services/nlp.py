"""
Lightweight NLP entity extraction.
Pulls structured data (income, employer, loan purpose, amount, tenure, consent)
out of the live transcript without sending data to an external NLP API.
"""
import re

INCOME_PATTERNS = [
    r'(?:income|earn(?:ing)?s?|salary|sallery|sallary|make|making)[^\d]{0,30}(?:rs\.?|inr|₹)?\s*(\d[\d,]*)\s*(k|thousand|lakh|lakhs|cr|crore|crores)?',
    r'(?:rs\.?|inr|₹)\s*(\d[\d,]*)\s*(k|thousand|lakh|lakhs)?\s*(?:per month|a month|monthly|/month|p\.?m\.?)',
    r'(\d[\d,]*)\s*(k|thousand|lakh|lakhs)?\s*(?:per month|a month|monthly|/month|p\.?m\.?)',
]

LOAN_AMOUNT_PATTERNS = [
    r'(?:borrow|loan(?: of)?|need|want|require|looking for)[^\d]{0,30}(?:rs\.?|inr|₹)?\s*(\d[\d,]*)\s*(k|thousand|lakh|lakhs|cr|crore)?',
    r'(?:rs\.?|inr|₹)\s*(\d[\d,]*)\s*(k|thousand|lakh|lakhs)?\s*(?:loan|please|amount)',
]

TENURE_PATTERNS = [
    r'(?:over|for|tenure of|repay in|term of)\s*(\d+)\s*(months?|years?|yrs?|mos?)',
    r'(\d+)\s*[-]?\s*(months?|years?|yrs?)\s*(?:tenure|term|loan)',
]

EMPLOYER_PATTERNS = [
    r"(?:work(?:ing)?(?:\s+at|\s+for|\s+as|\s+with)?|employed (?:at|by|with)|company is|firm is|i'?m at|with)\s+([A-Z][A-Za-z0-9&\.\-\s]{1,40}?)(?:[\.,\;]| as | and | since | for |$)",
    r"(?:^|\s)([A-Z][A-Za-z]{2,20}(?:\s[A-Z][A-Za-z]{2,20}){0,3})\s+(?:Pvt|Ltd|Limited|Inc|Corp|LLP|Group|Bank|Solutions|Technologies|Services)",
]

OCCUPATION_KEYWORDS = [
    'engineer','developer','manager','analyst','consultant','doctor','teacher',
    'accountant','designer','architect','officer','executive','salesperson',
    'sales','nurse','lawyer','professor','professional','technician','operator',
    'pharmacist','dentist','contractor','freelancer','entrepreneur','founder',
    'business owner','shopkeeper','self employed','self-employed'
]

PURPOSE_KEYWORDS = {
    'home': ['home','house','flat','apartment','renovation','interior','property'],
    'wedding': ['wedding','marriage','shaadi','engagement'],
    'medical': ['medical','hospital','treatment','surgery','health'],
    'education': ['education','college','tuition','course','school','university','study'],
    'business': ['business','shop','expansion','inventory','equipment','startup'],
    'travel': ['travel','vacation','trip','holiday','foreign'],
    'vehicle': ['car','bike','vehicle','two-wheeler','automobile','scooter'],
    'debt-consolidation': ['consolidat','existing loan','credit card','debt'],
    'personal': ['personal','emergency','family','urgent','expense'],
}


def _to_int(num_str: str, multiplier_word: str = "") -> int:
    """Normalize 'fifty thousand', '50k', '5 lakh', '50,000' to integer rupees."""
    n = int(num_str.replace(",", "").strip())
    m = (multiplier_word or "").lower().strip()
    if m in ("k", "thousand"):
        n *= 1000
    elif m in ("lakh", "lakhs"):
        n *= 100000
    elif m in ("cr", "crore", "crores"):
        n *= 10000000
    return n


def extract_income(text: str) -> int | None:
    if not text:
        return None
    t = text.lower()
    for pat in INCOME_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            try:
                groups = m.groups()
                num = groups[0]
                mult = groups[1] if len(groups) > 1 else ""
                val = _to_int(num, mult or "")
                # Heuristic: if a tiny number was given, e.g. "50" without unit, treat as thousands
                if val < 1000 and (mult or "") == "":
                    val *= 1000
                if 5000 <= val <= 5000000:
                    return val
            except Exception:
                continue
    return None


def extract_income_from_answer(text: str) -> int | None:
    """
    Bare-number fallback for the income answer: handles 'fifty thousand',
    '50000', '50k', '5 lakh', '₹50000', etc. Used when the user is
    answering the income question directly and may not say the word 'income'.
    """
    if not text:
        return None
    t = text.lower().strip()
    # word-form numbers
    word_map = {
        "ten thousand": 10000, "fifteen thousand": 15000, "twenty thousand": 20000,
        "twenty five thousand": 25000, "thirty thousand": 30000, "forty thousand": 40000,
        "fifty thousand": 50000, "sixty thousand": 60000, "seventy thousand": 70000,
        "seventy five thousand": 75000, "eighty thousand": 80000, "ninety thousand": 90000,
        "one lakh": 100000, "1 lakh": 100000, "one and a half lakh": 150000,
        "two lakh": 200000, "2 lakh": 200000, "three lakh": 300000, "five lakh": 500000,
    }
    for w, n in word_map.items():
        if w in t:
            return n
    # numeric with optional unit
    m = re.search(r'(\d[\d,]*)\s*(k|thousand|lakh|lakhs|l|cr|crore)?', t, re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1).replace(",", ""))
            u = (m.group(2) or "").lower().strip()
            if u in ("k", "thousand"):
                n *= 1000
            elif u in ("l", "lakh", "lakhs"):
                n *= 100000
            elif u in ("cr", "crore"):
                n *= 10000000
            elif n < 1000:
                # bare small number → assume thousands ("fifty" → 50000)
                n *= 1000
            if 5000 <= n <= 5000000:
                return n
        except Exception:
            pass
    return None


def extract_age(text: str) -> int | None:
    """Extract a declared age from a phrase like 'I am 32', '32 years old', 'thirty two'."""
    if not text:
        return None
    t = text.lower()
    m = re.search(r'(?:i am|i\'m|my age is|age is|age)\s+(\d{2})', t)
    if not m:
        m = re.search(r'\b(\d{2})\s*(?:years?\s*old|yrs?|year|yr)\b', t)
    if not m:
        m = re.search(r'\b(\d{2})\b', t)
    if m:
        try:
            n = int(m.group(1))
            if 18 <= n <= 80:
                return n
        except Exception:
            pass
    word_age = {
        "eighteen":18, "nineteen":19, "twenty":20, "twenty one":21, "twenty two":22,
        "twenty three":23, "twenty four":24, "twenty five":25, "twenty six":26,
        "twenty seven":27, "twenty eight":28, "twenty nine":29, "thirty":30,
        "thirty one":31, "thirty two":32, "thirty three":33, "thirty four":34,
        "thirty five":35, "thirty six":36, "thirty seven":37, "thirty eight":38,
        "thirty nine":39, "forty":40, "forty five":45, "fifty":50, "fifty five":55,
        "sixty":60, "sixty five":65,
    }
    for w, n in word_age.items():
        if w in t:
            return n
    return None


def extract_loan_amount(text: str) -> int | None:
    if not text:
        return None
    t = text.lower()
    for pat in LOAN_AMOUNT_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            try:
                groups = m.groups()
                num = groups[0]
                mult = groups[1] if len(groups) > 1 else ""
                val = _to_int(num, mult or "")
                if val < 1000 and (mult or "") == "":
                    val *= 1000
                if 10000 <= val <= 50000000:
                    return val
            except Exception:
                continue
    return None


def extract_tenure_months(text: str) -> int | None:
    if not text:
        return None
    t = text.lower()
    for pat in TENURE_PATTERNS:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                unit = m.group(2)
                if unit and unit.startswith("year") or unit and unit.startswith("yr"):
                    n *= 12
                if 6 <= n <= 84:
                    return n
            except Exception:
                continue
    return None


def extract_employer(text: str) -> str | None:
    if not text:
        return None
    for pat in EMPLOYER_PATTERNS:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip(" ,.;")
            if 2 <= len(cand) <= 60:
                return cand
    # Occupation fallback
    t = text.lower()
    for kw in OCCUPATION_KEYWORDS:
        if kw in t:
            return kw.title()
    return None


def extract_loan_purpose(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    for label, kws in PURPOSE_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return label
    return None


def detect_consent(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    affirm = ["yes", "i consent", "i agree", "i accept", "absolutely",
              "of course", "sure", "confirm", "go ahead", "i do"]
    deny = ["no", "don't", "do not", "refuse", "decline"]
    has_yes = any(a in t for a in affirm)
    has_no = any(d in t.split() for d in deny)
    return has_yes and not has_no


def extract_name(text: str) -> str | None:
    """Best-effort name extraction from a self-introduction."""
    if not text:
        return None
    patterns = [
        r"(?:my name is|i am|i'm|this is|name'?s)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if 2 <= len(cand) <= 60:
                return cand.title()
    # Two-three capitalized words at the start
    m = re.match(r"\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    if m:
        return m.group(1)
    return None


def extract_all(transcript_lines: list[dict]) -> dict:
    """
    transcript_lines: [{speaker, text, timestamp, question_id}, ...]
    Returns a dict of structured entities.
    """
    customer_text = " ".join(
        line.get("text", "") for line in transcript_lines if line.get("speaker") == "customer"
    )
    full_text = " ".join(line.get("text", "") for line in transcript_lines)

    # Per-question answer text (more reliable than scanning the full transcript)
    answers: dict[str, str] = {}
    for line in transcript_lines:
        if line.get("speaker") == "customer" and line.get("question_id"):
            qid = line["question_id"]
            answers[qid] = (answers.get(qid, "") + " " + (line.get("text") or "")).strip()

    # Income: try whole-transcript first, then fall back to the income answer alone
    income = extract_income(customer_text)
    if income is None and answers.get("income"):
        income = extract_income_from_answer(answers["income"])

    # Age: pull from the dedicated age question first, then fall back to a transcript-wide scan
    age_declared = extract_age(answers.get("age", "")) or extract_age(customer_text)

    # Consent: prefer the consent-tagged line if we have it
    consent_text = answers.get("consent", customer_text)
    consent_given = detect_consent(consent_text)

    return {
        "name": extract_name(customer_text),
        "age_declared": age_declared,
        "employer": extract_employer(customer_text),
        "monthly_income_inr": income,
        "loan_amount_requested_inr": extract_loan_amount(customer_text),
        "tenure_months": extract_tenure_months(customer_text),
        "loan_purpose": extract_loan_purpose(customer_text),
        "consent_given": consent_given,
        "raw_customer_text_chars": len(customer_text),
        "raw_full_text_chars": len(full_text),
    }
