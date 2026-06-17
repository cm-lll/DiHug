import re, hashlib

COMMON_REPLACEMENTS = {
    # ---- specific ----
    r"\bmit\b": "massachusetts institute of technology",
    r"\bstanford\b": "stanford university",
    r"\buc berkeley\b": "university of california berkeley",

    # ---- generic ----
    r"\buniv\b": "university",
    r"\binst\b": "institute",
    r"\btech\b": "technology",
    r"\bdept\b": "department",
    r"\bdept of\b": "department of",
}

def canonicalize_org_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[.,;:()\"']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in COMMON_REPLACEMENTS.items():
        s = re.sub(pat, repl, s)
    s = re.sub(r"\b(dept|department|school|faculty|laboratory|lab)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def org_id_from_name(name: str) -> str:
    if not name:
        return ""
    canon = canonicalize_org_name(name)
    return hashlib.md5(canon.encode("utf8")).hexdigest()
