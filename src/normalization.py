import re
import pandas as pd

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_NON_ALNUM_NOSPACE = re.compile(r"[^a-z0-9]")
_MULTI_SP = re.compile(r" +")

NAME_CANON = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "private": "pvt",
    "limited": "ltd",
    "&": "and",
}

ADDR_CANON = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "place": "pl",
    "court": "ct",
    "circle": "cir",
}


def _safe(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    return str(s)


def _basic(s):
    s = _safe(s).lower().replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    return _MULTI_SP.sub(" ", _WS.sub(" ", s)).strip()


def normalize_name(s):
    t = _basic(s)
    if not t:
        return ""
    return " ".join(NAME_CANON.get(w, w) for w in t.split(" "))


def normalize_address(s):
    t = _basic(s)  # punctuation already -> space; numbers/PINs preserved
    if not t:
        return ""
    return " ".join(ADDR_CANON.get(w, w) for w in t.split(" "))


def compact(s):
    """Alphanumeric-only, no spaces — for character-based matching."""
    return _NON_ALNUM_NOSPACE.sub("", _safe(s).lower())


def add_norm_columns(df):
    """Return a copy with name_norm/addr_norm/name_compact/addr_compact added."""
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_name).astype("string")
    df["addr_norm"] = df["business_address"].map(normalize_address).astype("string")
    df["name_compact"] = df["name_norm"].str.replace(" ", "", regex=False).astype("string")
    df["addr_compact"] = df["addr_norm"].str.replace(" ", "", regex=False).astype("string")
    return df
