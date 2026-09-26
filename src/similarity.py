import re

from rapidfuzz.distance import JaroWinkler, Levenshtein

_DIGIT = re.compile(r"\d+")

def tokens(s):
    return s.split(" ") if s else []

def token_set(s):
    return set(tokens(s)) if s else set()

def digit_tokens(s):
    return set(_DIGIT.findall(s)) if s else set()

def jaccard(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0

def token_overlap_ratio(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    return inter / min(len(a_tokens), len(b_tokens))


def digit_overlap(a_digits, b_digits):
    if not a_digits or not b_digits:
        return 0.0
    inter = len(a_digits & b_digits)
    union = len(a_digits | b_digits)
    return inter / union if union else 0.0

def levenshtein_ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)

def jaro_winkler(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a, b)

def substring_containment(a, b):
    if not a or not b:
        return 0.0
    return 1.0 if (a in b or b in a) else 0.0

def length_ratio(a, b):
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    return min(la, lb) / max(la, lb)

def first_token_match(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    return 1.0 if a_tokens[0] == b_tokens[0] else 0.0

def pairwise_features(a_name, b_name, a_addr, b_addr, a_name_compact, b_name_compact, a_country, b_country):
    a_ntok_list, b_ntok_list = tokens(a_name), tokens(b_name)
    a_ntok, b_ntok = set(a_ntok_list), set(b_ntok_list)
    a_atok, b_atok = token_set(a_addr), token_set(b_addr)
    da, db = digit_tokens(a_addr), digit_tokens(b_addr)

    return (
        1.0 if a_name and a_name == b_name else 0.0,                 # name_exact
        1.0 if a_name_compact and a_name_compact == b_name_compact else 0.0,  # name_compact_exact
        jaccard(a_ntok, b_ntok),                                     # name_jaccard
        token_overlap_ratio(a_ntok, b_ntok),                         # name_overlap_ratio
        levenshtein_ratio(a_name, b_name),                           # name_levenshtein
        jaro_winkler(a_name, b_name),                                # name_jaro_winkler
        substring_containment(a_name, b_name),                       # name_containment
        first_token_match(a_ntok_list, b_ntok_list),                 # name_first_token
        length_ratio(a_name, b_name),                                # name_len_ratio
        1.0 if a_addr and a_addr == b_addr else 0.0,                 # addr_exact
        jaccard(a_atok, b_atok),                                     # addr_jaccard
        token_overlap_ratio(a_atok, b_atok),                         # addr_overlap_ratio
        levenshtein_ratio(a_addr, b_addr),                           # addr_levenshtein
        jaro_winkler(a_addr, b_addr),                                # addr_jaro_winkler
        substring_containment(a_addr, b_addr),                       # addr_containment
        length_ratio(a_addr, b_addr),                                # addr_len_ratio
        digit_overlap(da, db),                                       # addr_digit_overlap
        1.0 if (da and db) else 0.0,                                 # addr_has_digits_both
        1.0 if a_country and a_country == b_country else 0.0,        # country_match
    )