import os
import re
import unicodedata
import pandas as pd
from difflib import SequenceMatcher

IDENTITY_PATH = "data/processed/venue_identity.csv"

ALIASES = {
    "feroz shah kotla": "arun jaitley stadium",
    "ferozeshah kotla": "arun jaitley stadium",
    "feroz shah kotla ground": "arun jaitley stadium",
    "ferozeshah kotla ground": "arun jaitley stadium",
    "kotla": "arun jaitley stadium",
    "arun jaitley cricket stadium": "arun jaitley stadium",
    "m chinnaswamy stadium": "m chinnaswamy stadium",
    "m a chidambaram stadium": "m a chidambaram stadium",
    "ma chidambaram stadium": "m a chidambaram stadium",
    "chepauk": "m a chidambaram stadium",
    "wankhede": "wankhede stadium",
    "wankhede cricket stadium": "wankhede stadium",
    "eden gardens cricket ground": "eden gardens",
    "eden garden": "eden gardens",
    "narendra modi cricket stadium": "narendra modi stadium",
    "sardar patel stadium": "narendra modi stadium",
    "hpca stadium": "h p c a stadium",
    "himachal pradesh cricket association stadium": "h p c a stadium",
}


def normalize_venue(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char
        for char in value
        if not unicodedata.combining(char)
    )

    value = value.replace("&", " and ")
    value = re.sub(r"[’'`\".,()\[\]{}:/\\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    replacements = {
        "cricket ground": "",
        "cricket stadium": "stadium",
        "international cricket stadium": "stadium",
        "international stadium": "stadium",
        "sports complex": "",
        "cricket club": "",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def _alias_target(value):
    normalized = normalize_venue(value)
    return ALIASES.get(normalized, normalized)


def _token_score(a, b):
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), len(b_tokens))


def _venue_score(query, candidate):
    sequence = SequenceMatcher(
        None,
        query,
        candidate
    ).ratio()

    token = _token_score(
        query,
        candidate
    )

    return (0.65 * sequence) + (0.35 * token)


def load_identity():
    if not os.path.exists(IDENTITY_PATH):
        return pd.DataFrame(
            columns=[
                "cricbuzz_venue",
                "historical_venue",
                "match_method",
                "confidence"
            ]
        )

    df = pd.read_csv(IDENTITY_PATH)

    for col in [
        "cricbuzz_venue",
        "historical_venue",
        "match_method",
        "confidence"
    ]:
        if col not in df.columns:
            df[col] = None

    return df


def update_identity(
    cricbuzz_venue,
    historical_venue,
    match_method,
    confidence
):
    if not cricbuzz_venue or not historical_venue:
        return

    os.makedirs(
        os.path.dirname(IDENTITY_PATH),
        exist_ok=True
    )

    df = load_identity()

    row = {
        "cricbuzz_venue": str(cricbuzz_venue).strip(),
        "historical_venue": str(historical_venue).strip(),
        "match_method": match_method,
        "confidence": float(confidence)
    }

    if not df.empty:
        normalized_existing = df[
            "cricbuzz_venue"
        ].astype(str).map(normalize_venue)

        mask = (
            normalized_existing
            == normalize_venue(cricbuzz_venue)
        )

        if mask.any():
            index = df.index[mask][0]
            for key, value in row.items():
                df.loc[index, key] = value
        else:
            df = pd.concat(
                [df, pd.DataFrame([row])],
                ignore_index=True
            )
    else:
        df = pd.DataFrame([row])

    df.to_csv(
        IDENTITY_PATH,
        index=False
    )


def resolve_venue(
    venue,
    features_df,
    min_confidence=0.82,
    min_margin=0.04
):
    original = "" if venue is None else str(venue).strip()
    normalized = normalize_venue(original)

    result = {
        "cricbuzz_venue": original,
        "historical_venue": None,
        "method": "unmatched",
        "confidence": 0.0
    }

    if not normalized:
        return result

    historical_values = (
        features_df["venue"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not historical_values:
        return result

    historical_map = {}

    for value in historical_values:
        key = normalize_venue(value)
        if key:
            historical_map.setdefault(key, value)

    alias_normalized = _alias_target(original)

    if alias_normalized in historical_map:
        historical = historical_map[alias_normalized]
        result.update(
            {
                "historical_venue": historical,
                "method": "alias",
                "confidence": 1.0
            }
        )
        return result

    if normalized in historical_map:
        historical = historical_map[normalized]
        result.update(
            {
                "historical_venue": historical,
                "method": "exact",
                "confidence": 1.0
            }
        )
        return result

    scored = []

    for key, original_historical in historical_map.items():
        score = _venue_score(
            alias_normalized,
            key
        )
        scored.append(
            (score, original_historical, key)
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if not scored:
        return result

    best_score, best_venue, _ = scored[0]
    second_score = (
        scored[1][0]
        if len(scored) > 1
        else 0.0
    )

    if (
        best_score >= min_confidence
        and (best_score - second_score) >= min_margin
    ):
        result.update(
            {
                "historical_venue": best_venue,
                "method": "fuzzy",
                "confidence": round(best_score, 6)
            }
        )

    return result
