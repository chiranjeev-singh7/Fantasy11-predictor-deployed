import os
import re
import pandas as pd
from difflib import SequenceMatcher

IDENTITY_PATH = "data/processed/player_identity.csv"

ALIASES = {
    "vaibhav sooryavanshi": "v suryavanshi",
    "jordan cox": "jm cox",
    "liam dawson": "la dawson",
    "gus atkinson": "aap atkinson",
    "charith asalanka": "kic asalanka",
    "dushmantha chameera": "pvd chameera",
    "wanindu hasaranga":"pwh de silva",
    "kamindu mendis":"phkd mendis"
}


def normalize_name(name):
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()

    name = re.sub(r"\([^)]*\)", "", name)

    replacements = {
        ".": "",
        "'": "",
        "-": " ",
        "_": " ",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    name = re.sub(r"\s+", " ", name).strip()

    return name


def name_tokens(name):
    return normalize_name(name).split()


def last_name(name):
    tokens = name_tokens(name)
    return tokens[-1] if tokens else ""


def initials(name):
    tokens = name_tokens(name)

    if not tokens:
        return ""

    if len(tokens) == 1:
        return tokens[0][0]

    return "".join(token[0] for token in tokens)


def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_name(a),
        normalize_name(b)
    ).ratio()


def initial_match(cricbuzz_name, historical_name):
    cb_tokens = name_tokens(cricbuzz_name)
    hist_tokens = name_tokens(historical_name)

    if not cb_tokens or not hist_tokens:
        return False

    if last_name(cricbuzz_name) != last_name(historical_name):
        return False

    cb_first = cb_tokens[0]
    hist_first = hist_tokens[0]

    if len(hist_first) == 1:
        return cb_first.startswith(hist_first)

    if len(cb_first) == 1:
        return hist_first.startswith(cb_first)

    return False


def find_best_match(player_name, historical_names):

    original_normalized = normalize_name(player_name)

    normalized = ALIASES.get(
        original_normalized,
        original_normalized
    )

    exact = [
        name
        for name in historical_names
        if normalize_name(name) == normalized
    ]

    if exact:
        method = (
            "alias"
            if normalized != original_normalized
            else "exact"
        )

        return exact[0], method, 1.0

    initial_candidates = [
        name
        for name in historical_names
        if initial_match(player_name, name)
    ]

    if len(initial_candidates) == 1:
        return initial_candidates[0], "initial_match", 0.98

    candidates = []

    for name in historical_names:
        score = similarity(player_name, name)

        if last_name(player_name) == last_name(name):
            score += 0.15

        if initials(player_name) == initials(name):
            score += 0.10

        candidates.append(
            (name, min(score, 1.0))
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if not candidates:
        return None, "unmatched", 0.0

    best_name, best_score = candidates[0]

    second_score = (
        candidates[1][1]
        if len(candidates) > 1
        else 0
    )

    if (
        best_score >= 0.88
        and best_score - second_score >= 0.05
    ):
        return (
            best_name,
            "fuzzy_match",
            best_score
        )

    return None, "unmatched", best_score


def extract_cricbuzz_id(profile_url):
    if not isinstance(profile_url, str):
        return None

    match = re.search(
        r"/profiles/(\d+)",
        profile_url
    )

    if match:
        return match.group(1)

    return None


def load_identity():
    if not os.path.exists(IDENTITY_PATH):
        return pd.DataFrame(
            columns=[
                "cricbuzz_name",
                "historical_name",
                "cricbuzz_id",
                "match_method",
                "confidence",
            ]
        )

    return pd.read_csv(
        IDENTITY_PATH
    )


def save_identity(rows):
    os.makedirs(
        os.path.dirname(IDENTITY_PATH),
        exist_ok=True
    )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["cricbuzz_name"],
            keep="last"
        )

    df.to_csv(
        IDENTITY_PATH,
        index=False
    )

    return df


def resolve_player(
    player_name,
    features_df,
    profile_url=None
):

    identity_df = load_identity()

    if not identity_df.empty:

        cricbuzz_id = extract_cricbuzz_id(
            profile_url
        )

        if cricbuzz_id is not None:

            match = identity_df[
                identity_df["cricbuzz_id"]
                .astype(str)
                == str(cricbuzz_id)
            ]

            if not match.empty:
                return {
                    "historical_name": match.iloc[0][
                        "historical_name"
                    ],
                    "method": "cricbuzz_id",
                    "confidence": 1.0,
                }

        match = identity_df[
            identity_df["cricbuzz_name"]
            .map(normalize_name)
            == normalize_name(player_name)
        ]

        if not match.empty:
            return {
                "historical_name": match.iloc[0][
                    "historical_name"
                ],
                "method": match.iloc[0][
                    "match_method"
                ],
                "confidence": float(
                    match.iloc[0]["confidence"]
                ),
            }

    historical_names = (
        features_df["player"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    historical_name, method, confidence = (
        find_best_match(
            player_name,
            historical_names
        )
    )

    return {
        "historical_name": historical_name,
        "method": method,
        "confidence": confidence,
    }


def update_identity(
    player_name,
    historical_name,
    profile_url=None,
    method="manual",
    confidence=1.0
):

    df = load_identity()

    new_row = {
        "cricbuzz_name": player_name,
        "historical_name": historical_name,
        "cricbuzz_id": extract_cricbuzz_id(
            profile_url
        ),
        "match_method": method,
        "confidence": confidence
    }

    if df.empty:

        df = pd.DataFrame(
            [new_row],
            columns=[
                "cricbuzz_name",
                "historical_name",
                "cricbuzz_id",
                "match_method",
                "confidence"
            ]
        )

    else:

        existing = (
            df["cricbuzz_name"]
            .map(normalize_name)
            == normalize_name(player_name)
        )

        if existing.any():

            index = df.index[existing][0]

            for column, value in new_row.items():
                df.loc[index, column] = value

        else:

            df.loc[len(df)] = new_row

    df = df.drop_duplicates(
        subset=["cricbuzz_name"],
        keep="last"
    )

    os.makedirs(
        os.path.dirname(IDENTITY_PATH),
        exist_ok=True
    )

    df.to_csv(
        IDENTITY_PATH,
        index=False
    )

    return df