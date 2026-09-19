import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd

from src.processing.player_identity import (
    find_best_match,
    extract_cricbuzz_id,
    save_identity
)

FEATURES_PATH = "data/features/player_match_features_full.csv"


def load_features():
    df = pd.read_csv(FEATURES_PATH)

    return df


def build_mapping(players):

    features_df = load_features()

    historical_names = (
        features_df["player"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rows = []

    for player in players:

        name = player["name"]

        historical_name, method, confidence = find_best_match(
            name,
            historical_names
        )

        rows.append({
            "cricbuzz_name": name,
            "historical_name": historical_name,
            "cricbuzz_id": extract_cricbuzz_id(
                player.get("profile_url")
            ),
            "match_method": method,
            "confidence": confidence
        })

    return rows


def run():

    print("Enter Cricbuzz player names.")
    print("Enter blank line when finished.\n")

    players = []

    while True:

        name = input("Player: ").strip()

        if not name:
            break

        profile_url = input("Cricbuzz profile URL: ").strip()

        players.append({
            "name": name,
            "profile_url": profile_url
        })

    rows = build_mapping(players)

    df = save_identity(rows)

    print("\nPLAYER MAPPING")
    print("=" * 90)

    print(
        df[
            [
                "cricbuzz_name",
                "historical_name",
                "match_method",
                "confidence"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", "data/processed/player_identity.csv")


if __name__ == "__main__":
    run()