import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
import numpy as np
import joblib

from processing.player_identity import (
    resolve_player,
    extract_cricbuzz_id,
    update_identity
)
from processing.venue_identity import (
    resolve_venue,
    update_identity as update_venue_identity
)

FEATURES_PATH = "data/features/player_match_features_full.csv"
MODELS_PATH = "models"


def load_model(fmt):
    fmt = str(fmt).strip().lower()

    path = os.path.join(
        MODELS_PATH,
        f"model_ranker_{fmt}.pkl"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no trained model for format '{fmt}' at {path}"
        )

    return joblib.load(path)


def load_features():
    df = pd.read_csv(FEATURES_PATH)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


def _innings_type(
    team,
    toss_winner,
    toss_decision
):

    toss_winner = str(toss_winner).strip()
    toss_decision = str(toss_decision).strip().lower()

    bats_first = (
        (
            toss_decision == "bat"
            and team == toss_winner
        )
        or
        (
            toss_decision == "field"
            and team != toss_winner
        )
    )

    return (
        "bat_first"
        if bats_first
        else "bat_second"
    )


def _player_intrinsic_features(
    player,
    feature_cols,
    features_df,
    profile_url=None
):

    resolved = resolve_player(
        player,
        features_df,
        profile_url
    )

    historical_name = resolved["historical_name"]

    if historical_name is None:

        return (
            {
                col: 0
                for col in feature_cols
            },
            False,
            resolved
        )

    hist = features_df[
        features_df["player"]
        == historical_name
    ].sort_values("date")

    if hist.empty:

        return (
            {
                col: 0
                for col in feature_cols
            },
            False,
            resolved
        )

    latest = hist.iloc[-1]

    row = {}

    for col in feature_cols:

        if col in latest.index:
            row[col] = latest[col]
        else:
            row[col] = 0

    return (
        row,
        True,
        resolved
    )


def _recompute_venue_opponent_features(
    player,
    opponent,
    venue,
    innings_type,
    features_df,
    global_mean,
    venue_baseline
):

    if player is None:

        return {
            "avg_points_venue": global_mean,
            "num_matches_venue": 0,
            "avg_points_opponent": global_mean,
            "num_matches_opponent": 0,
            "avg_points_vs_opponent_last_5": global_mean,
            "avg_points_vs_opponent_at_venue_last_5": global_mean,
            "avg_points_bat_first": global_mean,
            "avg_points_bat_second": global_mean,
            "venue_scoring_baseline": venue_baseline
        }

    hist = features_df[
        features_df["player"] == player
    ].sort_values("date")

    if hist.empty:

        return {
            "avg_points_venue": global_mean,
            "num_matches_venue": 0,
            "avg_points_opponent": global_mean,
            "num_matches_opponent": 0,
            "avg_points_vs_opponent_last_5": global_mean,
            "avg_points_vs_opponent_at_venue_last_5": global_mean,
            "avg_points_bat_first": global_mean,
            "avg_points_bat_second": global_mean,
            "venue_scoring_baseline": venue_baseline
        }

    venue_hist = hist[
        hist["venue"].astype(str).str.lower()
        == str(venue).lower()
    ]["fantasy_points"]

    opp_hist = hist[
        hist["opponent"].astype(str).str.lower()
        == str(opponent).lower()
    ]["fantasy_points"]

    opp_venue_hist = hist[
        (
            hist["opponent"].astype(str).str.lower()
            == str(opponent).lower()
        )
        &
        (
            hist["venue"].astype(str).str.lower()
            == str(venue).lower()
        )
    ]["fantasy_points"]

    innings_hist = hist[
        hist["innings_type"].astype(str).str.lower()
        == innings_type.lower()
    ]["fantasy_points"]

    def shrink(series, min_samples=5):

        if len(series) == 0:
            return global_mean

        weight = (
            len(series)
            /
            (len(series) + min_samples)
        )

        return (
            weight * series.mean()
            +
            (1 - weight) * global_mean
        )

    return {
        "avg_points_venue": shrink(
            venue_hist
        ),

        "num_matches_venue": len(
            venue_hist
        ),

        "avg_points_opponent": shrink(
            opp_hist
        ),

        "num_matches_opponent": len(
            opp_hist
        ),

        "avg_points_vs_opponent_last_5": (
            opp_hist.tail(5).mean()
            if len(opp_hist)
            else global_mean
        ),

        "avg_points_vs_opponent_at_venue_last_5": (
            opp_venue_hist.tail(5).mean()
            if len(opp_venue_hist)
            else global_mean
        ),

        "avg_points_bat_first": (
            innings_hist.mean()
            if innings_type == "bat_first"
            and len(innings_hist)
            else global_mean
        ),

        "avg_points_bat_second": (
            innings_hist.mean()
            if innings_type == "bat_second"
            and len(innings_hist)
            else global_mean
        ),

        "venue_scoring_baseline": venue_baseline
    }


def predict_from_match_id(
    match_id,
    features_df=None
):

    features_df = (
        features_df
        if features_df is not None
        else load_features()
    )

    match_players = features_df[
        features_df["match_id"] == match_id
    ].copy()

    if match_players.empty:
        raise ValueError(
            f"no data found for match_id {match_id}"
        )

    fmt = match_players[
        "format"
    ].iloc[0]

    model = load_model(fmt)

    feature_cols = list(
        model.feature_names_in_
    )

    for col in feature_cols:

        if col not in match_players.columns:
            match_players[col] = 0

    X = match_players[
        feature_cols
    ].fillna(0)

    match_players[
        "pred_score"
    ] = model.predict(X)

    ranking = (
        match_players
        .sort_values(
            "pred_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = (
        ranking.index + 1
    )

    return ranking[
        [
            "rank",
            "player",
            "team",
            "pred_score"
        ]
    ]


def predict_live_match(
    fmt,
    team1,
    team2,
    team1_players,
    team2_players,
    venue,
    toss_winner,
    toss_decision,
    features_df=None
):

    features_df = (
        features_df
        if features_df is not None
        else load_features()
    )

    model = load_model(fmt)

    feature_cols = list(
        model.feature_names_in_
    )

    venue_resolved = resolve_venue(
        venue,
        features_df
    )

    historical_venue = venue_resolved["historical_venue"]
    venue_for_features = (
        historical_venue
        if historical_venue is not None
        else venue
    )

    fmt_hist = features_df[
        features_df["format"].astype(str).str.lower()
        == str(fmt).lower()
    ].copy()

    if fmt_hist.empty:
        raise ValueError(
            f"No historical data available for format '{fmt}'"
        )

    global_mean = (
        fmt_hist["fantasy_points"].mean()
    )

    venue_hist_all = fmt_hist[
        fmt_hist["venue"].astype(str).str.lower()
        == str(venue_for_features).lower()
    ]

    venue_baseline = (
        venue_hist_all["fantasy_points"].mean()
        if len(venue_hist_all)
        else global_mean
    )

    squad = []

    for player in team1_players:

        if isinstance(player, dict):

            name = player.get("name", "")
            profile_url = player.get(
                "profile_url"
            )
            role = player.get("role")

        else:

            name = str(player)
            profile_url = None
            role = None

        squad.append(
            {
                "name": name,
                "team": team1,
                "opponent": team2,
                "profile_url": profile_url,
                "role": role
            }
        )

    for player in team2_players:

        if isinstance(player, dict):

            name = player.get("name", "")
            profile_url = player.get(
                "profile_url"
            )
            role = player.get("role")

        else:

            name = str(player)
            profile_url = None
            role = None

        squad.append(
            {
                "name": name,
                "team": team2,
                "opponent": team1,
                "profile_url": profile_url,
                "role": role
            }
        )

    rows = []

    identity_rows = []

    for player_info in squad:

        player = player_info["name"]
        team = player_info["team"]
        opponent = player_info["opponent"]
        profile_url = player_info["profile_url"]
        role = player_info["role"]

        innings_type = _innings_type(
            team,
            toss_winner,
            toss_decision
        )

        row, has_history, resolved = (
            _player_intrinsic_features(
                player,
                feature_cols,
                features_df,
                profile_url
            )
        )

        historical_player = (
            resolved["historical_name"]
        )

        if historical_player is not None:

            overrides = (
                _recompute_venue_opponent_features(
                    historical_player,
                    opponent,
                    venue_for_features,
                    innings_type,
                    features_df,
                    global_mean,
                    venue_baseline
                )
            )

        else:

            overrides = {
                "avg_points_venue": global_mean,
                "num_matches_venue": 0,
                "avg_points_opponent": global_mean,
                "num_matches_opponent": 0,
                "avg_points_vs_opponent_last_5": global_mean,
                "avg_points_vs_opponent_at_venue_last_5": global_mean,
                "avg_points_bat_first": global_mean,
                "avg_points_bat_second": global_mean,
                "venue_scoring_baseline": venue_baseline
            }

        for col, value in overrides.items():

            if col in row:
                row[col] = value

        row["player"] = player
        row["team"] = team
        row["_has_history"] = has_history
        row["_historical_name"] = historical_player
        row["_match_method"] = resolved["method"]
        row["_match_confidence"] = resolved["confidence"]
        row["_role"] = role

        identity_rows.append(
            {
                "cricbuzz_name": player,
                "historical_name": historical_player,
                "cricbuzz_id": extract_cricbuzz_id(
                    profile_url
                ),
                "match_method": resolved["method"],
                "confidence": resolved["confidence"]
            }
        )

        rows.append(row)

    match_df = pd.DataFrame(rows)

    for col in feature_cols:

        if col not in match_df.columns:
            match_df[col] = 0

    X = match_df[
        feature_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    match_df[
        "pred_score"
    ] = model.predict(X)

    for identity in identity_rows:

        if (
            identity["historical_name"] is not None
            and identity["match_method"] != "exact"
        ):

            update_identity(
                identity["cricbuzz_name"],
                identity["historical_name"],
                identity["cricbuzz_id"],
                identity["match_method"],
                identity["confidence"]
            )

    if venue_resolved["historical_venue"] is not None:
        update_venue_identity(
            venue_resolved["cricbuzz_venue"],
            venue_resolved["historical_venue"],
            venue_resolved["method"],
            venue_resolved["confidence"]
        )

    print()
    print("=" * 110)
    print("VENUE IDENTITY MATCHING")
    print("=" * 110)
    print(
        f"Cricbuzz Venue: {venue_resolved['cricbuzz_venue']}"
    )
    print(
        f"Historical Venue: {venue_resolved['historical_venue'] or 'Not matched'}"
    )
    print(
        f"Method: {venue_resolved['method']}"
    )
    print(
        f"Confidence: {venue_resolved['confidence']:.6f}"
    )

    print()
    print("=" * 110)
    print("PLAYER IDENTITY MATCHING")
    print("=" * 110)

    identity_display = match_df[
        [
            "player",
            "team",
            "_historical_name",
            "_match_method",
            "_match_confidence"
        ]
    ].copy()

    identity_display.columns = [
        "Cricbuzz Player",
        "Team",
        "Historical Player",
        "Method",
        "Confidence"
    ]

    print(
        identity_display.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("FULL PREDICTION RANKING")
    print("=" * 110)

    ranking = (
        match_df
        .sort_values(
            "pred_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = (
        ranking.index + 1
    )

    ranking_display = ranking[
        [
            "rank",
            "player",
            "team",
            "pred_score"
        ]
    ].copy()

    ranking_display.columns = [
        "Rank",
        "Player",
        "Team",
        "Predicted Score"
    ]

    print(
        ranking_display.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print("TOP 11")
    print("=" * 110)

    top11 = ranking_display.head(11).copy()

    print(
        top11.to_string(
            index=False
        )
    )

    return ranking_display


def main():

    print()
    print("=" * 70)
    print("DREAM11 CRICKET PREDICTOR")
    print("=" * 70)

    print()
    print("1 = Historical match")
    print("2 = Upcoming match - manual squads")
    print("3 = Upcoming match - CricketData")

    mode = input(
        "\nSelect mode: "
    ).strip()

    if mode == "1":

        match_id = int(
            input("Match ID: ").strip()
        )

        result = predict_from_match_id(
            match_id
        )

        print()
        print("=" * 90)
        print("HISTORICAL MATCH PREDICTION")
        print("=" * 90)

        print(
            result.to_string(
                index=False
            )
        )

    elif mode == "2":

        fmt = input(
            "Format (T20/ODI/Test/Hundred): "
        ).strip()

        team1 = input(
            "Team 1: "
        ).strip()

        team2 = input(
            "Team 2: "
        ).strip()

        team1_players = [
            p.strip()
            for p in input(
                "Team 1 squad (comma separated): "
            ).split(",")
            if p.strip()
        ]

        team2_players = [
            p.strip()
            for p in input(
                "Team 2 squad (comma separated): "
            ).split(",")
            if p.strip()
        ]

        venue = input(
            "Venue: "
        ).strip()

        toss_winner = input(
            "Toss winner: "
        ).strip()

        toss_decision = input(
            "Toss decision (bat/field): "
        ).strip()

        result = predict_live_match(
            fmt,
            team1,
            team2,
            team1_players,
            team2_players,
            venue,
            toss_winner,
            toss_decision
        )

    elif mode == "3":

        from ingestion.cricbuzz_client import (
            fetch_fixture_for_prediction
        )

        team1 = input(
            "Team 1: "
        ).strip()

        team2 = input(
            "Team 2: "
        ).strip()

        fixture = fetch_fixture_for_prediction(
            team1,
            team2
        )

        print()
        print(
            f"{fixture['team1']} vs "
            f"{fixture['team2']}"
        )

        print(
            f"Venue: {fixture['venue']}"
        )

        print(
            f"Format: {fixture['format']}"
        )

        if fixture.get(
            "squads_confirmed_xi",
            False
        ):
            print(
                "Playing XI: confirmed"
            )
        else:
            print(
                "Playing XI: not confirmed"
            )

        toss_winner = input(
            "Toss winner "
            "(leave blank if not happened): "
        ).strip()

        toss_decision = input(
            "Toss decision (bat/field): "
        ).strip()

        if not toss_winner:

            toss_winner = fixture[
                "team1"
            ]

            toss_decision = "bat"

        if not toss_decision:

            toss_decision = "bat"

        if fixture.get("squads_confirmed_xi", False):
            prediction_team1_players = fixture["team1_playing_xi"]
            prediction_team2_players = fixture["team2_playing_xi"]
        else:
            prediction_team1_players = fixture["team1_players"]
            prediction_team2_players = fixture["team2_players"]

        print()
        print("=" * 70)
        print("PLAYERS USED FOR PREDICTION")
        print("=" * 70)
        print(f"{fixture['team1'].title()} ({len(prediction_team1_players)}):")
        for player in prediction_team1_players:
            print(f"- {player['name']}")
        print(f"\n{fixture['team2'].title()} ({len(prediction_team2_players)}):")
        for player in prediction_team2_players:
            print(f"- {player['name']}")

        result = predict_live_match(
            fixture["format"],
            fixture["team1"],
            fixture["team2"],
            prediction_team1_players,
            prediction_team2_players,
            fixture["venue"],
            toss_winner,
            toss_decision
        )

    else:

        print(
            "Invalid input. Enter 1, 2 or 3."
        )


if __name__ == "__main__":
    main()