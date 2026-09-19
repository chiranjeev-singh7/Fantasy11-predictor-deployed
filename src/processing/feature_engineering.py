import pandas as pd
import numpy as np
import os

PROCESSED_PATH = "data/processed"
FEATURES_PATH = "data/features"


def normalize_season(season):
    return season.astype(str).str.extract(r"(\d{4})")[0].astype("Int64")


def add_match_context(stats, matches_df):
    matches_df = matches_df.copy()
    if "id" in matches_df.columns and "match_id" not in matches_df.columns:
        matches_df = matches_df.rename(columns={"id": "match_id"})

    matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
    matches_df["season_year"] = normalize_season(matches_df["season"])
    matches_df["match_pressure_metric"] = 1

    stats = stats.merge(
        matches_df[["match_id", "date", "season_year", "team1", "team2",
                     "toss_winner", "toss_decision", "match_pressure_metric"]],
        on="match_id", how="left"
    )

    stats["innings_type"] = "unknown"
    bat_first_mask = (
        ((stats["toss_decision"] == "bat") & (stats["team"] == stats["toss_winner"])) |
        ((stats["toss_decision"] == "field") & (stats["team"] != stats["toss_winner"]))
    )
    stats.loc[bat_first_mask, "innings_type"] = "bat_first"
    stats.loc[~bat_first_mask, "innings_type"] = "bat_second"

    stats["opponent"] = stats.apply(
        lambda row: row["team2"] if row["team"] == row["team1"] else row["team1"], axis=1
    )

    return stats


def _global_running_mean(stats, value_col):
    by_date = stats.sort_values("date")
    running = by_date[value_col].shift(1).expanding().mean()
    result = pd.Series(index=stats.index, dtype=float)
    result.loc[by_date.index] = running.values
    overall_mean = stats[value_col].mean()
    return result.fillna(overall_mean)


def _shrunk_stat(stats, group_cols, value_col, window, min_samples, global_fallback):
    g = stats.groupby(group_cols)[value_col]
    if window == "expanding":
        group_mean = g.transform(lambda x: x.shift(1).expanding().mean())
    else:
        group_mean = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    group_count = stats.groupby(group_cols).cumcount()
    weight = np.minimum(group_count / (group_count + min_samples), 1)

    result = weight * group_mean.fillna(0) + (1 - weight) * global_fallback
    return result


def generate_all_features(player_stats, matches_df):
    stats = add_match_context(player_stats, matches_df)
    stats = stats.sort_values(["player", "date"]).reset_index(drop=True)

    global_fallback = _global_running_mean(stats, "fantasy_points")

    g_player = stats.groupby("player")["fantasy_points"]
    stats["avg_points_last_5"] = g_player.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    stats["total_points_last_5"] = g_player.transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0)
    stats["num_past_matches"] = stats.groupby("player").cumcount()
    stats["is_debut"] = (stats["num_past_matches"] == 0).astype(int)
    stats["form_ewm_5"] = g_player.transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean()
    ).fillna(0)
    stats["form_std_5"] = g_player.transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    ).fillna(stats["fantasy_points"].std())

    stats["avg_points_venue"] = _shrunk_stat(
        stats, ["player", "venue"], "fantasy_points", "expanding", 5, global_fallback
    )
    stats["num_matches_venue"] = stats.groupby(["player", "venue"]).cumcount()

    match_level = stats.groupby(["match_id", "venue", "date"])["fantasy_points"].mean().reset_index()
    match_level = match_level.rename(columns={"fantasy_points": "match_avg_points"}).sort_values(["venue", "date"])
    match_level["venue_scoring_baseline"] = match_level.groupby("venue")["match_avg_points"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    match_level["venue_scoring_baseline"] = match_level["venue_scoring_baseline"].fillna(
        match_level["match_avg_points"].mean()
    )
    stats = stats.merge(match_level[["match_id", "venue_scoring_baseline"]], on="match_id", how="left")

    stats["avg_points_opponent"] = _shrunk_stat(
        stats, ["player", "opponent"], "fantasy_points", "expanding", 5, global_fallback
    )
    stats["num_matches_opponent"] = stats.groupby(["player", "opponent"]).cumcount()

    stats["avg_points_vs_opponent_last_5"] = _shrunk_stat(
        stats, ["player", "opponent"], "fantasy_points", "last5", 3, global_fallback
    )

    stats["avg_points_vs_opponent_at_venue_last_5"] = _shrunk_stat(
        stats, ["player", "opponent", "venue"], "fantasy_points", "last5", 2, global_fallback
    )

    g_innings_type = stats.groupby(["player", "innings_type"])["fantasy_points"]
    innings_avg = g_innings_type.transform(lambda x: x.shift(1).expanding().mean())
    stats["avg_points_bat_first"] = innings_avg.where(stats["innings_type"] == "bat_first")
    stats["avg_points_bat_second"] = innings_avg.where(stats["innings_type"] == "bat_second")

    # ======================
    # BATTING POSITION - historical only, never this match's actual position
    # ======================
    stats["avg_batting_position_last_5"] = stats.groupby("player")["batting_position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    stats["avg_batting_position_last_5"] = stats["avg_batting_position_last_5"].fillna(
        stats["batting_position"].median()
    )
    stats["bat_pos_std"] = stats.groupby("player")["batting_position"].transform(
        lambda x: x.shift(1).expanding().std()
    ).fillna(4)

    stats["predicted_bat_pos_group"] = np.select(
        [stats["avg_batting_position_last_5"] <= 3, stats["avg_batting_position_last_5"] <= 7],
        [0, 1], default=2
    )

    for group_val, label in [(0, "top"), (1, "middle"), (2, "lower")]:
        indicator_col = f"_is_pos_{label}"
        stats[indicator_col] = (stats["predicted_bat_pos_group"] == group_val).astype(float)
        stats[f"pct_matches_in_pos_{label}"] = stats.groupby("player")[indicator_col].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        stats.drop(columns=[indicator_col], inplace=True)

    stats["avg_points_at_bat_pos_group"] = _shrunk_stat(
        stats, ["player", "predicted_bat_pos_group"], "fantasy_points", "expanding", 4, global_fallback
    )
    stats["avg_points_last_5_same_bat_pos"] = stats.groupby(["player", "predicted_bat_pos_group"])["fantasy_points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0)

    # ======================
    # BOWLING PHASE - historical ratios only, never this match's actual phase split.
    # pp_econ_rate / pp_wickets_per_over dropped: fantasy_points.py only tracks
    # phase *ball counts*, not phase-level runs conceded/wickets, so a leak-free
    # version of those two isn't derivable from current data.
    # ======================
    career_bowling_balls = stats.groupby("player")["balls"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    career_powerplay_balls = stats.groupby("player")["powerplay_balls"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    career_death_balls = stats.groupby("player")["death_balls"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    stats["career_powerplay_ratio"] = career_powerplay_balls / (career_bowling_balls + 1)
    stats["career_death_ratio"] = career_death_balls / (career_bowling_balls + 1)

    # ======================
    # PLAYER ROLE
    # ======================
    career_batting_balls = stats.groupby("player")["balls_faced"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)
    career_stumpings = stats.groupby("player")["stumped"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    stats["batting_involvement_ratio"] = career_batting_balls / (career_batting_balls + career_bowling_balls + 1)
    stats["is_keeper"] = (career_stumpings > 0).astype(int)

    # ======================
    # TEAM STRENGTH - uses only form_ewm_5, itself already shift(1)-based, so no leak
    # ======================
    team_form = stats.groupby(["match_id", "team"])["form_ewm_5"].transform("mean")
    team_count = stats.groupby(["match_id", "team"])["form_ewm_5"].transform("count")
    stats["own_team_strength"] = ((team_form * team_count - stats["form_ewm_5"]) / (team_count - 1).replace(0, 1))

    opp_team_form = (
        stats.groupby(["match_id", "team"])["form_ewm_5"].mean()
        .reset_index().rename(columns={"team": "opp_lookup", "form_ewm_5": "opponent_team_strength"})
    )
    stats = stats.merge(
        opp_team_form, left_on=["match_id", "opponent"], right_on=["match_id", "opp_lookup"], how="left"
    ).drop(columns=["opp_lookup"])

    stats = stats.replace([np.inf, -np.inf], np.nan)
    numeric_cols = stats.select_dtypes(include=[np.number]).columns
    stats[numeric_cols] = stats[numeric_cols].fillna(0)

    return stats


if __name__ == "__main__":
    os.makedirs(FEATURES_PATH, exist_ok=True)

    player_stats = pd.read_csv(os.path.join(PROCESSED_PATH, "player_match_stats.csv"))
    matches_df = pd.read_csv(os.path.join(PROCESSED_PATH, "matches.csv"))

    if "team" not in player_stats.columns:
        print("⚠️ 'team' column missing, run the updated fantasy_points.py first")
        exit()

    final_df = generate_all_features(player_stats, matches_df)
    final_df.to_csv(os.path.join(FEATURES_PATH, "player_match_features_full.csv"), index=False)
    final_df.to_csv(os.path.join(PROCESSED_PATH, "player_features.csv"), index=False)
    print(f"✅ saved {len(final_df)} rows across {final_df['league'].nunique()} leagues to {FEATURES_PATH}")
    print(f"📊 Features generated: {len(final_df.columns)} total columns")