import pandas as pd
import os

PROCESSED_PATH = "data/processed"

BATTING_SR_THRESHOLDS = {
    "T20": [(50, -6), (60, -4), (70, -2)],
    "ODI": [(30, -6), (40, -4), (50, -2)],
    "Test": [],
}

BOWLING_ECON_THRESHOLDS = {
    "T20": {"good": [(4, 6), (5, 4), (6, 2)], "bad": [(11, -6), (10, -4), (9, -2)]},
    "ODI": {"good": [(2.5, 6), (3.5, 4), (4.5, 2)], "bad": [(7, -6), (6, -4), (5, -2)]},
    "Test": {"good": [], "bad": []},
}


def load_processed():
    matches = pd.read_csv(os.path.join(PROCESSED_PATH, "matches.csv"))
    deliveries = pd.read_csv(os.path.join(PROCESSED_PATH, "deliveries.csv"))
    return matches, deliveries


def compute_batting_position(df):
    """Calculate batting position (1-11) for each batter in each match"""
    df_sorted = df.sort_values(["match_id", "inning", "over", "ball"])
    first_app = df_sorted.drop_duplicates(["match_id", "batter"], keep="first").copy()
    first_app["batting_position"] = first_app.groupby(["match_id", "batting_team"]).cumcount() + 1
    return first_app[["match_id", "batter", "batting_position"]].rename(columns={"batter": "player"})


def compute_bowling_phases(legal_deliveries):
    """Calculate balls bowled in powerplay/middle/death phases for each bowler"""
    ld = legal_deliveries.copy()
    ld["phase"] = "other"

    t20_mask = ld["format"] == "T20"
    odi_mask = ld["format"] == "ODI"

    ld.loc[t20_mask & (ld["over"] < 6), "phase"] = "powerplay"
    ld.loc[t20_mask & (ld["over"] >= 15), "phase"] = "death"
    ld.loc[t20_mask & (ld["over"] >= 6) & (ld["over"] < 15), "phase"] = "middle"

    ld.loc[odi_mask & (ld["over"] < 10), "phase"] = "powerplay"
    ld.loc[odi_mask & (ld["over"] >= 40), "phase"] = "death"
    ld.loc[odi_mask & (ld["over"] >= 10) & (ld["over"] < 40), "phase"] = "middle"

    phases = ld.groupby(["match_id", "bowler", "phase"]).size().unstack(fill_value=0).reset_index()
    phases.columns.name = None
    phases.rename(columns={"bowler": "player"}, inplace=True)
    for col in ["powerplay", "middle", "death"]:
        if col not in phases.columns:
            phases[col] = 0
    phases.rename(columns={"powerplay": "powerplay_balls", "middle": "middle_balls", "death": "death_balls"}, inplace=True)
    return phases[["match_id", "player", "powerplay_balls", "middle_balls", "death_balls"]]


def compute_fantasy_points(deliveries, matches):
    df = deliveries.merge(
        matches[["id", "league", "format", "venue"]],
        left_on="match_id", right_on="id", how="left"
    )

    extras_type = df["extras_type"].fillna("")
    df["balls_faced"] = (~extras_type.str.contains("wides")).astype(int)

    # ======================
    # BATTING STATS + POSITION
    # ======================
    batting = df.groupby(["match_id", "batter", "batting_team", "league", "format", "venue"]).agg(
        runs=("batsman_runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
    ).reset_index()
    batting.rename(columns={"batter": "player", "batting_team": "team"}, inplace=True)

    # ADD BATTING POSITION FEATURE
    batting_pos = compute_batting_position(deliveries)
    batting = batting.merge(batting_pos, on=["match_id", "player"], how="left")

    batting["duck"] = ((batting["runs"] == 0) & (batting["balls_faced"] > 0)).astype(int)
    batting["half"] = ((batting["runs"] >= 50) & (batting["runs"] < 100)).astype(int)
    batting["century"] = (batting["runs"] >= 100).astype(int)

    # ======================
    # BOWLING STATS + PHASES
    # ======================
    legal_deliveries = df[~df["extras_type"].fillna("").str.contains("wides")]
    bowl = legal_deliveries.groupby(["match_id", "bowler", "bowling_team"]).agg(
        balls=("ball", "count"),
        conceded=("total_runs", "sum"),
        wickets=("player_dismissed", lambda x: x.notna().sum()),
    ).reset_index()
    bowl.rename(columns={"bowler": "player", "bowling_team": "team"}, inplace=True)
    bowl["maidens"] = 0

    # ADD BOWLING PHASE FEATURES
    bowling_phases = compute_bowling_phases(legal_deliveries)
    bowl = bowl.merge(bowling_phases, on=["match_id", "player"], how="left")

    # ======================
    # FIELDING STATS
    # ======================
    dismissals = df[df["dismissal_kind"].isin(["caught", "run out", "stumped"])].copy()
    field = dismissals.groupby(["match_id", "fielder", "bowling_team", "dismissal_kind"]).size().unstack(fill_value=0).reset_index()
    field.columns.name = None
    field.rename(columns={"fielder": "player", "bowling_team": "team"}, inplace=True)
    for col in ["caught", "run out", "stumped"]:
        if col not in field.columns:
            field[col] = 0

    # ======================
    # MERGE ALL COMPONENTS
    # ======================
    ps = batting.merge(bowl, on=["match_id", "player"], how="outer", suffixes=("", "_bowl"))
    ps = ps.merge(field, on=["match_id", "player"], how="outer", suffixes=("", "_field"))

    ps["team"] = ps["team"].fillna(ps.get("team_bowl")).fillna(ps.get("team_field"))
    ps.drop(columns=[c for c in ["team_bowl", "team_field"] if c in ps.columns], inplace=True)

    numeric_cols = [
        "runs", "balls_faced", "fours", "sixes", "duck", "half", "century",
        "balls", "conceded", "wickets", "maidens", "caught", "run out", "stumped",
        "batting_position",  # NEW FEATURE
        "powerplay_balls",   # NEW FEATURE
        "middle_balls",      # NEW FEATURE
        "death_balls"        # NEW FEATURE
    ]
    for col in numeric_cols:
        if col not in ps.columns:
            ps[col] = 0
    ps[numeric_cols] = ps[numeric_cols].fillna(0)

    ps.drop(columns=["league", "format", "venue"], inplace=True)
    ps = ps.merge(matches[["id", "league", "format", "venue"]], left_on="match_id", right_on="id", how="left")
    ps.drop(columns=["id"], inplace=True)

    def pt(row):
        fmt = row["format"]
        p = 4
        p += row["runs"]
        p += row["fours"] * 1 + row["sixes"] * 2
        p += row["half"] * 8 + row["century"] * 16
        p -= row["duck"] * 2
        p += row["wickets"] * 25
        if row["wickets"] >= 5:
            p += 16
        elif row["wickets"] >= 4:
            p += 8
        p += row.get("caught", 0) * 8 + row.get("stumped", 0) * 12 + row.get("run out", 0) * 12

        econ = BOWLING_ECON_THRESHOLDS.get(fmt, BOWLING_ECON_THRESHOLDS["T20"])
        if row["balls"] >= 12:
            rpo = row["conceded"] / (row["balls"] / 6)
            for cap, bonus in econ["good"]:
                if rpo < cap:
                    p += bonus
                    break
            for cap, penalty in econ["bad"]:
                if rpo > cap:
                    p += penalty
                    break

        sr_thresholds = BATTING_SR_THRESHOLDS.get(fmt, BATTING_SR_THRESHOLDS["T20"])
        if row["balls_faced"] >= 10 and sr_thresholds:
            sr = row["runs"] / (row["balls_faced"] / 100)
            for cap, penalty in sr_thresholds:
                if sr < cap:
                    p += penalty
                    break

        return p

    ps["fantasy_points"] = ps.apply(pt, axis=1)
    return ps


if __name__ == "__main__":
    matches, deliveries = load_processed()
    player_stats = compute_fantasy_points(deliveries, matches)
    player_stats.to_csv(os.path.join(PROCESSED_PATH, "player_match_stats.csv"), index=False)
    print(f"✅ saved {len(player_stats)} player-match rows with fantasy points across {player_stats['league'].nunique()} leagues")