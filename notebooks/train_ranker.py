import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRanker

FEATURES_PATH = "data/features/player_match_features_full.csv"
MODELS_PATH = "models"

DROP_COLS = [
    "fantasy_points", "player", "team", "opponent", "venue", "league", "format",
    "innings_type", "toss_winner", "toss_decision", "team1", "team2",
    "date", "season_year", "match_pressure_metric",
    "runs", "balls_faced", "fours", "sixes", "duck", "half", "century",
    "balls", "conceded", "wickets", "maidens", "caught", "run out", "stumped",
    "batting_position", "powerplay_balls", "middle_balls", "death_balls",
]

# set True to simulate scorecard-only data (no ball-by-ball) by dropping every
# feature that can only be derived from over/ball-level sequencing
SCORECARD_ONLY = False

BALL_BY_BALL_ONLY_COLS = [
    "avg_batting_position_last_5", "bat_pos_std", "predicted_bat_pos_group",
    "pct_matches_in_pos_top", "pct_matches_in_pos_middle", "pct_matches_in_pos_lower",
    "avg_points_at_bat_pos_group", "avg_points_last_5_same_bat_pos",
    "career_powerplay_ratio", "career_death_ratio",
]

if SCORECARD_ONLY:
    DROP_COLS = DROP_COLS + BALL_BY_BALL_ONLY_COLS

TEST_FRACTION = 0.15


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    df = df.dropna(subset=["fantasy_points", "match_id"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def split_train_test(df):
    cutoff = df["date"].quantile(1 - TEST_FRACTION)
    train_df = df[df["date"] <= cutoff].copy()
    test_df = df[df["date"] > cutoff].copy()
    return train_df, test_df


def prepare_xy(df):
    df = df.sort_values("match_id")
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + ["match_id"])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["fantasy_points"].clip(lower=0, upper=31)
    group = df.groupby("match_id").size().values
    return X, y, group, df


def evaluate_top11_overlap(test_df, preds):
    test_df = test_df.copy()
    test_df["pred_score"] = preds
    overlaps = []
    for _, group in test_df.groupby("match_id"):
        actual = group.sort_values("fantasy_points", ascending=False).head(11)["player"].values
        predicted = group.sort_values("pred_score", ascending=False).head(11)["player"].values
        overlaps.append(len(set(actual).intersection(set(predicted))))
    return np.mean(overlaps) if overlaps else float("nan")


def train_for_format(df, fmt):
    fmt_df = df[df["format"] == fmt]
    if fmt_df["match_id"].nunique() < 20:
        print(f"⚠️ skipping {fmt}, too few matches ({fmt_df['match_id'].nunique()})")
        return

    train_df, test_df = split_train_test(fmt_df)
    train_df, val_df = split_train_test(train_df)
    X_train, y_train, train_group, train_df = prepare_xy(train_df)
    X_val, y_val, val_group, val_df = prepare_xy(val_df)
    X_test, y_test, test_group, test_df = prepare_xy(test_df)

    ranker = XGBRanker(
        objective="rank:ndcg",
        ndcg_exp_gain=False,
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="ndcg@11",
    )
    ranker.fit(
        X_train, y_train, group=train_group,
        eval_set=[(X_val, y_val)], eval_group=[val_group],
        verbose=False,
    )

    preds = ranker.predict(X_test)
    avg_overlap = evaluate_top11_overlap(test_df, preds)

    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = os.path.join(MODELS_PATH, f"model_ranker_{fmt.lower()}{'_scorecard_only' if SCORECARD_ONLY else ''}.pkl")
    joblib.dump(ranker, model_path)

    print(f"✅ {fmt}: {fmt_df['match_id'].nunique()} matches, "
          f"avg top-11 overlap {avg_overlap:.2f}/11 ({avg_overlap / 11 * 100:.1f}%) -> {model_path}")


def run():
    print(f"mode: {'SCORECARD-ONLY (no ball-by-ball features)' if SCORECARD_ONLY else 'full feature set'}\n")
    df = load_features()
    for fmt in df["format"].dropna().unique():
        train_for_format(df, fmt)


if __name__ == "__main__":
    run()