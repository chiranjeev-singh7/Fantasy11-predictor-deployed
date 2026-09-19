import os
import json
import yaml
import pandas as pd

CONFIG_PATH = "config/leagues.yaml"
RAW_PATH = "data/raw/cricsheet"
PROCESSED_PATH = "data/processed"


def load_leagues():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["leagues"]


def parse_match_info(match_id, league_key, meta, info):
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    result = None
    result_margin = None

    if "by" in outcome:
        by = outcome["by"]
        if "runs" in by:
            result, result_margin = "runs", by["runs"]
        elif "wickets" in by:
            result, result_margin = "wickets", by["wickets"]
    elif "result" in outcome:
        result = outcome["result"]

    teams = info.get("teams", [None, None])
    toss = info.get("toss", {})

    return {
        "id": match_id,
        "league": league_key,
        "format": meta["format"],
        "gender": info.get("gender"),
        "season": info.get("season"),
        "city": info.get("city"),
        "date": info.get("dates", [None])[0],
        "match_type": info.get("match_type"),
        "player_of_match": ", ".join(info.get("player_of_match", []) or []),
        "venue": info.get("venue"),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result": result,
        "result_margin": result_margin,
        "target_runs": info.get("target", {}).get("runs"),
        "target_overs": info.get("target", {}).get("overs"),
    }


def parse_deliveries(match_id, innings, teams):
    rows = []

    for inning_idx, inning in enumerate(innings, start=1):
        batting_team = inning.get("team")
        bowling_team = next((t for t in teams if t != batting_team), None)

        for over_block in inning.get("overs", []):
            over_num = over_block["over"]

            for ball_num, delivery in enumerate(over_block.get("deliveries", []), start=1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                first_wicket = wickets[0] if wickets else {}
                fielders = first_wicket.get("fielders", [])
                fielder_name = fielders[0].get("name") if fielders else None

                rows.append({
                    "match_id": match_id,
                    "inning": inning_idx,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_num,
                    "ball": ball_num,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "non_striker": delivery.get("non_striker"),
                    "batsman_runs": runs.get("batter", 0),
                    "extra_runs": runs.get("extras", 0),
                    "total_runs": runs.get("total", 0),
                    "extras_type": ", ".join(sorted(extras.keys())) if extras else None,
                    "is_wicket": 1 if wickets else 0,
                    "player_dismissed": first_wicket.get("player_out"),
                    "dismissal_kind": first_wicket.get("kind"),
                    "fielder": fielder_name,
                })

    return rows


def parse_league(league_key, meta):
    league_dir = os.path.join(RAW_PATH, league_key)
    if not os.path.isdir(league_dir):
        print(f"⚠️ skipping {league_key}, no raw data found at {league_dir}")
        return [], []

    match_rows = []
    delivery_rows = []

    for fname in os.listdir(league_dir):
        if not fname.endswith(".json"):
            continue

        match_id = int(fname.replace(".json", ""))
        with open(os.path.join(league_dir, fname)) as f:
            data = json.load(f)

        info = data.get("info", {})
        match_rows.append(parse_match_info(match_id, league_key, meta, info))
        delivery_rows.extend(parse_deliveries(match_id, data.get("innings", []), info.get("teams", [])))

    return match_rows, delivery_rows


def run():
    leagues = load_leagues()
    all_matches = []
    all_deliveries = []

    for key, meta in leagues.items():
        if not meta.get("active", False):
            continue
        matches, deliveries = parse_league(key, meta)
        all_matches.extend(matches)
        all_deliveries.extend(deliveries)
        print(f"✅ {key}: {len(matches)} matches, {len(deliveries)} deliveries parsed")

    os.makedirs(PROCESSED_PATH, exist_ok=True)
    pd.DataFrame(all_matches).to_csv(os.path.join(PROCESSED_PATH, "matches.csv"), index=False)
    pd.DataFrame(all_deliveries).to_csv(os.path.join(PROCESSED_PATH, "deliveries.csv"), index=False)
    print(f"\n✅ saved {len(all_matches)} matches and {len(all_deliveries)} deliveries to {PROCESSED_PATH}")


if __name__ == "__main__":
    run()