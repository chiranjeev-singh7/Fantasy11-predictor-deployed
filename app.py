import os
import sys
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict.predictor import predict_from_match_id, predict_live_match
from ingestion.cricbuzz_client import fetch_fixture_for_prediction
from processing.player_identity import resolve_player
from processing.venue_identity import resolve_venue

FEATURES_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "features",
    "player_features_deploy.csv.gz"
)

st.set_page_config(
    page_title="Dream11 Cricket Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 24px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🏏 Dream11 Cricket Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">ML-powered player ranking using historical cricket performance and match context</div>',
    unsafe_allow_html=True
)


@st.cache_data(show_spinner="Loading historical player features...")
def load_deployment_features():
    df = pd.read_csv(
        FEATURES_PATH,
        compression="gzip"
    )
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )
    return df


@st.cache_data(show_spinner=False)
def get_historical_match_ids(features_df):
    return sorted(
        pd.to_numeric(
            features_df["match_id"],
            errors="coerce"
        ).dropna().astype(int).unique().tolist(),
        reverse=True
    )


def parse_players(text):
    values = []
    for item in str(text).replace("\n", ",").split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


def show_fixture(fixture):
    st.markdown('<div class="section-title">Match Details</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match", f"{fixture['team1'].title()} vs {fixture['team2'].title()}")
    c2.metric("Format", fixture["format"])
    c3.metric("Venue", fixture["venue"])
    c4.metric(
        "Players",
        str(fixture.get("total_players", 0))
    )

    if fixture.get("squads_confirmed_xi", False):
        st.success("Playing XI confirmed — prediction uses 11 players from each team.")
    else:
        st.info("Playing XI not announced — prediction uses the available squad.")

    left, right = st.columns(2)

    with left:
        st.markdown(f"### {fixture['team1'].title()}")
        players = fixture["team1_players"]
        data = []
        for i, player in enumerate(players, 1):
            data.append({
                "#": i,
                "Player": player["name"],
                "Role": player.get("role", "")
            })
        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.markdown(f"### {fixture['team2'].title()}")
        players = fixture["team2_players"]
        data = []
        for i, player in enumerate(players, 1):
            data.append({
                "#": i,
                "Player": player["name"],
                "Role": player.get("role", "")
            })
        st.dataframe(
            pd.DataFrame(data),
            use_container_width=True,
            hide_index=True
        )


def normalize_prediction_result(result):
    display = result.copy()

    rename_map = {
        "rank": "Rank",
        "player": "Player",
        "team": "Team",
        "pred_score": "Predicted Score"
    }

    display = display.rename(columns=rename_map)

    if "Predicted Score" not in display.columns:
        raise ValueError(
            "Prediction result does not contain a predicted score column."
        )

    display["Predicted Score"] = pd.to_numeric(
        display["Predicted Score"],
        errors="coerce"
    ).round(5)

    if "Rank" not in display.columns:
        display["Rank"] = range(1, len(display) + 1)

    return display


def build_identity_rows(players, features_df):
    rows = []

    for player in players:
        if isinstance(player, dict):
            name = player.get("name", "")
            team = player.get("team", "")
            profile_url = player.get("profile_url")
        else:
            name = str(player)
            team = ""
            profile_url = None

        resolved = resolve_player(
            name,
            features_df,
            profile_url
        )

        rows.append({
            "Player": name,
            "Team": team,
            "Historical Player": resolved.get("historical_name"),
            "Method": resolved.get("method"),
            "Confidence": resolved.get("confidence")
        })

    identity_df = pd.DataFrame(rows)

    if not identity_df.empty:
        identity_df["Confidence"] = pd.to_numeric(
            identity_df["Confidence"],
            errors="coerce"
        ).round(4)

    return identity_df


def show_identity_matching(players, features_df, venue=None):
    with st.expander("Player identity matching", expanded=True):
        identity_df = build_identity_rows(
            players,
            features_df
        )

        st.dataframe(
            identity_df,
            use_container_width=True,
            hide_index=True
        )

    if venue is not None:
        with st.expander("Venue identity matching", expanded=True):
            venue_result = resolve_venue(
                venue,
                features_df
            )

            venue_df = pd.DataFrame([{
                "Cricbuzz Venue": venue_result["cricbuzz_venue"],
                "Historical Venue": venue_result["historical_venue"] or "Not matched",
                "Method": venue_result["method"],
                "Confidence": round(venue_result["confidence"], 4)
            }])

            st.dataframe(
                venue_df,
                use_container_width=True,
                hide_index=True
            )


def show_prediction(result, fixture=None, players=None, venue=None):
    st.markdown('<div class="section-title">Prediction Ranking</div>', unsafe_allow_html=True)

    display = normalize_prediction_result(result)

    top11 = display.head(11).copy()
    top11.index = range(1, len(top11) + 1)
    top11.index.name = "Rank"

    st.markdown("### 🏆 Top 11 by Model Score")
    st.dataframe(
        top11,
        use_container_width=True,
        hide_index=False
    )

    st.markdown("### Full Player Ranking")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    if fixture is not None:
        fixture_players = (
            fixture.get("team1_players", [])
            + fixture.get("team2_players", [])
        )
        show_identity_matching(
            fixture_players,
            load_deployment_features(),
            fixture.get("venue")
        )
    elif players is not None:
        show_identity_matching(
            players,
            load_deployment_features(),
            venue
        )


def get_toss_inputs(team1, team2, key_prefix):
    toss_status = st.radio(
        "Toss status",
        ["Not happened / unknown", "Toss available"],
        horizontal=True,
        key=f"{key_prefix}_toss_status"
    )

    if toss_status == "Toss available":
        toss_winner = st.selectbox(
            "Toss winner",
            [team1, team2],
            key=f"{key_prefix}_toss_winner"
        )
        toss_decision = st.selectbox(
            "Toss decision",
            ["bat", "field"],
            key=f"{key_prefix}_toss_decision"
        )
    else:
        toss_winner = team1
        toss_decision = "bat"
        st.caption(
            f"Toss is unknown, so the model input defaults to {team1} batting first."
        )

    return toss_winner, toss_decision


features_df = load_deployment_features()

with st.sidebar:
    st.header("Prediction Mode")
    mode = st.radio(
        "Choose mode",
        [
            "Upcoming Match - Cricbuzz",
            "Upcoming Match - Manual",
            "Historical Match"
        ]
    )

    st.divider()
    st.caption(f"Historical rows loaded: {len(features_df):,}")
    st.caption("Models: T20, ODI, Test, Hundred")

if mode == "Upcoming Match - Cricbuzz":
    st.markdown("## 🔴 Upcoming Match")
    st.write("Fetch the next matching fixture and squads directly from Cricbuzz.")

    c1, c2 = st.columns(2)
    with c1:
        team1 = st.text_input("Team 1", value="India")
    with c2:
        team2 = st.text_input("Team 2", value="Afghanistan")

    if st.button("🔎 Find Match", type="primary", use_container_width=True):
        if not team1.strip() or not team2.strip():
            st.error("Enter both teams.")
        else:
            try:
                with st.spinner("Searching Cricbuzz and fetching squads..."):
                    fixture = fetch_fixture_for_prediction(
                        team1.strip(),
                        team2.strip()
                    )
                st.session_state["fixture"] = fixture
                st.session_state["prediction_result"] = None
                st.rerun()
            except Exception as e:
                st.error(str(e))

    fixture = st.session_state.get("fixture")

    if fixture is not None:
        show_fixture(fixture)

        st.markdown("### Toss")
        toss_winner, toss_decision = get_toss_inputs(
            fixture["team1"],
            fixture["team2"],
            "cricbuzz"
        )

        if st.button("🚀 Predict Players", type="primary", use_container_width=True):
            try:
                if fixture.get("squads_confirmed_xi", False):
                    team1_players = fixture["team1_playing_xi"]
                    team2_players = fixture["team2_playing_xi"]
                else:
                    team1_players = fixture["team1_players"]
                    team2_players = fixture["team2_players"]

                with st.spinner("Running the ML ranking model..."):
                    result = predict_live_match(
                        fixture["format"],
                        fixture["team1"],
                        fixture["team2"],
                        team1_players,
                        team2_players,
                        fixture["venue"],
                        toss_winner,
                        toss_decision,
                        features_df=features_df
                    )

                st.session_state["prediction_result"] = result
            except Exception as e:
                st.error(str(e))

        result = st.session_state.get("prediction_result")
        if result is not None:
            show_prediction(result, fixture)

elif mode == "Upcoming Match - Manual":
    st.markdown("## 📝 Manual Match Prediction")
    st.write("Use this mode when you already have the squads and venue.")

    c1, c2 = st.columns(2)
    with c1:
        fmt = st.selectbox("Format", ["T20", "ODI", "Test", "Hundred"])
        team1 = st.text_input("Team 1", value="India", key="manual_team1")
        team1_text = st.text_area(
            "Team 1 squad",
            placeholder="Player 1, Player 2, Player 3, ...",
            height=180
        )

    with c2:
        venue = st.text_input("Venue", value="Arun Jaitley Stadium, Delhi")
        team2 = st.text_input("Team 2", value="Afghanistan", key="manual_team2")
        team2_text = st.text_area(
            "Team 2 squad",
            placeholder="Player 1, Player 2, Player 3, ...",
            height=180
        )

    toss_winner, toss_decision = get_toss_inputs(
        team1,
        team2,
        "manual"
    )

    if st.button("🚀 Predict Players", type="primary", use_container_width=True):
        team1_players = parse_players(team1_text)
        team2_players = parse_players(team2_text)

        if not team1_players or not team2_players:
            st.error("Enter players for both teams.")
        elif not team1.strip() or not team2.strip() or not venue.strip():
            st.error("Enter both team names and the venue.")
        else:
            try:
                with st.spinner("Running the ML ranking model..."):
                    result = predict_live_match(
                        fmt,
                        team1.strip(),
                        team2.strip(),
                        team1_players,
                        team2_players,
                        venue.strip(),
                        toss_winner,
                        toss_decision,
                        features_df=features_df
                    )
                manual_players = (
                    [{"name": p, "team": team1.strip()} for p in team1_players]
                    + [{"name": p, "team": team2.strip()} for p in team2_players]
                )
                show_prediction(
                    result,
                    players=manual_players,
                    venue=venue.strip()
                )
            except Exception as e:
                st.error(str(e))

else:
    st.markdown("## 📊 Historical Match")
    st.write("Rank players from a historical match using the trained model for that format.")

    match_id = st.text_input("Match ID", placeholder="Example: 1304047")

    if st.button("🚀 Predict Historical Match", type="primary", use_container_width=True):
        if not match_id.strip():
            st.error("Enter a match ID.")
        else:
            try:
                match_id_value = int(match_id.strip())
                with st.spinner("Running the historical match prediction..."):
                    result = predict_from_match_id(
                        match_id_value,
                        features_df=features_df
                    )
                show_prediction(result)
            except Exception as e:
                st.error(str(e))

    with st.expander("Find a match ID from the dataset"):
        ids = get_historical_match_ids(features_df)
        search = st.text_input("Search match ID", key="historical_search")
        filtered = ids
        if search.strip():
            filtered = [x for x in ids if search.strip() in str(x)]
        st.write(filtered[:100])
