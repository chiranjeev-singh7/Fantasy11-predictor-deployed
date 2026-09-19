import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.cricbuzz.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    )
}

PLAYER_ROLES = [
    "WK-Batter",
    "Batting Allrounder",
    "Bowling Allrounder",
    "Batter",
    "Bowler",
]

TEAM_ALIASES = {
    "afg": "afghanistan",
    "afghanistan": "afghanistan",
    "ind": "india",
    "india": "india",
    "aus": "australia",
    "australia": "australia",
    "eng": "england",
    "england": "england",
    "pak": "pakistan",
    "pakistan": "pakistan",
    "sa": "south africa",
    "rsa": "south africa",
    "south africa": "south africa",
    "nz": "new zealand",
    "new zealand": "new zealand",
    "sl": "sri lanka",
    "sri lanka": "sri lanka",
    "ban": "bangladesh",
    "bangladesh": "bangladesh",
    "wi": "west indies",
    "west indies": "west indies",
    "ire": "ireland",
    "ireland": "ireland",
    "zim": "zimbabwe",
    "zimbabwe": "zimbabwe",
    "ned": "netherlands",
    "netherlands": "netherlands",
    "usa": "united states",
    "united states": "united states",
    "uae": "united arab emirates",
    "united arab emirates": "united arab emirates",
    "nepal": "nepal",
    "om": "oman",
    "oman": "oman",
    "scotland": "scotland",
    "sco": "scotland",
    "nam": "namibia",
    "namibia": "namibia",
    "wi": "west indies",
}

TEAM_SEARCH_ALIASES = {
    "afghanistan": ["afghanistan", "afg"],
    "india": ["india", "ind"],
    "australia": ["australia", "aus"],
    "england": ["england", "eng"],
    "pakistan": ["pakistan", "pak"],
    "south africa": ["south africa", "rsa", "sa"],
    "new zealand": ["new zealand", "nz"],
    "sri lanka": ["sri lanka", "sl"],
    "bangladesh": ["bangladesh", "ban"],
    "west indies": ["west indies", "wi"],
    "ireland": ["ireland", "ire"],
    "zimbabwe": ["zimbabwe", "zim"],
    "netherlands": ["netherlands", "ned"],
    "united states": ["united states", "usa"],
    "united arab emirates": ["united arab emirates", "uae"],
    "nepal": ["nepal"],
    "oman": ["oman", "om"],
    "scotland": ["scotland", "sco"],
    "namibia": ["namibia", "nam"],
}

FORMAT_MAP = {
    "t20i": "T20",
    "t20": "T20",
    "odi": "ODI",
    "test": "Test",
    "hundred": "Hundred",
}

def fetch_page(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def normalize_team_name(name):
    name = str(name).lower().strip()
    return TEAM_ALIASES.get(name, name)

def team_matches(text, team):
    text = str(text).lower()
    team = normalize_team_name(team)
    aliases = TEAM_SEARCH_ALIASES.get(
        team,
        [team]
    )

    return any(
        re.search(
            rf"\b{re.escape(name)}\b",
            text
        )
        for name in aliases
    )

def parse_match_slug(live_url):
    part = live_url.split(
        "/live-cricket-scores/",
        1
    )[1].strip("/")

    parts = part.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Could not parse match URL: {live_url}"
        )

    match_id = parts[0]
    slug = parts[-1]

    first_slug = None
    second_slug = None

    if "-vs-" in slug:
        sides = slug.split("-vs-", 1)
        first_slug = sides[0]
        second_slug = sides[1]

        second_slug = re.sub(
            r"-(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|test|odi|t20i|t20|hundred).*$",
            "",
            second_slug,
            flags=re.IGNORECASE
        )

    return (
        match_id,
        normalize_team_name(first_slug) if first_slug else None,
        normalize_team_name(second_slug) if second_slug else None
    )

def find_upcoming_match(team1, team2):
    url = (
        f"{BASE_URL}/cricket-match/"
        "live-scores/upcoming-matches"
    )

    soup = fetch_page(url)

    team1 = normalize_team_name(team1)
    team2 = normalize_team_name(team2)

    candidates = []

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")

        if "/live-cricket-scores/" not in href:
            continue

        text = link.get_text(" ", strip=True)

        if not text:
            continue

        if (
            team_matches(text, team1)
            and team_matches(text, team2)
        ):
            candidates.append({
                "text": text,
                "url": urljoin(BASE_URL, href)
            })

    unique = []
    seen = set()

    for candidate in candidates:
        if candidate["url"] in seen:
            continue

        seen.add(candidate["url"])
        unique.append(candidate)

    if not unique:
        raise ValueError(
            f"No upcoming Cricbuzz match found for "
            f"{team1} vs {team2}"
        )

    match = unique[0]

    print()
    print("=" * 70)
    print("MATCH FOUND")
    print("=" * 70)
    print(match["text"])
    print(match["url"])

    return match

def convert_to_squad_url(live_url):
    if "/live-cricket-scores/" not in live_url:
        return live_url

    part = live_url.split(
        "/live-cricket-scores/",
        1
    )[1].strip("/")

    parts = part.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Could not extract match ID from {live_url}"
        )

    match_id = parts[0]
    match_slug = parts[-1]

    return (
        f"{BASE_URL}/cricket-match-squads/"
        f"{match_id}/{match_slug}"
    )

def detect_format(soup, live_url):
    url_text = live_url.lower()
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    text = soup.get_text(" ", strip=True).lower()

    if "t20i" in url_text or "t20i" in title:
        return "T20"
    if "t20" in url_text or "t20" in title:
        return "T20"
    if "odi" in url_text or "odi" in title:
        return "ODI"
    if "hundred" in url_text or "hundred" in title:
        return "Hundred"
    if re.search(r"\btest match\b|\btests\b", url_text) or re.search(r"\btest match\b|\btests\b", title):
        return "Test"

    combined = f"{title} {text}"
    if re.search(r"\bt20i\b|\bt20\b", combined):
        return "T20"
    if re.search(r"\bodi\b", combined):
        return "ODI"
    if re.search(r"\bhundred\b", combined):
        return "Hundred"
    if re.search(r"\btest match\b|\btests\b", combined):
        return "Test"

    return "T20"

def clean_venue(value):
    if not value:
        return None
    value = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    value = value.replace('\\"', '"').replace('\"', '"')
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^Venue\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    value = value.split("Date & Time:", 1)[0].strip(" |•")
    if len(value) < 3 or value.lower() in {"venue", "date & time"}:
        return None
    return value

def extract_venue(soup):
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        title = anchor.get("title", "")
        text = anchor.get_text(" ", strip=True)
        candidate = clean_venue(title) or clean_venue(text)
        if not candidate:
            continue
        if "/venues/" in href or "stadium" in candidate.lower() or "ground" in candidate.lower() or "stadium" in text.lower():
            return candidate

    raw_html = str(soup)
    patterns = [
        r'"title"\s*:\s*"([^"]*(?:Stadium|stadium|Ground|ground)[^"]*)"',
        r'Venue\s*:\s*.*?"title"\s*:\s*"([^"]+)"',
        r'Venue\s*:\s*.*?>([^<]{3,150})<',
        r'Venue\s*:\s*([^<]{3,150})',
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = clean_venue(match.group(1))
            if candidate:
                return candidate

    text = soup.get_text(" ", strip=True)
    match = re.search(
        r"Venue\s*:\s*(.+?)(?:Date\s*&\s*Time\s*:|$)",
        text,
        flags=re.IGNORECASE
    )
    if match:
        candidate = clean_venue(match.group(1))
        if candidate:
            return candidate

    return "Unknown"

def get_match_details(live_url):
    soup = fetch_page(live_url)

    match_id, team1, team2 = parse_match_slug(
        live_url
    )

    fmt = detect_format(
        soup,
        live_url
    )

    venue = extract_venue(
        soup
    )

    return {
        "match_id": match_id,
        "team1": team1,
        "team2": team2,
        "format": fmt,
        "venue": venue,
        "match_url": live_url,
    }

def parse_player_text(text):
    text = " ".join(
        str(text).split()
    )

    captain = "(C)" in text
    vice_captain = "(VC)" in text
    wicketkeeper = "(WK)" in text

    text = re.sub(
        r"\s*\(C\)",
        "",
        text
    )

    text = re.sub(
        r"\s*\(VC\)",
        "",
        text
    )

    text = re.sub(
        r"\s*\(WK\)",
        "",
        text
    )

    role = None

    for possible_role in PLAYER_ROLES:
        if text.endswith(possible_role):
            role = possible_role
            text = text[
                :-len(possible_role)
            ].strip()
            break

    if role is None:
        return None

    return {
        "name": text,
        "role": role,
        "captain": captain,
        "vice_captain": vice_captain,
        "wicketkeeper": wicketkeeper,
    }

def extract_players(url):
    soup = fetch_page(url)

    players = []
    seen = set()

    for link in soup.find_all(
        "a",
        href=True
    ):
        href = link.get(
            "href",
            ""
        )

        if "/profiles/" not in href:
            continue

        text = link.get_text(
            " ",
            strip=True
        )

        player = parse_player_text(
            text
        )

        if player is None:
            continue

        key = player["name"].lower()

        if key in seen:
            continue

        seen.add(key)

        player["profile_url"] = urljoin(
            BASE_URL,
            href
        )

        players.append(
            player
        )

    return players

def _extract_page_team_order(soup):
    page_teams = []
    seen = set()

    for element in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = element.get_text(" ", strip=True)
        normalized = normalize_team_name(text)

        if normalized in TEAM_SEARCH_ALIASES and normalized not in seen:
            seen.add(normalized)
            page_teams.append(normalized)

    return page_teams[:2]


def get_match_squad(
    team1,
    team2,
    squad_url=None,
    live_url=None
):
    if live_url is None:
        if squad_url is None:
            match = find_upcoming_match(
                team1,
                team2
            )
            live_url = match["url"]

    if squad_url is None:
        if live_url is None:
            raise ValueError("A Cricbuzz live URL or squad URL is required")
        squad_url = convert_to_squad_url(live_url)

    soup = fetch_page(squad_url)

    match_id, actual_team1, actual_team2 = parse_match_slug(
        live_url if live_url else squad_url
    )

    if actual_team1 is None:
        actual_team1 = normalize_team_name(team1)

    if actual_team2 is None:
        actual_team2 = normalize_team_name(team2)

    details = get_match_details(
        live_url if live_url else squad_url
    )

    page_team_order = _extract_page_team_order(soup)

    if len(page_team_order) != 2:
        page_team_order = [
            normalize_team_name(actual_team2),
            normalize_team_name(actual_team1)
        ]

    all_players = []
    playing_xi_players = []
    current_section = None
    seen = set()

    for element in soup.find_all(["h1", "h2", "h3", "h4", "a"]):
        if element.name in ["h1", "h2", "h3", "h4"]:
            heading = element.get_text(" ", strip=True).lower()

            if "support staff" in heading:
                current_section = None
            elif "playing xi" in heading or heading == "playing xi":
                current_section = "playing_xi"
            elif heading == "squad":
                current_section = "squad"
            continue

        if element.name != "a":
            continue

        href = element.get("href", "")

        if "/profiles/" not in href:
            continue

        text = element.get_text(" ", strip=True)
        player = parse_player_text(text)

        if player is None:
            continue

        key = player["name"].strip().lower()

        if key in seen:
            continue

        seen.add(key)

        player["profile_url"] = urljoin(
            BASE_URL,
            href
        )

        all_players.append(player)

        if current_section == "playing_xi":
            playing_xi_players.append(player)

    if len(all_players) < 2:
        raise ValueError(
            f"Could not identify Cricbuzz squad players. Found {len(all_players)} players."
        )

    if len(playing_xi_players) >= 22:
        page_team1_xi = playing_xi_players[:11]
        page_team2_xi = playing_xi_players[11:22]
        is_confirmed_xi = True
    else:
        page_team1_xi = []
        page_team2_xi = []
        is_confirmed_xi = False

    page_team1 = page_team_order[0]
    page_team2 = page_team_order[1]

    actual_team1_norm = normalize_team_name(actual_team1)
    actual_team2_norm = normalize_team_name(actual_team2)

    team1_playing_xi = []
    team2_playing_xi = []

    if is_confirmed_xi:
        xi_by_team = {
            page_team1: page_team1_xi,
            page_team2: page_team2_xi
        }

        team1_playing_xi = xi_by_team.get(actual_team1_norm, [])
        team2_playing_xi = xi_by_team.get(actual_team2_norm, [])

        if len(team1_playing_xi) != 11 or len(team2_playing_xi) != 11:
            raise ValueError(
                "Cricbuzz playing XI could not be mapped to the requested teams. "
                f"Page order: {page_team_order}, requested: {actual_team1_norm} vs {actual_team2_norm}."
            )

        for player in team1_playing_xi:
            player["team"] = actual_team1
            player["playing_xi"] = True

        for player in team2_playing_xi:
            player["team"] = actual_team2
            player["playing_xi"] = True

    team1_squad = []
    team2_squad = []

    if is_confirmed_xi:
        page_team1_squad = all_players[:11]
        page_team2_squad = all_players[11:]

        if page_team1 == actual_team1_norm:
            team1_squad = page_team1_squad
            team2_squad = page_team2_squad
        else:
            team1_squad = page_team2_squad
            team2_squad = page_team1_squad

        team1_xi_names = {
            p["name"].lower() for p in team1_playing_xi
        }
        team2_xi_names = {
            p["name"].lower() for p in team2_playing_xi
        }

        for player in team1_squad:
            player["team"] = actual_team1
            player["playing_xi"] = player["name"].lower() in team1_xi_names

        for player in team2_squad:
            player["team"] = actual_team2
            player["playing_xi"] = player["name"].lower() in team2_xi_names
    else:
        total_players = len(all_players)
        split_index = total_players // 2

        if split_index == 0:
            raise ValueError(
                "Could not split Cricbuzz squad into two teams."
            )

        page_team1_squad = all_players[:split_index]
        page_team2_squad = all_players[split_index:]

        if page_team1 == actual_team1_norm:
            team1_squad = page_team1_squad
            team2_squad = page_team2_squad
        elif page_team2 == actual_team1_norm:
            team1_squad = page_team2_squad
            team2_squad = page_team1_squad
        else:
            raise ValueError(
                "Cricbuzz squad teams could not be mapped to the requested teams. "
                f"Page order: {page_team_order}, requested: {actual_team1_norm} vs {actual_team2_norm}."
            )

        for player in team1_squad:
            player["team"] = actual_team1
            player["playing_xi"] = False

        for player in team2_squad:
            player["team"] = actual_team2
            player["playing_xi"] = False

    team1_bench = [
        p for p in team1_squad
        if p.get("playing_xi") is False
    ]

    team2_bench = [
        p for p in team2_squad
        if p.get("playing_xi") is False
    ]

    print()
    print("=" * 70)
    print("CRICBUZZ MATCH DETAILS")
    print("=" * 70)
    print(
        f"Match: {actual_team1.title()} vs {actual_team2.title()}"
    )
    print(
        f"Format: {details['format']}"
    )
    print(
        f"Venue: {details['venue']}"
    )
    print(
        f"Squad: {len(team1_squad)} + {len(team2_squad)}"
    )

    if is_confirmed_xi:
        print(
            f"Playing XI: {len(team1_playing_xi)} + {len(team2_playing_xi)}"
        )
        print(
            f"Bench: {len(team1_bench)} + {len(team2_bench)}"
        )
    else:
        print("Playing XI: Not announced")
        print("Using full squad for prediction")

    return {
        "team1": actual_team1,
        "team2": actual_team2,
        "format": details["format"],
        "venue": details["venue"],
        "match_id": match_id,
        "team1_players": team1_playing_xi if is_confirmed_xi else team1_squad,
        "team2_players": team2_playing_xi if is_confirmed_xi else team2_squad,
        "team1_playing_xi": team1_playing_xi,
        "team2_playing_xi": team2_playing_xi,
        "team1_squad": team1_squad,
        "team2_squad": team2_squad,
        "team1_bench": team1_bench,
        "team2_bench": team2_bench,
        "squads_confirmed_xi": is_confirmed_xi,
        "match_url": live_url,
        "squad_url": squad_url,
        "total_players": len(team1_playing_xi) + len(team2_playing_xi) if is_confirmed_xi else len(team1_squad) + len(team2_squad),
    }

def fetch_fixture_for_prediction(
    team1,
    team2
):
    match = find_upcoming_match(
        team1,
        team2
    )

    return get_match_squad(
        team1,
        team2,
        live_url=match["url"]
    )

if __name__ == "__main__":
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
    print("=" * 70)
    print("PLAYERS")
    print("=" * 70)

    print(
        f"\n{fixture['team1'].title()}"
    )

    for player in fixture["team1_players"]:
        status = (
            "XI"
            if player["playing_xi"] is True
            else "Bench"
            if player["playing_xi"] is False
            else "Squad"
        )
        print(
            f"- {player['name']} | "
            f"{player['role']} | "
            f"{status}"
        )

    print(
        f"\n{fixture['team2'].title()}"
    )

    for player in fixture["team2_players"]:
        status = (
            "XI"
            if player["playing_xi"] is True
            else "Bench"
            if player["playing_xi"] is False
            else "Squad"
        )
        print(
            f"- {player['name']} | "
            f"{player['role']} | "
            f"{status}"
        )
