import os
import zipfile
import io
import yaml
import requests

CONFIG_PATH = "config/leagues.yaml"
RAW_PATH = "data/raw/cricsheet"
BASE_URL = "https://cricsheet.org/downloads"


def load_leagues():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["leagues"]


def download_and_extract(league_key, slug):
    url = f"{BASE_URL}/{slug}_json.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    out_dir = os.path.join(RAW_PATH, league_key)
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)

    n_files = len([f for f in os.listdir(out_dir) if f.endswith(".json")])
    print(f"✅ {league_key}: {n_files} matches -> {out_dir}")


def run(only=None):
    leagues = load_leagues()
    for key, meta in leagues.items():
        if not meta.get("active", False):
            continue
        if only and key not in only:
            continue
        download_and_extract(key, meta["cricsheet_slug"])


if __name__ == "__main__":
    run()