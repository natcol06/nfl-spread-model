"""Load NFL play-by-play and schedule data, with local caching."""

import os
import pandas as pd
import nflreadpy as nfl

SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
DATA_DIR = "data"

PBP_COLS = [
    "game_id", "season", "week", "season_type",
    "posteam", "defteam", "epa", "pass", "rush",
]


def load_raw(seasons=SEASONS, use_cache=True):
    """Return (pbp, games). Downloads once, then reads from data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pbp_path = os.path.join(DATA_DIR, "pbp.parquet")
    games_path = os.path.join(DATA_DIR, "games.parquet")

    if use_cache and os.path.exists(pbp_path) and os.path.exists(games_path):
        return pd.read_parquet(pbp_path), pd.read_parquet(games_path)

    pbp = nfl.load_pbp(seasons).select(PBP_COLS).to_pandas()
    games = nfl.load_schedules(seasons).to_pandas()

    pbp.to_parquet(pbp_path)
    games.to_parquet(games_path)
    return pbp, games


if __name__ == "__main__":
    pbp, games = load_raw()
    print("pbp:  ", pbp.shape)
    print("games:", games.shape)