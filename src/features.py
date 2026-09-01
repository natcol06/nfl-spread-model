"""Build model-ready features from raw play-by-play and schedule data."""

import pandas as pd

WINDOW = 8
MIN_GAMES = 4

FEATURES = ["d_off_epa", "d_def_epa", "d_pace", "d_rest"]


def team_game_stats(pbp):
    """One row per team per game: offensive EPA, defensive EPA, plays."""
    plays = pbp[(pbp["season_type"] == "REG") &
                ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
    plays = plays.dropna(subset=["posteam", "defteam", "epa"])

    off = (plays.groupby(["game_id", "season", "week", "posteam"])
                .agg(off_epa=("epa", "mean"), plays=("epa", "size"))
                .reset_index()
                .rename(columns={"posteam": "team"}))

    deff = (plays.groupby(["game_id", "season", "week", "defteam"])
                 .agg(def_epa=("epa", "mean"))
                 .reset_index()
                 .rename(columns={"defteam": "team"}))

    return off.merge(deff, on=["game_id", "season", "week", "team"])


def rolling_form(team_games, window=WINDOW, min_games=MIN_GAMES):
    """Rolling averages shifted by one game so the current game never leaks."""
    team_games = team_games.sort_values(["team", "season", "week"]).copy()

    for col in ["off_epa", "def_epa", "plays"]:
        team_games[f"r_{col}"] = (
            team_games.groupby("team")[col]
            .transform(lambda s: s.shift(1)
                                  .rolling(window, min_periods=min_games)
                                  .mean())
        )
    return team_games


def build_features(pbp, games):
    """Return one row per game with difference features and both targets."""
    form = rolling_form(team_game_stats(pbp))
    form = form[["game_id", "team", "r_off_epa", "r_def_epa", "r_plays"]]

    df = games[games["game_type"] == "REG"].dropna(
        subset=["result", "spread_line"]
    )

    df = df.merge(form.add_prefix("home_"),
                  left_on=["game_id", "home_team"],
                  right_on=["home_game_id", "home_team"], how="left")
    df = df.merge(form.add_prefix("away_"),
                  left_on=["game_id", "away_team"],
                  right_on=["away_game_id", "away_team"], how="left")

    df["d_off_epa"] = df["home_r_off_epa"] - df["away_r_off_epa"]
    df["d_def_epa"] = df["home_r_def_epa"] - df["away_r_def_epa"]
    df["d_pace"] = df["home_r_plays"] - df["away_r_plays"]
    df["d_rest"] = df["home_rest"] - df["away_rest"]

    df["home_won"] = (df["result"] > 0).astype(int)
    df["home_covered"] = (df["result"] > df["spread_line"]).astype(int)

    return df.dropna(subset=FEATURES)


if __name__ == "__main__":
    from data import load_raw

    pbp, games = load_raw()
    model_df = build_features(pbp, games)
    print("rows:         ", len(model_df))
    print("cover rate:   ", round(model_df["home_covered"].mean(), 3))
    print("home win rate:", round(model_df["home_won"].mean(), 3))