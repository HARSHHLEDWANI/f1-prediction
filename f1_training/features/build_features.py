import pandas as pd
from typing import Tuple

from f1_training.data.adapters.ergast import load_all


def build_features(
    train_until_season: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build leak-free features and targets using canonical data.
    Returns:
      - features_df
      - targets_df
    """

    races, results, qualifying = load_all()

    # -------------------------
    # Temporal filtering
    # -------------------------
    races = races.sort_values("race_date")

    train_race_ids = races.loc[
        races["season"] < train_until_season, "race_id"
    ]

    results = results[results["race_id"].isin(train_race_ids)]
    qualifying = qualifying[qualifying["race_id"].isin(train_race_ids)]

    # -------------------------
    # Join results + qualifying
    # -------------------------
    df = results.merge(
        qualifying,
        on=["race_id", "driver_id"],
        how="left",
    )

    df = df.merge(
        races[["race_id", "season"]],
        on="race_id",
        how="left",
    )

    df = df.sort_values(["driver_id", "race_id"])

    # -------------------------
    # NORMALIZE finish_position dtype
    # -------------------------
    df["finish_position"] = pd.to_numeric(
        df["finish_position"],
        errors="coerce",
    )

    # -------------------------
    # DRIVER FORM FEATURE
    # avg_finish_last_5
    # -------------------------
    df["finish_pos_filled"] = df["finish_position"].fillna(30)

    df["avg_finish_last_5"] = (
        df.groupby("driver_id")["finish_pos_filled"]
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # -------------------------
    # IMPUTATION (EXPLICIT & AUDITABLE)
    # -------------------------

    # Rookie / insufficient history → global median
    global_finish_median = df["avg_finish_last_5"].median()
    df["avg_finish_last_5"] = df["avg_finish_last_5"].fillna(
        global_finish_median
    )

    # Rare grid NaNs → worst grid
    max_grid = df["grid_position"].max()
    df["grid_position"] = df["grid_position"].fillna(max_grid)

    # -------------------------
    # TARGET: Top-5
    # -------------------------
    df["y_top5"] = (df["finish_position"] <= 5).astype(int)
    df.loc[df["finish_position"].isna(), "y_top5"] = 0

    # -------------------------
    # Final outputs
    # -------------------------
    features = df[
        [
            "race_id",
            "driver_id",
            "grid_position",
            "avg_finish_last_5",
        ]
    ].copy()

    targets = df[
        [
            "race_id",
            "driver_id",
            "y_top5",
        ]
    ].copy()

    return features, targets
