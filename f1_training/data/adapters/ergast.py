import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]  # f1-training/
RAW_DIR = BASE_DIR / "data" / "raw"


def load_ergast_races() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "races.csv")

    return pd.DataFrame({
        "race_id": df["raceId"],
        "season": df["year"],
        "round": df["round"],
        "circuit_id": df["circuitId"],
        "race_date": pd.to_datetime(df["date"]),
    })


def load_ergast_results() -> pd.DataFrame:
    results = pd.read_csv(RAW_DIR / "results.csv")
    status_df = pd.read_csv(RAW_DIR / "status.csv")

    # Join to get human-readable status
    results = results.merge(
        status_df[["statusId", "status"]],
        on="statusId",
        how="left",
    )

    status_str = results["status"].fillna("").str.lower()
    classified_mask = status_str.str.contains("finished")

    return pd.DataFrame({
        "race_id": results["raceId"],
        "driver_id": results["driverId"],
        "team_id": results["constructorId"],
        "grid_position": results["grid"].fillna(0).astype(int),
        "finish_position": results["position"].where(classified_mask),
        "status": results["status"],
        "points": results["points"].fillna(0.0),
    })

def load_ergast_qualifying() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "qualifying.csv")

    def time_to_ms(t):
        if pd.isna(t):
            return None
        try:
            mins, rest = t.split(":")
            secs, ms = rest.split(".")
            return (
                int(mins) * 60_000
                + int(secs) * 1_000
                + int(ms)
            )
        except Exception:
            return None

    q_time_ms = df["q3"].fillna(df["q2"]).fillna(df["q1"])
    q_time_ms = q_time_ms.apply(time_to_ms)

    return pd.DataFrame({
        "race_id": df["raceId"],
        "driver_id": df["driverId"],
        "qualifying_position": df["position"],
        "qualifying_time_ms": q_time_ms,
    })


def load_all():
    """
    Single entry point used by training code.
    """
    races = load_ergast_races()
    results = load_ergast_results()
    qualifying = load_ergast_qualifying()

    return races, results, qualifying
