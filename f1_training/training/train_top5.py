import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

from f1_training.features.build_features import build_features


ARTIFACT_DIR = Path("artifacts/top5")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def train():
    # -------------------------
    # Build features + targets
    # -------------------------
    X, y = build_features(train_until_season=2022)

    df = X.merge(
        y,
        on=["race_id", "driver_id"],
        how="inner",
    )

    feature_cols = [
        "grid_position",
        "avg_finish_last_5",
    ]

    X_train = df[feature_cols]
    y_train = df["y_top5"]

    # -------------------------
    # BASELINE: Grid position heuristic
    # -------------------------
    baseline_probs = (df["grid_position"] <= 5).astype(float)
    baseline_loss = log_loss(y_train, baseline_probs)

    # -------------------------
    # MODEL
    # -------------------------
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    model_probs = model.predict_proba(X_train)[:, 1]
    model_loss = log_loss(y_train, model_probs)

    # -------------------------
    # Sanity gate
    # -------------------------
    if model_loss >= baseline_loss:
        raise RuntimeError(
            f"Model rejected: logloss {model_loss:.4f} "
            f">= baseline {baseline_loss:.4f}"
        )

    print(f"Baseline logloss: {baseline_loss:.4f}")
    print(f"Model logloss:    {model_loss:.4f}")

    # -------------------------
    # Save artifact
    # -------------------------
    dump(model, ARTIFACT_DIR / "model.joblib")

    metadata = {
        "model_type": "GradientBoostingClassifier",
        "target": "top5",
        "features": feature_cols,
        "train_until_season": 2022,
        "metrics": {
            "log_loss": model_loss,
            "baseline_log_loss": baseline_loss,
        },
    }

    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model artifact saved to artifacts/top5/")


if __name__ == "__main__":
    train()
