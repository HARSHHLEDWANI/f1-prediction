"""
Top 5 Probability Model Loader

IMPORTANT:
- This is a STUB implementation.
- Replace `DummyTop5Model` with a real trained model loader later.
- Public interface MUST remain unchanged.
"""

from typing import Dict


class DummyTop5Model:
    """
    Temporary placeholder model.
    This exists ONLY to validate the pipeline.
    """

    def predict_proba(self, features: Dict[str, float]) -> float:
        # TODO: Replace with real ML inference
        return 0.5


def load_top5_model():
    """
    Model factory.
    Later this will:
    - load a .joblib / .onnx model
    - handle versioning
    """
    # TODO: Replace DummyTop5Model with real model
    return DummyTop5Model()
