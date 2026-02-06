"""
Expected Performance Model Loader

STUB IMPLEMENTATION — SAFE TO REPLACE
"""

from typing import Dict, Tuple


class DummyExpectedPerformanceModel:
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        # TODO: Replace with real regression model
        expected_position = 5.0
        expected_points = 10.0
        return expected_position, expected_points


def load_expected_model():
    """
    Factory function for expected performance model.
    """
    # TODO: Replace DummyExpectedPerformanceModel with trained model
    return DummyExpectedPerformanceModel()
