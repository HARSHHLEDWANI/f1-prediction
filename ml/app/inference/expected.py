from app.models.expected_model import load_expected_model
from app.schemas.request import ExpectedPerformanceRequest
from app.schemas.response import ExpectedPerformanceResponse

_expected_model = load_expected_model()


def predict_expected_performance(
    req: ExpectedPerformanceRequest,
) -> ExpectedPerformanceResponse:
    features = {
        "grid_position": req.grid_position,
        "driver_elo": req.driver_elo,
        "constructor_strength": req.constructor_strength,
        "track_affinity": req.track_affinity,
        "recent_form": req.recent_form,
    }

    expected_position, expected_points = _expected_model.predict(features)

    return ExpectedPerformanceResponse(
        race_id=req.race_id,
        driver_id=req.driver_id,
        expected_position=expected_position,
        expected_points=expected_points,
    )
