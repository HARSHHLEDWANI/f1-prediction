from app.models.top5_model import load_top5_model
from app.schemas.request import Top5PredictionRequest
from app.schemas.response import Top5PredictionResponse

# Load once at startup
_top5_model = load_top5_model()


def predict_top5(req: Top5PredictionRequest) -> Top5PredictionResponse:
    features = {
        "grid_position": req.grid_position,
        "driver_elo": req.driver_elo,
        "constructor_strength": req.constructor_strength,
        "track_affinity": req.track_affinity,
        "recent_form": req.recent_form,
    }

    probability = _top5_model.predict_proba(features)

    return Top5PredictionResponse(
        race_id=req.race_id,
        driver_id=req.driver_id,
        top5_probability=probability,
    )
