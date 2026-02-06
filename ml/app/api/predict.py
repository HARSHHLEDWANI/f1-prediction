import logging
from fastapi import APIRouter

from app.schemas.request import (
    Top5PredictionRequest,
    ExpectedPerformanceRequest,
)
from app.schemas.response import (
    Top5PredictionResponse,
    ExpectedPerformanceResponse,
)
from app.inference.top5 import predict_top5
from app.inference.expected import predict_expected_performance

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/top5",
    response_model=Top5PredictionResponse,
)
def top5_prediction(req: Top5PredictionRequest):
    logger.info(
        "Top5 prediction requested",
        extra={
            "race_id": req.race_id,
            "driver_id": req.driver_id,
        },
    )

    result = predict_top5(req)

    logger.info(
        "Top5 prediction completed",
        extra={
            "race_id": result.race_id,
            "driver_id": result.driver_id,
            "probability": result.top5_probability,
        },
    )

    return result


@router.post(
    "/expected",
    response_model=ExpectedPerformanceResponse,
)
def expected_performance(req: ExpectedPerformanceRequest):
    logger.info(
        "Expected performance prediction requested",
        extra={
            "race_id": req.race_id,
            "driver_id": req.driver_id,
        },
    )

    result = predict_expected_performance(req)

    logger.info(
        "Expected performance prediction completed",
        extra={
            "race_id": result.race_id,
            "driver_id": result.driver_id,
            "expected_position": result.expected_position,
        },
    )

    return result
