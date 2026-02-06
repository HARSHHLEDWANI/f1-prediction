from fastapi import APIRouter

from app.schemas.request import Top5PredictionRequest, ExpectedPerformanceRequest
from app.schemas.response import Top5PredictionResponse, ExpectedPerformanceResponse
from app.inference.top5 import predict_top5
from app.inference.expected import predict_expected_performance

router = APIRouter()


@router.post(
    "/top5",
    response_model=Top5PredictionResponse,
)
def top5_prediction(req: Top5PredictionRequest):
    """
    Predict probability of a driver finishing in Top 5.
    """
    return predict_top5(req)
@router.post(
    "/expected",
    response_model=ExpectedPerformanceResponse,
)
def expected_performance(req: ExpectedPerformanceRequest):
    """
    Predict expected finishing position and points for a driver.
    """
    return predict_expected_performance(req)