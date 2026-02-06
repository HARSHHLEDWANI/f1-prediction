from pydantic import BaseModel


# ---------- TOP 5 PROBABILITY ----------

class Top5PredictionResponse(BaseModel):
    race_id: str
    driver_id: str

    top5_probability: float    # value between 0 and 1


# ---------- EXPECTED VS ACTUAL ----------

class ExpectedPerformanceResponse(BaseModel):
    race_id: str
    driver_id: str

    expected_position: float
    expected_points: float
