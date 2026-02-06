from fastapi import FastAPI

from app.api.predict import router as predict_router

app = FastAPI(
    title="F1 Prediction ML Service",
    version="0.1.0",
)

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Prediction routes
app.include_router(
    predict_router,
    prefix="/predict",
    tags=["predictions"],
)
