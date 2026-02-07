from fastapi import FastAPI
from app.api.predict import router as predict_router
from app.config import setup_logging
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

setup_logging()
logger = logging.getLogger("app.errors")


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
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled exception",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )