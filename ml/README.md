# F1 Prediction — ML Inference Service

This service provides machine-learning–based predictions for Formula 1 races.

## Scope (Phase 1)
- Top 5 finish probability prediction
- Expected driver performance estimation
- Stateless ML inference
- Docker-ready FastAPI service

## What This Service Does NOT Do
- No user authentication
- No database access
- No feature engineering
- No model training

## Architecture
- FastAPI for inference
- Models loaded once at startup
- Clear separation between:
  - schemas
  - inference logic
  - model loading

## Model Status
⚠️ Current models are **stub implementations**.
They will be replaced with trained models once:
- feature engineering is finalized
- offline evaluation is complete

Public interfaces will remain stable.
