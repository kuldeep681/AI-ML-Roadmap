# Project 02 - Customer Churn (MLOps Upgrade)

## Phase Mapping

- Phase 07
- Extends: Project 01

---

## Goal

Convert your ML model into a production-grade ML pipeline using MLOps.

---

## What You Already Have

From Phase 01:

- Trained ML model
- Training code
- Dataset

---

## What You Will Build NOW

### MLOps System

- Data validation
- Data versioning
- Experiment tracking
- Model registry
- Automated pipeline

---

## Tech Stack

- Pandera
- DVC
- MLflow
- Prefect
- FastAPI
- Docker (basic)

---

## Build Steps

1. Add Pandera validation
2. Setup DVC for dataset
3. Integrate MLflow
4. Track experiments
5. Register best model
6. Create Prefect pipeline
7. Build FastAPI for inference
8. Dockerize application

---

## Pipeline Architecture

```text
Data → Validation → DVC → Training → MLflow → Model Registry → API
```
