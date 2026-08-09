# Phase 07 - MLOps

## Goal

Convert your ML system into a reproducible, versioned, automated and monitored production pipeline using MLOps practices.

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

In real-world systems, ML is NOT just training models.

You must:

- Track experiments
- Version data
- Automate pipelines
- Validate inputs
- Monitor performance
- Retrain models

👉 This phase makes your ML system production-ready.

---

## Prerequisites

- Complete Phase 06
- Customer Churn Model (Phase 01)
- FastAPI + Docker (Phase 02)
- RAG + Agents understanding (Phase 05–06)

---

## 7.1 MLOps Foundations

### Resources

1. MLOps Zoomcamp  
   https://github.com/DataTalksClub/mlops-zoomcamp

2. Overview  
   https://datatalks.club/blog/mlops-zoomcamp.html

---

### Learn (High Level)

- ML lifecycle
- Reproducibility
- Experiment tracking
- Model registry
- Data versioning
- Monitoring

👉 Do NOT try to memorize — focus on flow

---

## 7.2 Data Validation (Pandera)

### Resource

https://pandera.readthedocs.io/

---

### Learn

- Data schemas
- Column validation
- Type checking
- Constraints

👉 Apply directly to your dataset

---

## 7.3 Data Versioning (DVC)

### Resource

https://dvc.org/doc/start

---

### Learn

- Track datasets
- Version data
- DVC pipelines

👉 Use local storage (no cloud required)

---

## 7.4 Experiment Tracking (MLflow)

### Resource

https://mlflow.org/docs/latest/index.html

---

### Learn

- Track runs
- Log parameters
- Log metrics
- Save artifacts

#### Model Registry

- Register models
- Version models

👉 Run MLflow locally (no paid infra)

---

## 7.5 Workflow Automation (Prefect)

### Resource

https://docs.prefect.io/latest/get-started/

---

### Learn

- Flows
- Tasks
- Scheduling
- Retries

👉 Keep it simple — one pipeline is enough

---

## 7.6 Monitoring (Evidently)

### Resource

https://docs.evidentlyai.com/

---

### Learn

- Data drift
- Model drift
- Data quality

👉 Generate reports locally

---

## ⚠️ Cost Policy

- Run everything locally
- Use free tools only
- Avoid cloud unless required
- AWS is OPTIONAL (free tier only)

---

## Apply to Flagship Project

### Customer Churn Prediction System (Upgrade)

---

## Project Goal

Build a complete ML pipeline that:

- Validates data
- Tracks experiments
- Versions datasets
- Registers models
- Automates training
- Monitors performance

---

## MLOps Pipeline

```text
Raw Data
   │
   ▼
Pandera Validation
   │
   ▼
DVC Versioning
   │
   ▼
Training Pipeline (Prefect)
   │
   ▼
MLflow Tracking
   │
   ▼
Model Registry
   │
   ▼
FastAPI Inference
   │
   ▼
Monitoring (Evidently)
```
