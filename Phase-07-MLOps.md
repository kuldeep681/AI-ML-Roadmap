# Phase 07 - MLOps

## Goal

Transform your Machine Learning model into a reproducible, versioned, monitored and production-ready ML system using modern MLOps practices.

**Estimated Duration:** 8–10 Weeks

---

## Why this Phase?

Building an ML model is only the beginning.

In production, Machine Learning Engineers must also:

- Version datasets
- Track experiments
- Register models
- Automate training
- Validate data
- Monitor model performance
- Retrain models
- Deploy reproducible pipelines

This phase teaches you the complete Machine Learning lifecycle.

---

## Prerequisites

- Complete Phase 06
- Customer Churn Prediction System
- FastAPI Deployment
- Docker
- AWS EC2

---

## 7.1 MLOps Foundations

### Resources

1. MLOps Zoomcamp

https://github.com/DataTalksClub/mlops-zoomcamp

2. MLOps Zoomcamp Overview

https://datatalks.club/blog/mlops-zoomcamp.html

> Follow the repository in order.

---

### Learn

MLOps Fundamentals

- ML Lifecycle
- Reproducibility
- Experiment Tracking
- Model Registry
- Data Versioning
- Deployment Pipeline
- Monitoring

---

## 7.2 Pandera

### Resource

1. Pandera Documentation

https://pandera.readthedocs.io/

---

### Learn

Data Validation

- Data Schemas
- Column Validation
- Data Types
- Constraints
- Validation Reports

---

## 7.3 DVC

### Resource

1. DVC Getting Started

https://dvc.org/doc/start

---

### Learn

Data Versioning

- Initialize DVC
- Track Datasets
- Remote Storage
- Version Control
- Pipelines

---

## 7.4 MLflow

### Resource

1. MLflow Documentation

https://mlflow.org/docs/latest/index.html

---

### Learn

Experiment Tracking

- Runs
- Parameters
- Metrics
- Artifacts

Model Registry

- Register Models
- Model Versions
- Stage Management

Model Serving

- MLflow Serving Basics

---

## 7.5 Prefect

### Resource

1. Prefect Get Started

https://docs.prefect.io/latest/get-started/

---

### Learn

Workflow Automation

- Flows
- Tasks
- Scheduling
- Retries
- Logging

---

## 7.6 Evidently

### Resource

1. Evidently Documentation

https://docs.evidentlyai.com/

---

### Learn

Monitoring

- Data Drift
- Target Drift
- Data Quality
- Model Performance
- Monitoring Reports

---

## Apply to Flagship Project

### Production Customer Churn Prediction System

Upgrade your existing project.

---

### Add Features

Data Validation

- Pandera Validation

Data Versioning

- DVC

Experiment Tracking

- MLflow

Model Registry

- MLflow Registry

Workflow Automation

- Prefect

Deployment

- Docker
- AWS

Monitoring

- Evidently Reports

CI/CD

- GitHub Actions

---

### MLOps Pipeline

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
Training Pipeline
    │
    ▼
MLflow Tracking
    │
    ▼
Model Registry
    │
    ▼
FastAPI
    │
    ▼
Docker
    │
    ▼
AWS EC2
    │
    ▼
Evidently Monitoring
```

---

### Build Order

1. Validate Dataset using Pandera
2. Version Dataset using DVC
3. Track Experiments using MLflow
4. Register Best Model
5. Build Prefect Training Pipeline
6. Integrate GitHub Actions
7. Deploy Updated API
8. Generate Evidently Reports
9. Monitor Production Model

---

## Deliverables

Project

- Production Customer Churn Prediction System (MLOps Version)

Repository

```text
customer-churn-prediction/
│
├── app/
├── data/
├── dvc.yaml
├── flows/
├── mlruns/
├── monitoring/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── .github/
└── README.md
```

Artifacts

- DVC Pipeline
- MLflow Experiments
- Model Registry
- Prefect Flow
- Evidently Reports
- GitHub Actions Workflow
- Docker Deployment

---

## Skills Gained

- MLOps Fundamentals
- Data Validation
- Data Versioning
- Experiment Tracking
- Model Registry
- Workflow Automation
- Model Monitoring
- Production ML Pipelines
- Continuous Integration
- Continuous Deployment

---

## Completion Criteria

Move to **Phase 08** only after:

- Completed the MLOps Zoomcamp.
- Added Pandera validation.
- Versioned data using DVC.
- Tracked experiments using MLflow.
- Registered the production model.
- Automated training using Prefect.
- Integrated GitHub Actions.
- Generated Evidently monitoring reports.
- Deployed the updated application.

---

## Ready for Next Phase

In **Phase 08**, you'll learn Kubernetes fundamentals and deploy your Dockerized Machine Learning application locally using Kubernetes.
