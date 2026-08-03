# Production Customer Churn ML Platform

## Goal

Build a complete production-ready Machine Learning platform that predicts whether a customer is likely to churn.

The project starts as a simple Machine Learning model and gradually evolves into a fully deployed, monitored and production-grade ML system.

---

## Why this Project?

This project teaches the complete lifecycle of a Machine Learning application.

Instead of stopping after training a model, you'll learn how to:

- Build ML models
- Deploy APIs
- Store predictions
- Containerize applications
- Deploy to the cloud
- Track experiments
- Version datasets
- Monitor production models
- Deploy using Kubernetes

By the end of the roadmap, this project becomes a real production ML platform.

---

## Project Timeline

| Phase    | Progress                           |
| -------- | ---------------------------------- |
| Phase 01 | Machine Learning Development       |
| Phase 02 | API Development & Cloud Deployment |
| Phase 07 | MLOps Integration                  |
| Phase 08 | Kubernetes Deployment              |

---

## Final Architecture

```text
                Dataset
                   │
                   ▼
          Data Validation (Pandera)
                   │
                   ▼
            Feature Engineering
                   │
                   ▼
              Model Training
                   │
                   ▼
         MLflow Experiment Tracking
                   │
                   ▼
            Model Registry
                   │
                   ▼
              FastAPI API
                   │
                   ▼
             Docker Container
                   │
                   ▼
              AWS EC2 Server
                   │
                   ▼
            Kubernetes Cluster
                   │
                   ▼
          Monitoring (Evidently)
```

---

# Tech Stack

## Machine Learning

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

---

## Backend

- FastAPI
- Pydantic
- SQLAlchemy

---

## Database

- PostgreSQL

---

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

## Cloud

- AWS EC2
- AWS S3
- Nginx
- CloudWatch

---

## MLOps

- Pandera
- DVC
- MLflow
- Prefect
- Evidently

---

## Deployment

- Kubernetes
- kind

---

# Learning Roadmap

## Phase 01

### Goal

Build and evaluate a Machine Learning model.

### Topics Covered

Machine Learning

- Data Cleaning
- EDA
- Feature Engineering
- Train/Test Split
- Logistic Regression
- Random Forest
- XGBoost
- Cross Validation
- Hyperparameter Tuning
- SHAP Explainability

### Build

- Dataset Preparation
- ML Pipeline
- Trained Model
- Evaluation Report
- SHAP Report
- Saved joblib Model

---

## Phase 02

### Goal

Convert the ML model into a production application.

### Topics Covered

Backend

- FastAPI
- Pydantic
- SQLAlchemy

Infrastructure

- Docker
- Docker Compose

CI/CD

- GitHub Actions

Cloud

- AWS EC2
- S3
- Nginx
- CloudWatch

### Build

APIs

- /predict
- /health
- /explain

Database

- Prediction Logs

Deployment

- Docker
- AWS EC2
- Nginx

---

## Phase 07

### Goal

Apply real-world MLOps practices.

### Topics Covered

Data Validation

- Pandera

Version Control

- DVC

Experiment Tracking

- MLflow

Workflow Automation

- Prefect

Monitoring

- Evidently

### Build

- Data Validation Pipeline
- Dataset Versioning
- Experiment Tracking
- Model Registry
- Automated Training Workflow
- Monitoring Dashboard

---

## Phase 08

### Goal

Deploy using Kubernetes.

### Topics Covered

- Pods
- Deployments
- Services
- ConfigMaps
- Secrets
- Scaling

### Build

- Kubernetes Deployment
- Kubernetes Service
- ConfigMap
- Secret
- Local kind Cluster

---

# Final Features

## Machine Learning

- Customer Churn Prediction
- Model Explainability
- Hyperparameter Tuning

---

## API

- Prediction API
- Explainability API
- Health Check API

---

## Database

- Store Prediction History
- Store Model Metadata

---

## Cloud

- AWS Deployment
- Reverse Proxy
- Secure Environment Variables

---

## MLOps

- Data Validation
- Experiment Tracking
- Model Registry
- Automated Training
- Monitoring
- Version Control

---

## Kubernetes

- Container Orchestration
- Scaling
- Configuration Management

---

# Folder Structure

```text
customer-churn-ml-platform/

├── app/
├── data/
├── models/
├── notebooks/
├── monitoring/
├── flows/
├── kubernetes/
├── tests/
├── .github/
│   └── workflows/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Skills Covered

## Machine Learning

- Classical ML
- XGBoost
- SHAP
- Feature Engineering

---

## Backend

- FastAPI
- REST APIs
- SQLAlchemy

---

## Databases

- PostgreSQL

---

## DevOps

- Docker
- Docker Compose
- GitHub Actions

---

## Cloud

- AWS EC2
- S3
- Nginx
- CloudWatch

---

## MLOps

- Pandera
- DVC
- MLflow
- Prefect
- Evidently

---

## Kubernetes

- Deployments
- Services
- ConfigMaps
- Secrets

---

# Final Deliverables

- Production-ready Machine Learning API
- Dockerized Application
- AWS Deployment
- GitHub Actions Pipeline
- MLflow Experiment Tracking
- DVC Dataset Versioning
- Automated Training Workflow
- Monitoring Dashboard
- Kubernetes Deployment

---

# Interview Topics Covered

Machine Learning

- Supervised Learning
- Model Evaluation
- XGBoost
- SHAP
- Feature Engineering

Backend

- FastAPI
- REST APIs
- Database Integration

Cloud & DevOps

- Docker
- CI/CD
- AWS EC2
- Nginx

MLOps

- MLflow
- DVC
- Pandera
- Prefect
- Evidently

Kubernetes

- Pods
- Deployments
- Services

---

# Future Improvements

Possible enhancements after completing the roadmap:

- Real-time prediction pipeline
- Batch prediction pipeline
- Drift-triggered model retraining
- A/B model deployment
- Feature Store integration
- Kubernetes deployment on Amazon EKS
- Model serving with KServe or BentoML
