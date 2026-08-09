# 🚀 Project 02 - Customer Churn (Phase 07 - MLOps Upgrade)

---

## 🎯 Objective

Convert ML system into a production-grade MLOps pipeline.

---

## 🧠 System Architecture

Data → Validation → Versioning → Training → Tracking → Registry → API → Docker

---

## 📁 Updated Project Structure

```text
project/
│
├── data/
│
├── src/
│ ├── validation/
│ │ └── schema.py
│ │
│ ├── pipelines/
│ │ └── training_pipeline.py
│ │
│ ├── api/
│ │ ├── main.py
│ │ └── routes.py
│ │
│ └── models/
│
├── dvc.yaml
├── Dockerfile
├── requirements.txt
```

---

## 🧩 Build Steps

1. Add Pandera validation
2. Setup DVC
3. Integrate MLflow
4. Track experiments
5. Register model
6. Create Prefect pipeline
7. Build FastAPI
8. Dockerize application

---

## 🔌 API Specification

Endpoint: POST /predict

Response:
{
"prediction": 1,
"probability": 0.82
}

---

## 🚫 Constraints

- Lightweight system
- Free-tier friendly
- Avoid heavy tools

---

## 🎯 Final Outcome

- Reproducible ML pipeline
- Experiment tracking
- Dockerized API
