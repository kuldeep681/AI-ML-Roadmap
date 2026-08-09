# Phase 02 - FastAPI, Docker, CI/CD & AWS Deployment

## Goal

Convert your Machine Learning model into a **production-ready application** and deploy it to the cloud.

**Estimated Duration:** 5–7 Weeks

---

## Why this Phase?

Training a model is only the first step.

A Machine Learning Engineer must also know how to:

- Build APIs
- Containerize applications
- Automate testing
- Deploy to cloud
- Run and manage real systems

> This phase transforms your ML model into a real-world application.

---

## Prerequisites

- Complete Phase 01
- Customer Churn Model trained
- Model saved using joblib
- Basic understanding of prediction workflow

---

## 2.1 FastAPI (API Development)

### Resources

1. FastAPI Official Tutorial  
   https://fastapi.tiangolo.com/tutorial/

2. FastAPI SQL Databases  
   https://fastapi.tiangolo.com/tutorial/sql-databases/

---

### Learn

#### API Development

- Routing
- Path Parameters
- Query Parameters
- Request Body
- Response Models

#### Validation

- Pydantic Models
- Request & Response Validation

#### Dependency Injection

- Depends()
- Reusable Dependencies

#### Database

- SQLAlchemy Basics
- Database Sessions
- CRUD Operations

#### Error Handling

- HTTPException
- Custom Errors

#### Documentation

- Swagger UI
- OpenAPI

---

### Objective

Convert your ML model into an API that:

- Accepts input data
- Returns prediction
- Returns model explanation

---

## 2.2 Docker & Docker Compose

### Resources

1. Docker Get Started  
   https://docs.docker.com/get-started/

2. Docker Compose  
   https://docs.docker.com/compose/gettingstarted/

---

### Learn

#### Docker

- Images
- Containers
- Dockerfile
- Volumes
- Networks

#### Docker Compose

- Multi-service setup
- Environment variables
- Service communication

---

### Objective

Run your full application using:

```bash
docker compose up --build
```

Architecture:

```
FastAPI → PostgreSQL
```

---

## 2.3 GitHub Actions (CI)

### Resources

1. GitHub Actions Quickstart  
   https://docs.github.com/en/actions/writing-workflows/quickstart

---

### Learn

- Workflows
- Running tests automatically
- Building Docker images

---

### Objective

Create a workflow that:

- Installs dependencies
- Runs tests
- Builds Docker image

Trigger:

- Push
- Pull Request

---

## 2.4 AWS Fundamentals (FOCUSED)

### Resources

AWS Skill Builder  
https://explore.skillbuilder.aws/

Focus only on:

- IAM
- EC2
- S3
- VPC Basics
- Security Groups
- CloudWatch
- Billing

---

### Learn

- What is EC2
- How servers work
- Basic networking
- Cost awareness (IMPORTANT)

---

### Setup

- Enable MFA
- Create IAM user
- Set billing alerts

> Never use root account daily

---

## 2.5 AWS EC2 Deployment

### Learn

#### EC2

- Launch instance
- Security groups
- SSH connection

#### Deployment

- Install Docker
- Run container

#### Nginx

- Reverse proxy
- Port forwarding

#### S3

- Store model backups

#### CloudWatch

- Logs and monitoring basics

---

### Objective

Deploy your application so it is:

- Running on EC2
- Accessible via public IP
- Logging requests

---

## Flagship Project Progress

### Customer Churn Prediction System

Enhance your existing project by adding:

---

### API Layer

- /predict → returns prediction
- /health → system status
- /explain → SHAP explanation

---

### Database

- Store prediction logs
- Store request/response history

---

### Containerization

- Dockerize the application
- Add Docker Compose

---

### CI

- Add GitHub Actions workflow

---

### Deployment

- Deploy to AWS EC2
- Configure Nginx
- Store model backup in S3

---

## Expected Outcome

By the end of this phase, you will have:

- A running ML API
- A containerized system
- Automated testing setup
- A deployed cloud application

---

## Deliverables

Application

- Production-ready FastAPI application

Infrastructure

- Dockerized setup
- Docker Compose configuration
- GitHub Actions workflow
- AWS EC2 deployment
- Nginx configuration
- CloudWatch logging

---

## Skills Gained

- FastAPI
- API Design
- SQLAlchemy Basics
- Docker
- Docker Compose
- GitHub Actions
- AWS EC2
- S3
- CloudWatch
- Nginx
- Deployment

---

## Completion Criteria

Move to **Phase 03** only after:

- Built working API
- Connected API to database
- Dockerized application
- Created Docker Compose setup
- Created GitHub Actions workflow
- Deployed application on EC2
- Verified public access
- Basic logging working

---

## Ready for Next Phase

In Phase 03, you will learn Deep Learning and PyTorch to build neural networks and move beyond classical ML.
