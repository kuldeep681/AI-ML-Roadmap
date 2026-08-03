# Phase 02 - FastAPI, Docker, CI/CD & AWS Deployment

## Goal

Deploy your Machine Learning model as a production-ready cloud application using FastAPI, Docker, GitHub Actions and AWS.

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

Training a Machine Learning model is only part of the job.

A Machine Learning Engineer should also know how to:

- Build APIs
- Containerize applications
- Automate testing
- Deploy applications to the cloud
- Monitor deployed services

This phase transforms your Customer Churn Model into a real-world application.

---

## Prerequisites

- Complete Phase 01
- Customer Churn Model trained
- Model saved using joblib

---

## 2.1 FastAPI

### Resources

1. FastAPI Official Tutorial

   https://fastapi.tiangolo.com/tutorial/

   Complete from:
   - First Steps
   - Path Parameters
   - Query Parameters
   - Request Body
   - Response Models
   - Dependencies
   - Security

2. FastAPI SQL Databases

   https://fastapi.tiangolo.com/tutorial/sql-databases/

3. FastAPI Docker Deployment

   https://fastapi.tiangolo.com/deployment/docker/

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
- Request Validation
- Response Validation

#### Dependency Injection

- Depends()
- Reusable Dependencies

#### Database

- SQLAlchemy Basics
- Database Sessions
- CRUD Operations

#### Authentication

- JWT Basics

#### Error Handling

- HTTPException
- Custom Exception Handlers

#### Documentation

- Swagger UI
- OpenAPI Docs

---

## 2.2 Docker & Docker Compose

### Resources

1. Docker Get Started

   https://docs.docker.com/get-started/

2. Docker Compose Getting Started

   https://docs.docker.com/compose/gettingstarted/

---

### Learn

Docker

- Images
- Containers
- Dockerfile
- Volumes
- Networks

Docker Compose

- docker-compose.yml
- Multiple Services
- Environment Variables

---

### Build

Your application should run using

```bash
docker compose up --build
```

Architecture

```text
FastAPI
    │
    ▼
PostgreSQL
```

---

## 2.3 GitHub Actions

### Resources

1. GitHub Actions Quickstart

https://docs.github.com/en/actions/writing-workflows/quickstart

2. GitHub Actions for Python

https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python

---

### Learn

- GitHub Workflows
- Python CI
- Running pytest
- Building Docker Images

---

### Build

Create one workflow that automatically:

- Installs dependencies
- Runs pytest
- Builds Docker Image

Trigger

- Every Push
- Every Pull Request

---

## 2.4 AWS Fundamentals

### Resources

1. AWS Cloud Practitioner Essentials

https://explore.skillbuilder.aws/learn/course/134/aws-cloud-practitioner-essentials

Complete only:

- IAM
- EC2
- S3
- VPC Basics
- Security Groups
- CloudWatch
- Billing & Pricing

---

### AWS Account Setup

Before creating any AWS resources:

- Enable Root MFA
- Create IAM Admin User
- Create Billing Alert
- Create Budget Alert

> Never use the Root Account for daily work.

---

## 2.5 AWS EC2 Deployment

### Resources

1. EC2 Getting Started

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html

2. EC2 Security Groups

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html

3. Connect to EC2 using SSH

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-to-linux-instance.html

4. Nginx Reverse Proxy

https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/

5. Amazon S3 Getting Started

https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html

---

### Learn

EC2

- Launch Instance
- Security Groups
- Key Pair
- Elastic IP (Basics)

Nginx

- Reverse Proxy
- Port Forwarding

S3

- Upload Files
- Private Buckets

CloudWatch

- Logs
- Monitoring Basics

---

### Deployment Steps

1. Launch Ubuntu EC2 Instance
2. Allow SSH only from your IP
3. Allow HTTP (Port 80)
4. Install Docker
5. Install Docker Compose
6. Clone Repository
7. Configure .env
8. Run Docker Compose
9. Install Nginx
10. Configure Reverse Proxy
11. Upload Model Backup to S3
12. Verify Application
13. Monitor Logs using CloudWatch

---

## Flagship Project Progress

Continue developing:

### Production Customer Churn Prediction System

### Add

API

- /predict
- /health
- /explain

Database

- PostgreSQL
- Prediction Logs

Containerization

- Docker
- Docker Compose

CI/CD

- GitHub Actions

Deployment

- AWS EC2
- Nginx
- S3 Backup

---

## Deliverables

Application

- Production FastAPI API

Infrastructure

- Dockerized Application
- Docker Compose Setup
- GitHub Actions Workflow
- AWS EC2 Deployment
- Nginx Reverse Proxy
- CloudWatch Logging

Repository

```text
customer-churn-prediction/
│
├── app/
├── models/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
├── nginx/
├── requirements.txt
└── README.md
```

---

## Skills Gained

- FastAPI
- Pydantic
- SQLAlchemy Basics
- JWT Basics
- Docker
- Docker Compose
- GitHub Actions
- AWS IAM
- EC2
- S3
- CloudWatch
- Nginx
- API Deployment

---

## Completion Criteria

Move to **Phase 03** only after:

- Completed FastAPI Tutorial.
- Built all required APIs.
- Connected FastAPI with PostgreSQL.
- Dockerized the application.
- Created Docker Compose setup.
- Created GitHub Actions workflow.
- Deployed the application to AWS EC2.
- Configured Nginx Reverse Proxy.
- Stored backups in S3.
- Verified application is publicly accessible.

---

## Ready for Next Phase

In **Phase 03**, you'll learn Deep Learning fundamentals and PyTorch to build neural network models before moving into LLMs.
