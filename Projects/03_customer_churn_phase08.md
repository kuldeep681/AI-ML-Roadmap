---

# 📄 **3. 03_customer_churn_phase08.md (KUBERNETES DEPLOYMENT)**

```markdown
# Project 03 - Customer Churn (Kubernetes Deployment)

## Phase Mapping

- Phase 08
- Extends: Project 02 (MLOps)

---

## Goal

Deploy your ML system using Kubernetes.

---

## What You Already Have

From Phase 07:

- FastAPI application
- Dockerized system
- ML pipeline

---

## What You Will Build NOW

### Kubernetes Deployment

- Deploy API using Kubernetes
- Scale application
- Manage configuration

---

## Tech Stack

- Kubernetes (kind)
- kubectl
- Docker

---

## Build Steps

1. Create kind cluster
2. Build Docker image
3. Load image into kind
4. Create deployment.yaml
5. Create service.yaml (NodePort)
6. Add ConfigMap
7. Add Secret
8. Deploy application
9. Scale replicas
10. Test API

---

## Architecture

```text
Docker → Kubernetes Deployment → Pods → Service → API Access
```
