# Phase 08 - Kubernetes

## Goal

Understand Kubernetes fundamentals and deploy your Dockerized ML application locally using Kubernetes.

**Estimated Duration:** 2–3 Weeks

---

## Why this Phase?

Docker helps you package applications.

Kubernetes helps you:

- Deploy containers
- Scale applications
- Manage failures
- Handle production environments

👉 You don’t need deep expertise — just strong fundamentals.

---

## Prerequisites

- Complete Phase 07
- Comfortable with Docker
- Understand Docker Compose

---

## ⚠️ Cost Policy

- Use LOCAL Kubernetes only
- Do NOT use paid cloud Kubernetes
- No AWS EKS / GCP GKE required

---

## 8.1 Kubernetes Fundamentals

### Resources

1. Kubernetes Basics  
   https://kubernetes.io/docs/tutorials/kubernetes-basics/

2. Deployments  
   https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

3. Services  
   https://kubernetes.io/docs/concepts/services-networking/service/

---

### Learn (Focused)

#### Core Architecture

- Control Plane (basic idea)
- Worker Nodes
- kubectl usage

---

#### Pods

- What is a Pod
- Single container vs multi-container (concept only)

---

#### Deployments

- Deployment
- ReplicaSet
- Scaling
- Rolling updates

---

#### Services

- ClusterIP
- NodePort (IMPORTANT)
- LoadBalancer (concept only)

---

#### Config Management

- ConfigMaps
- Secrets

---

#### Debugging

- kubectl logs
- kubectl describe
- kubectl get

---

## 8.2 kind (Kubernetes in Docker)

### Resource

https://kind.sigs.k8s.io/docs/user/quick-start/

---

### Learn

- Install kind
- Create cluster
- Connect kubectl
- Deploy apps
- Delete cluster

---

## Practice (MANDATORY)

Before your ML project:

### Step 1: Basic Deployment

- Deploy Nginx
- Expose via NodePort
- Access in browser

---

### Step 2: Scaling

- Increase replicas
- Observe behavior

---

### Step 3: Debugging

- View logs
- Describe pod
- Delete pod (auto recreate)

---

## Apply to Flagship Project

### Customer Churn Prediction System

Deploy your existing Dockerized ML API using Kubernetes.

---

## Project Goal

Run your ML system inside Kubernetes with:

- Deployment
- Service
- Config management
- Scaling

---

## What You Will Add

Kubernetes Files

- deployment.yaml
- service.yaml
- configmap.yaml
- secret.yaml

---

## Build Order (VERY IMPORTANT)

1. Create kind cluster
2. Build Docker image
3. Load image into kind
4. Create Deployment
5. Create Service (NodePort)
6. Access API locally
7. Add ConfigMap
8. Add Secret
9. Scale replicas
10. Test failure recovery (delete pod)

---

## Deployment Flow

```text
Docker Image
     │
     ▼
Kubernetes Deployment
     │
     ▼
Pods (Replicas)
     │
     ▼
Service (NodePort)
     │
     ▼
Local Access (Browser / Postman)
```
