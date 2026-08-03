# Phase 08 - Kubernetes

## Goal

Learn the fundamentals of Kubernetes and deploy your Dockerized Machine Learning application locally using Kubernetes.

**Estimated Duration:** 2–3 Weeks

---

## Why this Phase?

Docker allows you to package applications.

Kubernetes allows you to deploy, scale and manage those Docker containers in production.

You don't need to become a Kubernetes expert for entry-level AI/ML roles, but understanding the fundamentals is a valuable skill.

This phase focuses only on the concepts and practical knowledge commonly expected in interviews.

---

## Prerequisites

- Complete Phase 07
- Comfortable with Docker
- Comfortable with Docker Compose

---

## 8.1 Kubernetes Fundamentals

### Resources

1. Kubernetes Basics

https://kubernetes.io/docs/tutorials/kubernetes-basics/

2. Kubernetes Deployments

https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

3. Kubernetes Services

https://kubernetes.io/docs/concepts/services-networking/service/

---

### Learn

Core Concepts

- Kubernetes Architecture
- Control Plane
- Worker Nodes
- kubectl

Pods

- Pod
- Multi-container Pod (Concept)

Deployments

- Deployment
- ReplicaSet
- Scaling
- Rolling Updates

Services

- ClusterIP
- NodePort
- LoadBalancer (Concept)

Configuration

- ConfigMaps
- Secrets

Monitoring

- Logs
- Describe Resources
- Debugging

---

## 8.2 kind (Kubernetes in Docker)

### Resource

1. kind Quick Start

https://kind.sigs.k8s.io/docs/user/quick-start/

---

### Learn

- Install kind
- Create Local Cluster
- kubectl with kind
- Deploy Applications
- Delete Cluster

---

## Practice

Deploy simple applications before deploying your ML project.

Practice

- Deploy Nginx
- Scale Deployment
- Expose Service
- View Logs
- Delete Resources

---

## Apply to Flagship Project

### Production Customer Churn Prediction System

Deploy the existing Dockerized application using Kubernetes.

---

### Add

Kubernetes

- Deployment
- Service
- ConfigMap
- Secret

Local Cluster

- kind

---

### Build Order

1. Create kind Cluster
2. Build Docker Image
3. Create Deployment
4. Create Service
5. Create ConfigMap
6. Create Secret
7. Deploy Application
8. Verify API
9. Scale Deployment
10. Inspect Logs

---

## Deliverables

Project

- Customer Churn Prediction System (Kubernetes Deployment)

Repository

```text
customer-churn-prediction/
│
├── app/
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Artifacts

- Kubernetes Deployment
- Kubernetes Service
- ConfigMap
- Secret
- Local kind Cluster

---

## Skills Gained

- Kubernetes Basics
- Pods
- Deployments
- Services
- ConfigMaps
- Secrets
- Scaling
- kubectl
- kind
- Local Kubernetes Deployment

---

## Completion Criteria

Move to **Phase 09** only after:

- Completed the Kubernetes Basics tutorial.
- Created a local kind cluster.
- Deployed the application using Kubernetes.
- Exposed the application using a Service.
- Used ConfigMaps and Secrets.
- Scaled the deployment.
- Verified the application is running successfully.

---

## Ready for Next Phase

In **Phase 09**, you'll learn Parameter-Efficient Fine-Tuning (PEFT), LoRA and QLoRA, understand when fine-tuning is appropriate, and compare fine-tuning with prompting and RAG for real-world AI applications.
