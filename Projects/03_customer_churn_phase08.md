# 🚀 Project 03 - Customer Churn (Phase 08 - Kubernetes Deployment)

---

## 🎯 Objective

Deploy and scale ML system using Kubernetes.

---

## 🧠 System Architecture

Docker Image → Kubernetes Deployment → Pods → Service → API Access

---

## 📁 Kubernetes Structure

```text
k8s/
├── deployment.yaml
├── service.yaml
├── configmap.yaml
├── secret.yaml
```

---

## 🧩 Build Steps

1. Create cluster (kind)
2. Build Docker image
3. Load image into kind
4. Create deployment.yaml
5. Create service.yaml
6. Add ConfigMap
7. Add Secret
8. Deploy application
9. Scale replicas
10. Test API

---

## ⚙️ Commands

Create cluster:
kind create cluster

Build image:
docker build -t churn-api:v1 .

Load image:
kind load docker-image churn-api:v1

Deploy:
kubectl apply -f deployment.yaml  
kubectl apply -f service.yaml

Scale:
kubectl scale deployment churn-api --replicas=3

---

## 🌐 API Access

http://<node-ip>:<nodeport>/docs

---

## 🚫 Constraints

- Use lightweight cluster (kind)
- Avoid complex networking

---

## 🎯 Final Outcome

- Scalable API
- Multiple replicas
- Load-balanced system
