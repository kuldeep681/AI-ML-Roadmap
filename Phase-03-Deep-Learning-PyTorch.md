# Phase 03 - Deep Learning & PyTorch

## Goal

Understand Neural Networks and build a real Deep Learning application using PyTorch.

**Estimated Duration:** 5–7 Weeks

---

## Why this Phase?

Classical Machine Learning works well for structured data.

Deep Learning is used for:

- Computer Vision
- Natural Language Processing
- Recommendation Systems
- Foundations of LLMs

This phase builds the base required for Transformers, LLMs and RAG.

---

## Prerequisites

- Complete Phase 02
- Strong understanding of ML basics
- Comfortable with Python

---

## 3.1 Deep Learning Theory (Focused Learning)

### Resources

1. Deep Learning Specialization - DeepLearning.AI (Coursera)

https://www.coursera.org/specializations/deep-learning

👉 Complete only with focus (do NOT over-invest time)

- Course 1 - Neural Networks and Deep Learning
- Course 2 - Improving Deep Neural Networks
- Course 4 - Convolutional Neural Networks
- Course 5 - Sequence Models

⚠️ Course 3 is optional (read summaries only)

> Audit for free. Do NOT aim for perfection — aim for understanding.

---

### Learn (What Actually Matters)

#### Core Concepts

- What is a Neural Network
- Forward Propagation
- Backpropagation (intuition > math)
- Activation Functions (ReLU, Sigmoid)

#### Training

- Loss Functions
- Gradient Descent
- Learning Rate
- Overfitting vs Underfitting

#### Regularization

- Dropout
- Early Stopping

#### CNN (High Value)

- Convolutions
- Pooling
- Image features

#### Sequence Models (Basic Understanding)

- RNN (idea only)
- LSTM (intuition only)

#### Transformers (ONLY INTRO)

- Attention intuition
- Why Transformers replaced RNNs

---

## 3.2 PyTorch (Core Only)

### Resources

1. PyTorch Official Basics

https://pytorch.org/tutorials/beginner/basics/intro.html

---

### Learn

#### Essentials Only

- Tensors
- Dataset & DataLoader
- nn.Module
- Layers (Linear, Conv)
- Loss Functions
- Optimizers (Adam)

#### Training Loop (VERY IMPORTANT)

- Forward pass
- Loss calculation
- Backward pass
- Optimizer step

#### Model Usage

- Save model
- Load model
- Inference

---

## Practice

👉 Complete PyTorch tutorial examples  
👉 Do NOT spend time building multiple mini projects

---

## Flagship Project (MANDATORY)

Build ONE complete Deep Learning system.

### Choose ONE (Recommended Order)

1. ✅ Sentiment Analysis API (RECOMMENDED)
2. Image Classification API

---

## Project Goal

Build a working system:

- Train model
- Serve predictions via API
- Run using Docker

---

## Tech Stack

- PyTorch
- FastAPI
- Docker

---

## Build Order

1. Dataset Preparation
2. DataLoader
3. Model Creation (simple NN or LSTM/CNN)
4. Training Loop
5. Validation
6. Evaluation
7. Save Model
8. Build FastAPI API (/predict)
9. Dockerize Application

---

## ⚠️ Cost Policy

- Train on CPU (local machine)
- Use small datasets
- DO NOT use paid GPU services

---

## Deliverables

Project (Implemented in separate repo)

- Sentiment Analysis API  
  OR
- Image Classification API

---

## Repository (Project Repo Structure)

```text
deep-learning-project/
│
├── app/
├── dataset/
├── models/
├── notebooks/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```
