# Phase 01 - Machine Learning Foundations + XGBoost (SYSTEM-DRIVEN)

## Goal

Build a strong foundation in Classical Machine Learning and develop your **first end-to-end Machine Learning system** (not just a model).

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

Before Deep Learning, LLMs, or RAG, you must understand:

- How models learn
- How to evaluate models
- How to build reliable prediction systems

> Most importantly:
> **How to turn a model into something usable (system thinking)**

---

## Prerequisites

- Complete Phase 00
- Comfortable with Python
- Basic SQL knowledge

---

## 1.1 Machine Learning Theory (FOCUSED)

### Resources

1. Machine Learning Specialization - DeepLearning.AI (Coursera)

   https://www.coursera.org/specializations/machine-learning-introduction

> Focus on intuition and application. Avoid going too deep into math.

---

### Learn

#### Supervised Learning

- Linear Regression
- Logistic Regression

#### Model Improvement

- Overfitting vs Underfitting
- Bias vs Variance
- Regularization

#### Tree-Based Models

- Decision Trees
- Random Forests

#### Boosting (IMPORTANT)

- Gradient Boosting intuition

#### Evaluation (CRITICAL)

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 1.2 Practical Machine Learning + XGBoost (CORE)

### Resources

1. Kaggle - Intro to Machine Learning
2. Kaggle - Intermediate Machine Learning
3. Kaggle - Feature Engineering
4. Kaggle - Machine Learning Explainability

> Do not just complete — apply concepts directly in your project.

---

### Learn

#### Data Preparation

- Handling Missing Values
- Encoding Categorical Variables
- Feature Engineering
- Pipelines

#### Model Training

- Scikit-learn Workflow
- Cross Validation
- RandomizedSearchCV

#### XGBoost (MAIN MODEL)

- XGBClassifier
- Key Parameters:
  - n_estimators
  - learning_rate
  - max_depth
  - subsample
  - colsample_bytree
  - scale_pos_weight

---

### Explainability

- SHAP Feature Importance
- Global Explanation
- Local Explanation

---

## Mini Practice (MODIFIED)

Use Kaggle exercises only to understand concepts.

> Immediately apply everything in your main project.  
> Do NOT create separate mini projects.

---

## Flagship Project

# 🔥 Customer Churn Prediction SYSTEM

---

## Objective

Build a system that:

- Takes customer data
- Predicts churn
- Explains predictions

---

## Build Order

### Phase A: Data Preparation

1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering

---

### Phase B: Modeling

4. Logistic Regression (Baseline)
5. Random Forest (Baseline)
6. XGBoost (Final Model)

---

### Phase C: Evaluation

7. Cross Validation
8. Hyperparameter Tuning

---

### Phase D: Explainability

9. SHAP Explainability (MANDATORY)

---

### Phase E: System Preparation (IMPORTANT)

10. Create prediction pipeline:

```python
def predict_customer(data):
    # preprocess input
    # load trained model
    # return prediction
```

> This function will be used in Phase 02 for API development.

---

## Deliverables

### Project

- Customer Churn Prediction System

---

### Artifacts

- Trained XGBoost Model
- SHAP Visualizations
- Evaluation Metrics
- joblib Model File

---

### Repository Structure

```text
customer-churn-prediction/
│
├── data/
├── notebooks/
├── models/
├── src/
│   ├── train.py
│   ├── predict.py        # REQUIRED
│   └── preprocess.py
│
├── requirements.txt
└── README.md
```

---

## Skills Gained

- Classical Machine Learning
- Supervised Learning
- Tree-Based Models
- XGBoost (Industry Level)
- Feature Engineering
- Model Evaluation
- SHAP Explainability
- Model Persistence using joblib
- Basic ML System Design

---

## Completion Criteria

Move to **Phase 02** only after:

- Completed ML Specialization (focused understanding)
- Completed Kaggle courses (with application)
- Trained XGBoost model
- Performed Hyperparameter Tuning
- Generated SHAP explanations
- Saved model using joblib
- Created `predict.py` pipeline

---

## Ready for Next Phase

In Phase 02, you will:

- Convert this model into an API (FastAPI)
- Containerize using Docker
- Deploy using AWS EC2 or Render

> Since your prediction pipeline is ready, Phase 02 will be smooth and practical.
