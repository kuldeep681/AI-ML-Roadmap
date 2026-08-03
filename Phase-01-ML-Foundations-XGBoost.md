# Phase 01 - Machine Learning Foundations + XGBoost

## Goal

Build a strong foundation in Classical Machine Learning and train your first production-ready Machine Learning model.

**Estimated Duration:** 8–10 Weeks

---

## Why this Phase?

Before learning Deep Learning, LLMs or RAG, you need to understand how Machine Learning models work, how to evaluate them, and how to build reliable prediction systems.

---

## Prerequisites

- Complete Phase 00
- Comfortable with Python
- Basic SQL knowledge

---

## 1.1 Machine Learning Theory

### Resources

1. Machine Learning Specialization - DeepLearning.AI (Coursera)

   https://www.coursera.org/specializations/machine-learning-introduction

   Complete all three courses.
   - Course 1 - Supervised Machine Learning
   - Course 2 - Advanced Learning Algorithms
   - Course 3 - Unsupervised Learning, Recommenders & Reinforcement Learning

> Audit the course for free. Purchase the certificate only if you want it.

### Learn

#### Supervised Learning

- Linear Regression
- Logistic Regression
- Cost Function
- Gradient Descent

#### Model Improvement

- Regularization
- Bias vs Variance
- Overfitting
- Underfitting

#### Tree Based Models

- Decision Trees
- Random Forests

#### Unsupervised Learning

- Clustering

#### Model Evaluation

- Train / Validation / Test Split
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 1.2 Practical Machine Learning + XGBoost

### Resources

Complete these in order.

1. Kaggle - Intro to Machine Learning

   https://www.kaggle.com/learn/intro-to-machine-learning

2. Kaggle - Intermediate Machine Learning

   https://www.kaggle.com/learn/intermediate-machine-learning

3. Kaggle - Feature Engineering

   https://www.kaggle.com/learn/feature-engineering

4. Kaggle - Machine Learning Explainability

   https://www.kaggle.com/learn/machine-learning-explainability

---

### Learn

#### Data Preparation

- Handling Missing Values
- Categorical Variables
- Feature Engineering
- Pipelines

#### Model Training

- Scikit-learn Workflow
- Cross Validation
- RandomizedSearchCV

#### XGBoost

- XGBClassifier
- XGBRegressor
- Gradient Boosting Intuition
- n_estimators
- learning_rate
- max_depth
- subsample
- colsample_bytree
- scale_pos_weight

#### Evaluation

- Precision
- Recall
- F1 Score
- ROC-AUC

#### Explainability

- SHAP Feature Importance
- Global Explanation
- Local Explanation

---

## Mini Practice

Before starting the flagship project, practice using the concepts learned from Kaggle courses by experimenting with the datasets provided in those courses.

> No separate mini projects are required in this phase. Focus on completing the Kaggle exercises and notebooks.

---

## Flagship Project

### Production Customer Churn Prediction System

### Tech Stack

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

> FastAPI, PostgreSQL, Docker, AWS, MLflow and DVC will be integrated in later phases.

### Build Order

1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering
4. Logistic Regression Baseline
5. Random Forest Baseline
6. XGBoost Final Model
7. Cross Validation
8. Hyperparameter Tuning
9. SHAP Explainability
10. Save Model using joblib

> API Development, Dockerization and Deployment will be completed in Phase 02.

---

## Deliverables

Project

- Production Customer Churn Prediction System (Model Training Completed)

Artifacts

- Trained XGBoost Model
- SHAP Visualizations
- Model Metrics
- joblib Model File

Repository

```text
customer-churn-prediction/
│
├── data/
├── notebooks/
├── models/
├── src/
├── requirements.txt
└── README.md
```

---

## Skills Gained

- Classical Machine Learning
- Supervised Learning
- Decision Trees
- Random Forests
- XGBoost
- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- Model Evaluation
- SHAP Explainability
- Model Persistence using joblib

---

## Completion Criteria

Move to **Phase 02** only after:

- Completed the Machine Learning Specialization.
- Completed all four Kaggle courses.
- Trained an XGBoost model.
- Performed Hyperparameter Tuning.
- Generated SHAP explanations.
- Saved the model using joblib.
- Completed the Customer Churn Prediction model training.

---

## Ready for Next Phase

In Phase 02, you'll convert your trained Machine Learning model into a production-ready application using FastAPI, Docker, PostgreSQL, GitHub Actions and AWS.
