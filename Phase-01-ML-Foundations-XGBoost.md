# Phase 01 - Machine Learning Foundations + XGBoost

## Goal

Build a strong foundation in Classical Machine Learning and develop your **first end-to-end Machine Learning project**.

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

Before learning Deep Learning, LLMs, or RAG, you must understand:

- How Machine Learning models work
- How to evaluate models
- How to build reliable prediction systems

> This phase focuses on both **learning concepts** and **applying them in a real project**

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

   Complete all three courses:
   - Course 1 - Supervised Machine Learning
   - Course 2 - Advanced Learning Algorithms
   - Course 3 - Unsupervised Learning, Recommenders & Reinforcement Learning

> Focus on intuition and application. Avoid going too deep into math.

---

### Learn

#### Supervised Learning

- Linear Regression
- Logistic Regression

#### Model Improvement

- Overfitting
- Underfitting
- Bias vs Variance
- Regularization

#### Tree-Based Models

- Decision Trees
- Random Forests

#### Boosting

- Gradient Boosting intuition

#### Unsupervised Learning

- Clustering

#### Model Evaluation (VERY IMPORTANT)

- Train / Validation / Test Split
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 1.2 Practical Machine Learning + XGBoost

### Resources

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
- Key Parameters:
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

## Practice Approach

- Use Kaggle exercises to understand concepts
- Apply everything directly in your main project
- Avoid creating multiple small projects

---

## Flagship Project

### Customer Churn Prediction System

---

## Objective

Build a Machine Learning system that:

- Takes customer data as input
- Predicts whether a customer will churn
- Provides explanation for predictions

---

## Build Steps

1. Perform Exploratory Data Analysis (EDA)
2. Clean the dataset
3. Perform feature engineering
4. Train Logistic Regression (baseline)
5. Train Random Forest (baseline)
6. Train XGBoost (final model)
7. Perform Cross Validation
8. Tune hyperparameters
9. Generate SHAP explanations
10. Save the trained model

---

## Expected Outcome

By the end of this phase, you should have:

- A trained and tuned XGBoost model
- Model evaluation metrics
- SHAP-based explanations
- A reusable prediction workflow (conceptual understanding)

> API development and deployment will be done in the next phase

---

## Deliverables

Project

- Customer Churn Prediction System (Model Training Completed)

Artifacts

- Trained XGBoost Model
- SHAP Visualizations
- Model Metrics
- Saved Model File (joblib)

---

## Skills Gained

- Classical Machine Learning
- Supervised Learning
- Tree-Based Models
- XGBoost
- Feature Engineering
- Model Evaluation
- SHAP Explainability
- Model Persistence

---

## Completion Criteria

Move to **Phase 02** only after:

- Completed Machine Learning Specialization
- Completed Kaggle courses
- Trained XGBoost model
- Performed Hyperparameter Tuning
- Generated SHAP explanations
- Saved the model
- Clearly understand how prediction flow works

---

## Ready for Next Phase

In Phase 02, you will:

- Convert this model into an API
- Containerize using Docker
- Deploy using AWS EC2 or other platforms
