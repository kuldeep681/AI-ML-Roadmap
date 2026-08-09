# 🚀 Project 01 - Customer Churn Prediction (Phase 01 - Core ML System)

---

## 🎯 Objective

Build a production-ready Machine Learning pipeline for customer churn prediction.

---

## ⚠️ Core Requirements

- Modular code
- No hardcoded paths
- Reusable in later phases
- Clean separation of concerns

---

## 🧠 System Architecture

Data → Preprocessing → Feature Engineering → Training → Evaluation → Model Saving

---

## 📁 Project Structure

project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── config/
│   │   └── config_loader.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── save_model.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   └── utils/
│       └── logger.py
│
├── models/
│   └── churn_model.pkl
│
├── config.yaml
├── requirements.txt
└── run.py

---

## ⚙️ Configuration

Use config.yaml for:
- dataset path
- model parameters
- output paths

---

## 🧩 Build Steps

1. Load dataset  
2. Clean data  
3. Perform EDA  
4. Feature engineering  
5. Train model (XGBoost preferred)  
6. Evaluate model  
7. Save model using joblib  

---

## 📦 Output

- churn_model.pkl  
- evaluation metrics  

---

## 🚫 Constraints

- No API  
- No Docker  
- No deployment  

---

## 🔮 Future Compatibility

Will be extended with:
- MLOps (Phase 07)
- Kubernetes (Phase 08)

---

## 🎯 Final Outcome

A clean ML system ready for production upgrades
