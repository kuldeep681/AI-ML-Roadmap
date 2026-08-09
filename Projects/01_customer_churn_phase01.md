# 🚀 Project 01 - Customer Churn Prediction (ML + API + EC2 Deployment)

---

## 🎯 Objective

Build a production-ready Machine Learning system that:

- Trains a churn prediction model  
- Serves predictions via API  
- Deploys on AWS EC2 (Free Tier)  
- Works locally and in production with minimal changes  

---

## ⚠️ Core Requirements

- Modular code  
- No hardcoded paths  
- Reusable in later phases  
- Clean separation of concerns  
- Same code must run locally and on EC2  

---

## 🧠 System Architecture

Data → Preprocessing → Feature Engineering → Training → Evaluation → Model Saving → API → Deployment

---

## 📁 Project Structure
```text
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
│   │   ├── save_model.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   └── evaluate.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
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
```
---

## ⚙️ Configuration

Use config.yaml for:

- dataset path  
- model parameters  
- output paths  
- API settings  

---

## 🧩 Build Steps

### PART 1 — ML SYSTEM

1. Load dataset  
2. Clean data  
3. Perform EDA  
4. Feature engineering  
5. Train model (XGBoost preferred)  
6. Evaluate model  
7. Save model using joblib  

---

### PART 2 — API SYSTEM

- Build FastAPI app  
- Load trained model  
- Create /predict endpoint  

Input:
JSON with customer features  

Output:
prediction (0 or 1) and probability score  

---

### PART 3 — LOCAL TESTING

Run:
uvicorn src.api.main:app --reload  

Test:
http://127.0.0.1:8000/docs  

---

## ☁️ PART 4 — EC2 DEPLOYMENT (FREE TIER)

### Step 1 — Launch EC2

- Ubuntu instance  
- t2.micro or t3.micro  
- Open ports: 22 and 8000  

---

### Step 2 — Connect

ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>  

---

### Step 3 — Setup Environment

sudo apt update  
sudo apt install python3-pip -y  
pip3 install virtualenv  

virtualenv venv  
source venv/bin/activate  

---

### Step 4 — Transfer Code

git clone <your-repo>  

OR  

scp -i key.pem -r project ubuntu@<IP>:/home/ubuntu/  

---

### Step 5 — Install Dependencies

pip install -r requirements.txt  

---

### Step 6 — Run API

uvicorn src.api.main:app --host 0.0.0.0 --port 8000  

---

### Step 7 — Access API

http://<EC2_PUBLIC_IP>:8000/docs  

---

## 📦 Output

- Trained model file  
- Evaluation metrics  
- Running API (local + EC2)  

---

## 🚫 Constraints

- No Docker (for now)  
- Keep system lightweight  
- Free tier compatible  

---

## 🔮 Future Compatibility

Will be extended with:

- MLOps (Phase 07)  
- Kubernetes (Phase 08)  

---

## 🎯 Final Outcome

A complete ML system that:

- Trains churn model  
- Serves predictions via API  
- Runs locally  
- Deploys on EC2 without code changes  
