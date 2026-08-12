# 🔹 Exercise 5: Random Forest

(From: random-forests notebook)

---

## 📖 Concept

Problem with Decision Tree:

- High variance
- Easily overfits

👉 Solution: **Random Forest**

---

## ⚙️ What You Actually Did (REAL FLOW)

---

### Step 1 — Train Decision Tree (Baseline)

```
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
dt_preds = dt_model.predict(val_X)
```

---

### Step 2 — Evaluate Decision Tree

```
from sklearn.metrics import mean_absolute_error

dt_mae = mean_absolute_error(val_y, dt_preds)
```

---

### 🧠 Why This Step?

👉 Always create a **baseline model**
So you can compare improvements

---

### Step 3 — Train Random Forest

```
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
```

---

### 🧠 What Happens Internally

- Many trees are created
- Each tree sees slightly different data
- Predictions are averaged

👉 Reduces overfitting

---

### Step 4 — Predict & Evaluate

```
rf_preds = rf_model.predict(val_X)

rf_mae = mean_absolute_error(val_y, rf_preds)
```

---

## 🧠 What You Observed

👉 Random Forest MAE < Decision Tree MAE

👉 Better performance ✅

---

## ⚠️ Important Notes

- Works well with default settings
- Slower than Decision Tree
- Less interpretable

---

## ❌ Common Mistakes

- Not comparing with baseline
- Using Random Forest blindly

---

## 🎤 Interview Answer

👉 “Why Random Forest works better?”

Answer:

> It reduces overfitting by combining predictions from multiple decision trees, making the model more stable.

---

# 🔹 Exercise 6: Machine Learning Competition (Full Pipeline)

(From: machine-learning-competitions notebook)

---

## 📖 Concept

👉 This is **real-world ML workflow simulation**

You applied EVERYTHING together.

---

## ⚙️ What You Actually Did (REAL FLOW)

---

### Step 1 — Load Training Data

```
import pandas as pd

train_data = pd.read_csv("train.csv")
```

---

### Step 2 — Handle Missing Values

```
train_data = train_data.dropna(axis=0)
```

---

### ⚠️ Important

👉 Same approach used in earlier exercise
👉 Simple but not always best

---

### Step 3 — Define Target

```
y = train_data.SalePrice
```

---

### Step 4 — Select Features

```
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = train_data[features]
```

---

### Step 5 — Train Model

```
model = RandomForestRegressor(random_state=1)
model.fit(X, y)
```

---

### Step 6 — Load Test Data

```
test_data = pd.read_csv("test.csv")
```

---

### Step 7 — Prepare Test Features

```
test_X = test_data[features]
```

---

### Step 8 — Make Predictions

```
preds = model.predict(test_X)
```

---

### Step 9 — Create Submission File

```
output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': preds
})

output.to_csv('submission.csv', index=False)
```

---

## 🧠 What This Teaches

👉 Real ML is NOT just modeling

It includes:

- Data loading
- Cleaning
- Feature selection
- Training
- Prediction
- Output formatting

---

## ⚠️ Common Mistakes

- Mismatch in features between train & test
- Forgetting preprocessing on test data
- Wrong column names

---

## 🎤 Interview Answer

👉 “How do you deploy ML predictions?”

Answer:

> After training the model, I apply the same preprocessing on new data, generate predictions, and format them for downstream use like APIs or files.

---

# 🔥 FINAL REVISION (Exercise 5 + 6)

## 🧠 Core Flow

```
Train Baseline Model
    ↓
Train Better Model (Random Forest)
    ↓
Compare Performance
    ↓
Use Best Model
    ↓
Apply on New Data
    ↓
Generate Output
```

---

## 🎯 Full ML Pipeline

```
Load Data
    ↓
Clean Data
    ↓
Feature Selection
    ↓
Train Model
    ↓
Validate Model
    ↓
Improve Model
    ↓
Predict on New Data
    ↓
Save Results
```

---

## 🚀 Final Insight

👉 ML is NOT about algorithms

👉 ML is about:

- Correct process
- Clean data
- Proper validation

---

# 🏁 FINAL COURSE SUMMARY (WITH EXERCISES)

## 🧠 You Now Know

- How to explore data
- How to build models
- How to validate models
- How to avoid overfitting
- How to improve models
- How to create full pipeline

---

## 🎯 One-Line Memory Trick

👉 ML =

```
Data → Model → Validate → Improve → Deploy
```

---

## 💡 Ultimate Interview Insight

👉 A beginner says:

“I trained a model”

👉 A good candidate says:

“I validated, tuned, and improved the model before using it”

---
