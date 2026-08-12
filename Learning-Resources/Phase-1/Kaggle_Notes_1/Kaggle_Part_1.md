# 🧠 Intro to Machine Learning — Kaggle Notes (Part 1)

---

# 🔹 1. How Models Work (Decision Trees)

## 📖 Concept

Machine Learning =
👉 Learn patterns from past data
👉 Use those patterns to predict new data

Real-world analogy:

- Your cousin predicts house prices using **experience (patterns)**
- ML model does the same using **data**

---

## 🌳 Decision Tree (Core Idea)

- Works like **if-else rules**
- Splits data into groups
- Each group gets an average value

Example logic:

```
if bedrooms < 3 → low price
else → high price
```

---

## 🧠 Key Terms

- **Training (Fitting)** → Learning patterns from data
- **Training Data** → Data used to train
- **Prediction** → Output on new data
- **Split** → Rule dividing data
- **Leaf Node** → Final output

---

## ⚙️ Practical Usage

Used for:

- Price prediction
- Loan approval
- Fraud detection
- Basic ML baseline models

---

## ⚠️ Important Notes

- Simple tree → less accurate
- Deep tree → risk of overfitting
- Doesn’t need feature scaling
- Handles non-linear data

---

## 🔁 When NOT to Use

- Very small dataset
- Need high accuracy

👉 Use:

- Random Forest
- Gradient Boosting

---

## 🔥 Quick Insight

👉 Decision Tree = learned if-else rules
👉 Output = average of values in leaf

---

# 🔹 2. Data Exploration with Pandas

## 📖 Concept

Before building models:

👉 Understand your data first

Pandas is used for:

- Reading data
- Exploring data
- Cleaning data

---

## 🧪 Code

```
import pandas as pd

file_path = "melb_data.csv"
data = pd.read_csv(file_path)

data.describe()
```

---

## 🧠 Code Explanation

- `pd.read_csv()` → load dataset
- `DataFrame` → table-like structure
- `describe()` → summary statistics

---

## 📊 What describe() Shows

- **count** → non-missing values
- **mean** → average
- **std** → spread
- **min / max** → range
- **25%, 50%, 75%** → percentiles

---

## ⚠️ Important Notes

- Missing values are common
- Always inspect data before modeling
- Outliers can affect model

---

## 🔁 When NOT to Skip

❌ Never skip data exploration
👉 Leads to wrong models

---

## 📝 Exercise Insight

- Always check:
  - columns
  - missing values
  - basic stats

---

# 🔹 3. Selecting Data for Modeling

## 📖 Concept

You don’t use all data — you choose:

- **Target (y)** → what to predict
- **Features (X)** → inputs

---

## 🧪 Code

```
y = data.Price

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]
```

---

## 🧠 Code Explanation

- `y` → target variable
- `X` → input features
- Features should influence output

---

## ⚠️ Important Notes

- Bad features = bad model
- Too many features = noise
- Missing values → drop or handle

---

## 🔁 When NOT to Do

❌ Don’t randomly pick features
👉 Use logic or domain knowledge

---

## 📝 Exercise Insight

- Practice selecting correct columns
- Understand feature importance

---

# 🔹 4. Building Your First Model

## 📖 Concept

Steps in ML:

1. Define model
2. Fit (train)
3. Predict
4. Evaluate

---

## 🧪 Code

```
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

predictions = model.predict(X.head())
```

---

## 🧠 Code Explanation

- `DecisionTreeRegressor()` → regression model
- `random_state` → same results every run
- `fit()` → training
- `predict()` → output

---

## ⚠️ Important Notes

- Model learns from training data
- Predictions on same data are misleading
- Always validate (next part)

---

## 📝 Exercise Insight

- Train model
- Make predictions
- Understand output format

---

# 🔥 Final Quick Revision (Part 1)

- ML = pattern learning
- Decision Tree = rule-based model
- Pandas = data exploration tool
- X = features, y = target
- fit() = training
- predict() = inference

---

# 🚀 What’s Next

👉 Part 2: Model Validation + MAE + Train/Test Split

---
