# 🧠 Intro to Machine Learning — Kaggle Notes (Part 4)

---

# 🔹 13. Random Forest (Powerful Model)

## 📖 Concept

Decision Trees have a problem:

👉 You must choose between:

- Deep tree → overfitting ❌
- Shallow tree → underfitting ❌

---

## 🌲 Solution → Random Forest

👉 Instead of 1 tree → use MANY trees

- Each tree makes prediction
- Final output = **average of all trees**

---

## 🧠 Why It Works

- Reduces overfitting
- More stable predictions
- Combines multiple weak models → strong model

---

## ⚙️ Practical Usage

Used in:

- Real-world ML systems
- Competitions
- Baseline for tabular data

👉 Often gives good results without heavy tuning

---

## 🧪 Code

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(random_state=1)

model.fit(train_X, train_y)

preds = model.predict(val_X)

print(mean_absolute_error(val_y, preds))
```

---

## 🧠 Code Explanation

- `RandomForestRegressor()` → multiple trees
- `fit()` → trains all trees
- `predict()` → averages predictions

---

## 📊 Result Insight

- Decision Tree MAE ≈ 250,000
- Random Forest MAE ≈ 190,000

👉 Big improvement ✅

---

## ⚠️ Important Notes

- Slower than single tree
- Less interpretable
- Works well even with default settings

---

## 🔁 When NOT to Use

- Need high interpretability
- Very small dataset
- Low compute environment

👉 Alternatives:

- Linear models (simple)
- Gradient Boosting (more powerful)

---

# 🔹 14. Decision Tree vs Random Forest

| Feature          | Decision Tree | Random Forest |
| ---------------- | ------------- | ------------- |
| Accuracy         | Low–Medium    | High          |
| Overfitting      | High risk     | Reduced       |
| Speed            | Fast          | Slower        |
| Interpretability | High          | Low           |

---

# 🔹 15. Full ML Pipeline (IMPORTANT)

## 📖 Concept

Complete ML workflow:

1. Load data
2. Explore data
3. Select features (X, y)
4. Train model
5. Validate model
6. Tune parameters
7. Improve model (Random Forest)

---

## 🔥 This is your CORE ML LOOP

---

# 📝 Exercise Insights

From your exercises:

👉 You practiced:

- Building models
- Splitting data
- Evaluating MAE
- Tuning parameters
- Using Random Forest

👉 Key takeaway:
**Practice = understanding**

---

# ⚠️ Common Interview Mistakes

- Using training data for evaluation ❌
- Ignoring validation ❌
- Not handling overfitting ❌
- Not tuning models ❌

---

# 🔥 FINAL ONE-SHOT REVISION (FULL COURSE)

## 🧠 Core Concepts

- ML = learn patterns from data
- Decision Tree = rule-based model
- Random Forest = multiple trees

---

## 📊 Data Handling

- Use Pandas
- Understand dataset first
- Handle missing values

---

## 🎯 Modeling

- X = features
- y = target
- fit() = training
- predict() = inference

---

## 📉 Evaluation

- Use MAE
- Always validate
- Train-test split

---

## ⚖️ Model Problems

- Underfitting → too simple
- Overfitting → too complex

---

## ⚙️ Improvement

- Tune parameters (`max_leaf_nodes`)
- Use Random Forest

---

## 🚀 Golden Rules

- Never trust training accuracy
- Always validate
- Simpler model first
- Improve step-by-step

---

# 🏁 END OF COURSE SUMMARY

👉 You can now:

- Build ML models
- Evaluate them correctly
- Avoid major mistakes
- Improve performance

---

# 🎯 Final Insight

👉 Good ML ≠ complex model

👉 Good ML =

- Clean data
- Correct validation
- Right model choice

---
