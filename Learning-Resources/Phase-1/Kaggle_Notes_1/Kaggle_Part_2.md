# 🧠 Intro to Machine Learning — Kaggle Notes (Part 2)

---

# 🔹 5. Model Validation (VERY IMPORTANT)

## 📖 Concept

After building a model:

👉 The most important question is:
**“How good is my model?”**

We measure this using **model validation**

---

## ❌ Common Mistake

* Predict on **training data**
* Compare with actual values

👉 This is WRONG

Why?

* Model already **saw this data**
* It will look artificially accurate

---

## 🧠 Real Goal

👉 We care about performance on **new unseen data**

---

# 🔹 6. Evaluation Metric — MAE (Mean Absolute Error)

## 📖 Concept

MAE measures:

👉 Average difference between **actual vs predicted**

Formula:

```
error = actual - predicted  
MAE = average(|error|)
```

---

## 🧠 Interpretation

👉 “On average, model is off by X amount”

Example:

* Actual = 150k
* Predicted = 100k
* Error = 50k

---

## 🧪 Code

```
from sklearn.metrics import mean_absolute_error

predictions = model.predict(X)
mean_absolute_error(y, predictions)
```

---

## 🧠 Code Explanation

* `predict()` → generate predictions
* `mean_absolute_error()` → calculate average error

---

## ⚠️ Important Notes

* Lower MAE = better model
* Easy to understand metric
* Works well for regression

---

## 🔁 When NOT to Use

* When large errors should be penalized more
  👉 Use:
* RMSE (Root Mean Squared Error)

---

# 🔹 7. In-Sample vs Out-of-Sample

## 📖 Concept

### ❌ In-Sample

* Train + Test on same data
* Gives **fake confidence**

### ✅ Out-of-Sample

* Train on one set
* Test on unseen data

👉 This is real performance

---

## 🧠 Key Insight

👉 Good training accuracy ≠ good real-world model

---

# 🔹 8. Train-Test Split

## 📖 Concept

Split dataset into:

* **Training Data** → build model
* **Validation Data** → test model

---

## 🧪 Code

```
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
```

---

## 🧠 Code Explanation

* Splits data randomly
* `random_state` → same split every time

---

## 🧪 Full Validation Example

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

model = DecisionTreeRegressor()
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
mean_absolute_error(val_y, val_predictions)
```

---

## ⚠️ Important Notes

* Always validate model
* Never trust training score
* Validation simulates real-world usage

---

## 🔥 Real Insight (From Course)

* Training MAE ≈ very low (~500)
* Validation MAE ≈ very high (~250,000)

👉 Huge difference = model is **not reliable**

---

# 🔹 9. Why This Happens

## 📖 Concept

Model memorizes training data patterns
👉 But fails on new data

Example:

* Model thinks “green door = expensive house”
* Works in training data
* Fails in real world

---

## ⚠️ Important Notes

* Models can learn **wrong patterns**
* Validation prevents this mistake

---

# 📝 Exercise Insights

* Always:

  * Split data
  * Train model
  * Evaluate on validation set

* Don’t:

  * Evaluate on same data

---

# 🔥 Final Quick Revision (Part 2)

* Validation = test model on unseen data
* MAE = average prediction error
* Train-Test Split = avoid fake accuracy
* In-sample = misleading
* Out-of-sample = real performance

---

# 🚀 What’s Next

👉 Part 3: Overfitting vs Underfitting + Model Tuning

---