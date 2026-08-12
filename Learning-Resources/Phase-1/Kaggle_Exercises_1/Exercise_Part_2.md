# 🔹 Exercise 3: Model Validation

(From: model-validation notebook)

---

## 📖 Concept

Big problem in ML:

👉 Model looks good… but fails in real world

Why?

👉 Because you tested it on **training data**

---

## ⚙️ What You Actually Did (REAL FLOW)

---

### Step 1 — Split Data

```
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

---

### 🧠 Explanation

- `train_X`, `train_y` → used for training
- `val_X`, `val_y` → used for testing

👉 This simulates **real-world unseen data**

---

## 🎯 Core Idea

👉 Train on one set
👉 Test on another set

---

### Step 2 — Train Model

```
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)
```

---

### Step 3 — Make Predictions on Validation Data

```
val_predictions = model.predict(val_X)
```

---

### Step 4 — Measure Error

```
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
```

---

## 🧠 What is MAE?

👉 Mean Absolute Error

```
MAE = average(|actual - predicted|)
```

---

## 🎯 Why MAE?

- Easy to understand
- Same unit as target (price)

👉 “On average, model is off by ₹X”

---

## ⚠️ Important Notes

- Lower MAE = better model
- Always compare models using SAME metric

---

## ❌ Biggest Mistake

👉 Evaluating on training data

---

## 🎤 Interview Answer

If asked:

👉 “How do you evaluate model?”

Answer:

> I split data into train and validation sets and use metrics like MAE to evaluate performance on unseen data.

---

# 🔹 Exercise 4: Underfitting vs Overfitting

(From: underfitting-and-overfitting notebook)

---

## 📖 Concept

Two major ML problems:

---

### ❌ Underfitting

- Model too simple
- Doesn’t learn patterns

👉 Bad on:

- Training data
- Validation data

---

### ❌ Overfitting

- Model too complex
- Memorizes data

👉 Good on:

- Training data

👉 Bad on:

- Validation data

---

## ⚙️ What You Actually Did (REAL FLOW)

---

### Step 1 — Create Function to Measure MAE

```
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return mae
```

---

### 🧠 Why This Function?

👉 To test multiple models quickly

---

### Step 2 — Compare Different Models

```
for max_leaf_nodes in [5, 25, 50, 100, 250, 500]:
    print(max_leaf_nodes, get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y))
```

---

## 🧠 What You Observed

| Nodes           | Behavior     |
| --------------- | ------------ |
| Small (5)       | Underfitting |
| Medium (50–100) | Best         |
| Large (500+)    | Overfitting  |

---

## 🎯 Core Insight

👉 There is a **sweet spot**

Not too simple ❌
Not too complex ❌
👉 Just right ✅

---

## ⚠️ Important Notes

- Model complexity = controlled by parameters
- For Decision Tree:
  - `max_leaf_nodes` = key parameter

---

## ❌ Common Mistakes

- Using default parameters blindly
- Not tuning model

---

## 🎤 Interview Answer

👉 “How do you reduce overfitting?”

Answer:

> By tuning model complexity, using validation data, and selecting parameters like max_leaf_nodes to balance bias and variance.

---

# 🔥 FINAL REVISION (Exercise 3 + 4)

## 🧠 Core Flow

```
Split Data
    ↓
Train Model
    ↓
Predict on Validation Data
    ↓
Calculate MAE
    ↓
Tune Parameters
    ↓
Find Best Model
```

---

## 🎯 Golden Concepts

- Train vs Validation
- MAE
- Underfitting vs Overfitting
- Model tuning

---

## 🚀 Real-World Thinking

👉 ML is NOT:

“Train once → done” ❌

👉 ML is:

“Train → Evaluate → Improve → Repeat” 🔁

---

## 💡 Final Insight

👉 Best model =
Not highest training accuracy
👉 Lowest validation error

---
