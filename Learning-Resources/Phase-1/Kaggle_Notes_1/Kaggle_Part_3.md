# 🧠 Intro to Machine Learning — Kaggle Notes (Part 3)

---

# 🔹 10. Underfitting vs Overfitting

## 📖 Concept

After validation, you’ll notice:

👉 Model is either:

* Too simple ❌ (Underfitting)
* Too complex ❌ (Overfitting)

---

## ❌ Underfitting

### 📖 What it Means

* Model is **too simple**
* Fails to capture patterns

Example:

* Only splits data into 2 groups
* Ignores important features

---

### 🧠 Behavior

* Bad performance on:

  * Training data ❌
  * Validation data ❌

---

### ⚠️ Why It Happens

* Model too shallow
* Not enough features
* Not enough learning

---

## ❌ Overfitting

### 📖 What it Means

* Model is **too complex**
* Learns noise instead of patterns

Example:

* Too many splits
* Each leaf has very few data points

---

### 🧠 Behavior

* Training accuracy → very high ✅
* Validation accuracy → very poor ❌

---

### ⚠️ Why It Happens

* Tree too deep
* Too many leaves
* Memorizing instead of learning

---

## ⚖️ Goal

👉 Find balance between both

This is called:

### ✅ **Bias-Variance Tradeoff (intuition)**

* Underfitting → high bias
* Overfitting → high variance

---

# 🔹 11. Controlling Model Complexity

## 📖 Concept

In Decision Trees:

👉 Complexity = number of splits (depth)

More splits:

* More detailed learning
* Higher risk of overfitting

---

## 🎯 Key Parameter

### `max_leaf_nodes`

* Controls number of final nodes (leaves)
* Helps balance model complexity

---

## 🧪 Code (Comparison Function)

```
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)
```

---

## 🧪 Testing Different Values

```
for nodes in [5, 50, 500, 5000]:
    mae = get_mae(nodes, train_X, val_X, train_y, val_y)
    print(nodes, mae)
```

---

## 🧠 Result Insight

* Very small nodes → underfitting
* Very large nodes → overfitting
* Middle value → best performance

👉 Example: **500 nodes = best**

---

## ⚠️ Important Notes

* Don’t guess parameters
* Always test using validation data
* Best model = lowest validation error

---

# 🔹 12. Visual Intuition (IMPORTANT)

## 📖 Concept

Think like this:

* Underfitting → model too dumb
* Overfitting → model too smart (but useless)

Graph idea:

* X-axis → model complexity
* Y-axis → error

👉 Validation error forms a **U-shape**

Best point = lowest error

---

# 📝 Exercise Insights

* Try different `max_leaf_nodes`
* Compare MAE
* Choose best value

---

## ⚠️ Common Mistakes

* Choosing model based on training score ❌
* Not tuning parameters ❌
* Ignoring validation results ❌

---

# 🔥 Final Quick Revision (Part 3)

* Underfitting → too simple
* Overfitting → too complex
* Goal → balance both
* Use `max_leaf_nodes` to control tree
* Best model = lowest validation MAE

---

# 🚀 What’s Next

👉 Part 4: Random Forest (Powerful Models)

---