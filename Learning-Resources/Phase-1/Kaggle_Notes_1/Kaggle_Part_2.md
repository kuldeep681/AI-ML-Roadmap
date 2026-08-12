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

- Predict on **training data**
- Compare with actual values

👉 This is WRONG

Why?

- Model already **saw this data**
- It will look artificially accurate

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

- Actual = 150k
- Predicted = 100k
- Error = 50k

---

## 🧪 Code

```
from sklearn.metrics import mean_absolute_error

predictions = model.predict(X)
mean_absolute_error(y, predictions)
```

---

## 🧠 Code Explanation

- `predict()` → generate predictions
- `mean_absolute_error()` → calculate average error

---

## ⚠️ Important Notes

- Lower MAE = better model
- Easy to understand metric
- Works well for regression

---

## 🔁 When NOT to Use

- When large errors should be penalized more
  👉 Use:
- RMSE (Root Mean Squared Error)

---

# 🔹 7. In-Sample vs Out-of-Sample

## 📖 Concept

### ❌ In-Sample

- Train + Test on same data
- Gives **fake confidence**

### ✅ Out-of-Sample

- Train on one set
- Test on unseen data

👉 This is real performance

---

## 🧠 Key Insight

👉 Good training accuracy ≠ good real-world model

---

# 🔹 8. Train-Test Split

## 📖 Concept

Split dataset into:

- **Training Data** → build model
- **Validation Data** → test model

---

## 🧪 Code

```
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
```

---

## 🧠 Code Explanation

- Splits data randomly
- `random_state` → same split every time

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

- Always validate model
- Never trust training score
- Validation simulates real-world usage

---

## 🔥 Real Insight (From Course)

- Training MAE ≈ very low (~500)
- Validation MAE ≈ very high (~250,000)

👉 Huge difference = model is **not reliable**

---

# 🔹 9. Why This Happens

## 📖 Concept

Model memorizes training data patterns
👉 But fails on new data

Example:

- Model thinks “green door = expensive house”
- Works in training data
- Fails in real world

---

## ⚠️ Important Notes

- Models can learn **wrong patterns**
- Validation prevents this mistake

---

# 📝 Exercise Insights

- Always:
  - Split data
  - Train model
  - Evaluate on validation set

- Don’t:
  - Evaluate on same data

---

# 🔥 Final Quick Revision (Part 2)

- Validation = test model on unseen data
- MAE = average prediction error
- Train-Test Split = avoid fake accuracy
- In-sample = misleading
- Out-of-sample = real performance

---

# 🔹 10. Underfitting vs Overfitting

## 📖 Concept

After validation, you’ll notice:

👉 Model is either:

- Too simple ❌ (Underfitting)
- Too complex ❌ (Overfitting)

---

## ❌ Underfitting

### 📖 What it Means

- Model is **too simple**
- Fails to capture patterns

Example:

- Only splits data into 2 groups
- Ignores important features

---

### 🧠 Behavior

- Bad performance on:
  - Training data ❌
  - Validation data ❌

---

### ⚠️ Why It Happens

- Model too shallow
- Not enough features
- Not enough learning

---

## ❌ Overfitting

### 📖 What it Means

- Model is **too complex**
- Learns noise instead of patterns

Example:

- Too many splits
- Each leaf has very few data points

---

### 🧠 Behavior

- Training accuracy → very high ✅
- Validation accuracy → very poor ❌

---

### ⚠️ Why It Happens

- Tree too deep
- Too many leaves
- Memorizing instead of learning

---

## ⚖️ Goal

👉 Find balance between both

This is called:

### ✅ **Bias-Variance Tradeoff (intuition)**

- Underfitting → high bias
- Overfitting → high variance

---

# 🔹 11. Controlling Model Complexity

## 📖 Concept

In Decision Trees:

👉 Complexity = number of splits (depth)

More splits:

- More detailed learning
- Higher risk of overfitting

---

## 🎯 Key Parameter

### `max_leaf_nodes`

- Controls number of final nodes (leaves)
- Helps balance model complexity

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

- Very small nodes → underfitting
- Very large nodes → overfitting
- Middle value → best performance

👉 Example: **500 nodes = best**

---

## ⚠️ Important Notes

- Don’t guess parameters
- Always test using validation data
- Best model = lowest validation error

---

# 🔹 12. Visual Intuition (IMPORTANT)

## 📖 Concept

Think like this:

- Underfitting → model too dumb
- Overfitting → model too smart (but useless)

Graph idea:

- X-axis → model complexity
- Y-axis → error

👉 Validation error forms a **U-shape**

Best point = lowest error

---

# 📝 Exercise Insights

- Try different `max_leaf_nodes`
- Compare MAE
- Choose best value

---

## ⚠️ Common Mistakes

- Choosing model based on training score ❌
- Not tuning parameters ❌
- Ignoring validation results ❌

---

# 🔥 Final Quick Revision (Part 3)

- Underfitting → too simple
- Overfitting → too complex
- Goal → balance both
- Use `max_leaf_nodes` to control tree
- Best model = lowest validation MAE

---

# 🚀 What’s Next

👉 Part 4: Random Forest (Powerful Models)

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
