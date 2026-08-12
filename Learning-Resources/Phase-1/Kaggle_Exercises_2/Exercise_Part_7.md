# Kaggle Intermediate Machine Learning

# Exercise 7 — Gradient Boosting with XGBoost

---

## 🎯 Exercise Overview

This exercise puts the **Gradient Boosting / XGBoost** tutorial into practice.

The goal is to:

1. Build a basic XGBoost model.
2. Measure its MAE.
3. Tune XGBoost parameters to improve performance.
4. Intentionally choose a poor configuration and observe worse performance.

The main model used is:

```
XGBRegressor
```

The main evaluation metric is:

```
Mean Absolute Error (MAE)
```

The exercise demonstrates an important machine learning idea:

```
Model
  ↓
Measure performance
  ↓
Change hyperparameters
  ↓
Measure again
  ↓
Compare results
```

---

# 1. Dataset

The exercise uses the:

**Housing Prices Competition for Kaggle Learn Users**

dataset.

The target variable is:

```
SalePrice
```

The features are based on the Ames Housing dataset.

---

# 2. Train / Validation Split

The data is divided into:

```
X_train
X_valid
y_train
y_valid
```

using:

```
train_test_split()
```

The exercise uses:

```
train_size=0.8
test_size=0.2
random_state=0
```

So approximately:

```
80% → Training
20% → Validation
```

---

# 3. Load the Data

The training and test datasets are loaded using Pandas.

```
X = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

X_test_full = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)
```

---

# 4. Remove Missing Target Values

Rows where `SalePrice` is missing are removed:

```
X.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)
```

Then the target is separated:

```
y = X.SalePrice
```

The target column is removed from the feature DataFrame:

```
X.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)
```

So:

```
X → Features

y → SalePrice
```

---

# 5. Create Training and Validation Sets

The dataset is split using:

```
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)
```

The important idea:

```
Original dataset
      ↓
┌───────────────┐
│               │
Training      Validation
  80%            20%
│               │
↓               ↓
Train model    Evaluate model
```

---

# 6. What Is Cardinality?

The exercise uses the concept of **cardinality**.

Cardinality means:

> The number of unique values in a column.

For example:

```
Color
```

might contain:

```
Red
Blue
Green
```

Cardinality:

```
3
```

---

# 7. Select Low-Cardinality Categorical Columns

The exercise selects categorical columns with fewer than 10 unique values.

```
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10
    and X_train_full[cname].dtype == "object"
]
```

The conditions are:

```
dtype == "object"
    ↓
categorical column

nunique() < 10
    ↓
relatively low cardinality
```

This keeps the categorical representation manageable.

---

# 8. Select Numerical Columns

Numerical columns are selected with:

```
numeric_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].dtype in [
        'int64',
        'float64'
    ]
]
```

---

# 9. Combine the Selected Features

The exercise combines:

```
low_cardinality_cols
```

and:

```
numeric_cols
```

into:

```
my_cols = low_cardinality_cols + numeric_cols
```

Then the selected columns are copied:

```
X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
```

---

# 10. One-Hot Encoding

The exercise uses Pandas:

```
pd.get_dummies()
```

to one-hot encode the categorical variables.

```
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)
```

---

# 11. Why Align the Columns?

After one-hot encoding, different datasets can potentially have different columns.

For example:

Training:

```
Color_Red
Color_Blue
Color_Green
```

Validation might contain:

```
Color_Red
Color_Blue
```

The columns need to be aligned before passing them to the model.

The exercise uses:

```
X_train, X_valid = X_train.align(
    X_valid,
    join='left',
    axis=1
)
```

and:

```
X_train, X_test = X_train.align(
    X_test,
    join='left',
    axis=1
)
```

This makes the feature structure consistent.

---

# 12. Import XGBoost

The model used in this exercise is:

```
XGBRegressor
```

Import:

```
from xgboost import XGBRegressor
```

XGBoost stands for:

**Extreme Gradient Boosting**

It is an implementation of gradient boosting designed for high performance.

---

# STEP 1 — Build the Baseline Model

The first goal is to create a basic XGBoost model using default parameters.

The exercise specifically asks for:

```
XGBRegressor(random_state=0)
```

All other parameters are left at their defaults.

---

# 13. Step 1A — Define the Model

```
from xgboost import XGBRegressor

my_model_1 = XGBRegressor(
    random_state=0
)
```

The important point is:

```
random_state=0
```

No other parameter is manually changed.

---

# 14. Train the Model

The model is trained using:

```
my_model_1.fit(
    X_train,
    y_train
)
```

Conceptually:

```
X_train
   +
y_train
   ↓
XGBRegressor
   ↓
Trained model
```

---

# 15. Step 1B — Generate Predictions

The validation predictions are created using:

```
predictions_1 = my_model_1.predict(
    X_valid
)
```

The model has never trained on:

```
X_valid
```

So this gives us an estimate of how well the model generalizes.

---

# 16. Step 1C — Calculate MAE

The exercise uses:

```
mean_absolute_error()
```

Import:

```
from sklearn.metrics import mean_absolute_error
```

Then:

```
mae_1 = mean_absolute_error(
    predictions_1,
    y_valid
)
```

And:

```
print(
    "Mean Absolute Error:",
    mae_1
)
```

---

# 17. What Does MAE Tell Us?

MAE means:

**Mean Absolute Error**

It measures the average absolute difference between:

```
Actual value
```

and:

```
Predicted value
```

For example:

```
Actual:     300,000
Predicted:  280,000
```

Absolute error:

```
20,000
```

MAE calculates this type of error across all validation examples and averages it.

---

# 18. Why Lower MAE Is Better

Suppose two models produce:

```
Model A → MAE = 25,000

Model B → MAE = 40,000
```

Model A is better because:

```
25,000 < 40,000
```

Remember:

```
Lower MAE = Better
```

---

# 19. Baseline Model

The first model is our **baseline**.

Baseline means:

> A reference model that we can compare improved or worse models against.

The exercise calls it:

```
my_model_1
```

Everything in the next steps is compared against this model.

---

# STEP 2 — Improve the Model

Now the exercise asks us to improve the baseline model.

The tutorial introduced important XGBoost hyperparameters:

```
n_estimators

learning_rate
```

These can strongly affect model performance.

---

# 20. `n_estimators`

`n_estimators` controls:

> The number of boosting rounds / trees in the ensemble.

For example:

```
n_estimators=100
```

means the model builds 100 boosting stages.

In this exercise, the improved model uses:

```
n_estimators=1000
```

---

# 21. Why Increase `n_estimators`?

A small number of trees may cause:

```
Underfitting
```

The model may not be complex enough to capture the patterns in the data.

Increasing the number of trees gives the boosting process more opportunities to improve the predictions.

However, more trees also mean:

```
More computation
```

and if not controlled appropriately, potentially:

```
Overfitting
```

---

# 22. `learning_rate`

The learning rate controls:

> How much each new tree contributes to the overall model.

A smaller learning rate means each tree has a smaller influence.

Therefore, we can often use more trees.

Conceptually:

```
High learning rate
    ↓
Larger updates
    ↓
Fewer trees may be needed
```

Whereas:

```
Low learning rate
    ↓
Smaller updates
    ↓
More trees may be useful
```

---

# 23. Improved Model Configuration

The exercise uses:

```
my_model_2 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

Notice that:

```
n_estimators = 1000
```

and:

```
learning_rate = 0.05
```

are different from the default configuration.

---

# 24. Train Model 2

```
my_model_2.fit(
    X_train,
    y_train
)
```

---

# 25. Generate Predictions

```
predictions_2 = my_model_2.predict(
    X_valid
)
```

---

# 26. Calculate Model 2 MAE

```
mae_2 = mean_absolute_error(
    predictions_2,
    y_valid
)
```

Then:

```
print(
    "Mean Absolute Error:",
    mae_2
)
```

---

# 27. What Is the Goal of Step 2?

The exercise requires:

```
MAE_2 < MAE_1
```

In other words:

```
Improved model
    ↓
Lower MAE
    ↓
Better validation performance
```

The exact important point is not memorizing one particular MAE value.

The important lesson is:

> Hyperparameter changes can improve model performance.

---

# 28. Step 2 Complete Flow

```
X_train
   ↓
XGBRegressor
   │
   ├── n_estimators=1000
   │
   └── learning_rate=0.05
   ↓
Fit
   ↓
X_valid
   ↓
Predictions
   ↓
MAE
   ↓
Compare with Model 1
```

---

# STEP 3 — Break the Model

This step is intentionally different.

The goal is to create a model that performs **worse** than the original baseline model.

This helps build intuition about hyperparameters.

---

# 29. Poor Model Configuration

The exercise uses:

```
my_model_3 = XGBRegressor(
    n_estimators=1
)
```

Only one boosting tree / stage is used.

---

# 30. Why Can One Estimator Be Bad?

Gradient boosting improves its predictions through repeated boosting rounds.

Conceptually:

```
Initial prediction
      ↓
Tree 1
      ↓
Correct errors
      ↓
Tree 2
      ↓
Correct errors
      ↓
Tree 3
      ↓
...
      ↓
Final model
```

If we stop after:

```
1
```

boosting stage, the model has very little opportunity to improve.

This can result in:

```
Underfitting
```

---

# 31. Train Model 3

```
my_model_3.fit(
    X_train,
    y_train
)
```

---

# 32. Generate Predictions

```
predictions_3 = my_model_3.predict(
    X_valid
)
```

---

# 33. Calculate MAE

```
mae_3 = mean_absolute_error(
    predictions_3,
    y_valid
)
```

Then:

```
print(
    "Mean Absolute Error:",
    mae_3
)
```

---

# 34. Requirement for Step 3

The exercise requires:

```
MAE_3 > MAE_1
```

Meaning:

```
Model 3
   ↓
Higher MAE
   ↓
Worse performance
```

This demonstrates that poor hyperparameter choices can damage model performance.

---

# 35. Three Models Compared

The exercise creates three versions.

### Model 1 — Baseline

```
XGBRegressor(
    random_state=0
)
```

Purpose:

```
Baseline
```

---

### Model 2 — Improved

```
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

Purpose:

```
Improve performance
```

---

### Model 3 — Intentionally Poor

```
XGBRegressor(
    n_estimators=1
)
```

Purpose:

```
Demonstrate poor parameter choice
```

---

# 36. Conceptual Comparison

```
Model 1
   ↓
Default configuration
   ↓
Baseline MAE

        ↓

Model 2
   ↓
Tuned configuration
   ↓
Lower MAE
   ↓
Better

        ↓

Model 3
   ↓
Too few estimators
   ↓
Higher MAE
   ↓
Worse
```

---

# 37. The Most Important XGBoost Hyperparameters

From the tutorial and this exercise, remember:

## `n_estimators`

Controls the number of boosting stages / trees.

```
Too low
   ↓
Underfitting

More appropriate value
   ↓
Better learning

Too many without appropriate control
   ↓
Potential overfitting
```

---

## `learning_rate`

Controls how strongly each boosting stage contributes.

```
Smaller learning rate
    ↓
Smaller updates
    ↓
Often requires more estimators
```

---

## `random_state`

Controls reproducibility.

Example:

```
random_state=0
```

Using the same random state helps produce repeatable results.

---

# 38. Relationship Between `n_estimators` and `learning_rate`

These two parameters should be understood together.

Imagine:

```
n_estimators = number of steps
```

and:

```
learning_rate = size of each step
```

A smaller learning rate means:

```
smaller steps
```

Therefore you often need:

```
more steps
```

to reach a good solution.

Conceptually:

```
Small learning rate
       +
Large n_estimators
       ↓
Slow but potentially strong learning
```

---

# 39. Underfitting in This Exercise

Model 3:

```
n_estimators=1
```

is intentionally weak.

This demonstrates underfitting.

Underfitting means:

> The model is too simple to capture the underlying patterns in the data.

Symptoms:

```
Poor training performance
Poor validation performance
```

---

# 40. Overfitting Reminder

The opposite problem is:

```
Overfitting
```

A model can become too specialized to the training data.

Then:

```
Training performance → Very good
```

but:

```
Validation performance → Poor
```

The goal is not:

```
Make training error as low as possible
```

The goal is:

```
Generalize well to unseen data
```

---

# 41. Why Use a Validation Set?

The validation set allows us to compare models that were trained on:

```
X_train
```

against data they did not train on:

```
X_valid
```

This lets us determine whether a parameter change actually improves generalization.

---

# 42. Complete Exercise Code

```
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Load data
X = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

X_test_full = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)

# Remove missing targets
X.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)

# Separate target
y = X.SalePrice

X.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)

# Train-validation split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)

# Low-cardinality categorical columns
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10
    and X_train_full[cname].dtype == "object"
]

# Numerical columns
numeric_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].dtype in [
        'int64',
        'float64'
    ]
]

# Selected columns
my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encoding
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)

# Align columns
X_train, X_valid = X_train.align(
    X_valid,
    join='left',
    axis=1
)

X_train, X_test = X_train.align(
    X_test,
    join='left',
    axis=1
)

# -------------------------
# MODEL 1 — BASELINE
# -------------------------

my_model_1 = XGBRegressor(
    random_state=0
)

my_model_1.fit(
    X_train,
    y_train
)

predictions_1 = my_model_1.predict(
    X_valid
)

mae_1 = mean_absolute_error(
    predictions_1,
    y_valid
)

print(
    "Model 1 MAE:",
    mae_1
)

# -------------------------
# MODEL 2 — IMPROVED
# -------------------------

my_model_2 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)

my_model_2.fit(
    X_train,
    y_train
)

predictions_2 = my_model_2.predict(
    X_valid
)

mae_2 = mean_absolute_error(
    predictions_2,
    y_valid
)

print(
    "Model 2 MAE:",
    mae_2
)

# -------------------------
# MODEL 3 — POOR MODEL
# -------------------------

my_model_3 = XGBRegressor(
    n_estimators=1
)

my_model_3.fit(
    X_train,
    y_train
)

predictions_3 = my_model_3.predict(
    X_valid
)

mae_3 = mean_absolute_error(
    predictions_3,
    y_valid
)

print(
    "Model 3 MAE:",
    mae_3
)
```

---

# 43. Full Mental Model

Think of XGBoost as an iterative error-correction process.

```
Start
  ↓
Make prediction
  ↓
Find errors
  ↓
Add a tree that helps correct errors
  ↓
Update ensemble
  ↓
Find remaining errors
  ↓
Add another tree
  ↓
Repeat
  ↓
Final prediction
```

`n_estimators` controls approximately how many of these boosting stages are performed.

`learning_rate` controls how strongly each stage contributes.

---

# 44. What This Exercise Teaches Beyond XGBoost

The deeper lesson is:

> **Machine learning models are not just about choosing an algorithm. Hyperparameter choices can have a major effect on performance.**

You started with:

```
XGBRegressor()
```

Then experimented with:

```
n_estimators
```

and:

```
learning_rate
```

and observed how the model changed.

---

# ⚠️ Common Mistakes

### Mistake 1 — Using the validation data for training

Training should use:

```
X_train
y_train
```

Validation should be used for:

```
prediction
evaluation
```

---

### Mistake 2 — Comparing training MAE

The exercise compares predictions on:

```
X_valid
```

against:

```
y_valid
```

---

### Mistake 3 — Thinking more trees always means better

More trees can help, but there is no universal rule that:

```
more trees = better model
```

Performance must be measured.

---

### Mistake 4 — Confusing learning rate with number of trees

Remember:

```
n_estimators
→ number of boosting stages

learning_rate
→ contribution of each stage
```

---

### Mistake 5 — Thinking `n_estimators=1` means one feature

It does not.

It means the boosting ensemble contains only one estimator/tree.

---

### Mistake 6 — Memorizing the exact improved configuration

The exercise uses:

```
n_estimators=1000
learning_rate=0.05
```

But the important lesson is **hyperparameter tuning**, not that these values are universally optimal.

---

# 🧪 Code Cheat Sheet

## Basic XGBoost

```
model = XGBRegressor(
    random_state=0
)
```

---

## Train

```
model.fit(
    X_train,
    y_train
)
```

---

## Predict

```
predictions = model.predict(
    X_valid
)
```

---

## MAE

```
mae = mean_absolute_error(
    predictions,
    y_valid
)
```

---

## Tune Number of Trees

```
model = XGBRegressor(
    n_estimators=1000
)
```

---

## Tune Learning Rate

```
model = XGBRegressor(
    learning_rate=0.05
)
```

---

## Tune Both

```
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

---

# 🎤 Interview Revision

### What is XGBoost?

XGBoost stands for Extreme Gradient Boosting and is an implementation of gradient boosting.

### What type of algorithm is XGBoost?

An ensemble learning method based on gradient boosting, commonly using decision trees.

### What does `n_estimators` control?

The number of boosting stages / trees in the ensemble.

### What happens when `n_estimators` is too low?

The model can underfit.

### What does `learning_rate` control?

How strongly each new boosting stage contributes to the ensemble.

### What is the relationship between learning rate and number of estimators?

A smaller learning rate often requires more estimators to achieve good performance.

### Why use a validation set?

To evaluate how well the model generalizes to data it did not train on.

### What metric is used in this exercise?

Mean Absolute Error.

### Is lower MAE better?

Yes.

### Why is Model 1 important?

It provides a baseline for comparison.

### What is the goal of Model 2?

To achieve lower MAE than Model 1.

### What is the goal of Model 3?

To intentionally produce higher MAE than Model 1 and demonstrate how poor hyperparameter choices can hurt performance.

---

# 🧠 Exercise 7 — One-Minute Revision

```
XGBRegressor
     ↓
Gradient Boosting
     ↓
Build baseline
     ↓
Evaluate MAE
     ↓
Tune hyperparameters
     ↓
n_estimators
     +
learning_rate
     ↓
Lower MAE
     ↓
Better model
```

Baseline:

```
XGBRegressor(
    random_state=0
)
```

Improved exercise model:

```
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

Intentionally poor model:

```
XGBRegressor(
    n_estimators=1
)
```

Remember:

```
Lower MAE = Better

n_estimators
= number of boosting stages

learning_rate
= contribution of each stage

Too few estimators
= possible underfitting

Hyperparameter tuning
= test configurations and compare validation performance
```

---

# ⭐ Core Lesson

> **XGBoost builds an ensemble iteratively, adding trees that improve the model's predictions. Hyperparameters such as `n_estimators` and `learning_rate` strongly influence performance. The correct way to choose them is to experiment and evaluate on validation data rather than blindly assuming that a particular value is best.**
