# Kaggle Intermediate Machine Learning

# Exercise 4 — Gradient Boosting with XGBoost

---

## 🎯 Exercise Overview

This exercise is about **gradient boosting** using **XGBoost**.

You will:

- Prepare the Housing Prices dataset
- Build your first XGBoost regression model
- Generate validation predictions
- Calculate MAE
- Improve the model by changing hyperparameters
- Deliberately create a worse model
- Understand how model parameters affect performance

The main idea is:

> **Train an XGBoost model, evaluate it with MAE, then experiment with hyperparameters to improve or worsen its performance.**

---

# 1. Dataset

The exercise uses the **Housing Prices Competition for Kaggle Learn Users** dataset.

The target variable is:

```
SalePrice
```

The data is divided into:

```
X_train
X_valid
y_train
y_valid
```

There is also:

```
X_test
```

for the test dataset.

---

# 2. Load the Data

The exercise starts by loading:

```
train.csv
test.csv
```

using Pandas.

```
import pandas as pd
from sklearn.model_selection import train_test_split

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

# 3. Remove Rows Without the Target

The target is:

```
SalePrice
```

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

And removed from the feature DataFrame:

```
X.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)
```

So now:

```
X → Features
y → Target
```

---

# 4. Create Training and Validation Sets

The exercise uses an 80/20 split:

```
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)
```

Meaning:

```
80% → Training
20% → Validation
```

`random_state=0` makes the split reproducible.

---

# 5. What Is Cardinality?

The exercise introduces the term **cardinality**.

Cardinality means:

> **The number of unique values in a column.**

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

The exercise selects categorical columns having fewer than 10 unique values.

---

# 6. Select Low-Cardinality Categorical Columns

The code:

```
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10
    and X_train_full[cname].dtype == "object"
]
```

This means:

> Select columns that are categorical and have fewer than 10 unique values.

---

# 7. Select Numerical Columns

Numerical columns are selected using their data type:

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

# 8. Combine the Selected Features

The selected columns are combined:

```
my_cols = low_cardinality_cols + numeric_cols
```

Then the training, validation, and test datasets are reduced to those columns:

```
X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
```

---

# 9. One-Hot Encoding

The exercise uses Pandas' `get_dummies()` for one-hot encoding.

```
X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)
```

Categorical values are converted into numerical indicator columns.

For example:

```
Type = House
```

could become something like:

```
Type_House = 1
```

---

# 10. Align the Columns

There is an important step after one-hot encoding.

```
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
```

Why?

Because training, validation, and test datasets can have different categorical values.

For example:

Training:

```
Red
Blue
Green
```

Validation:

```
Red
Blue
```

After encoding, the columns might not automatically match.

`align()` makes their columns consistent.

---

# 🧠 Data Preparation Flow

The exercise's preprocessing can be remembered as:

```
Raw Dataset
     ↓
Remove missing SalePrice
     ↓
Separate X and y
     ↓
Train/Validation Split
     ↓
Find categorical columns
     ↓
Keep low-cardinality categorical columns
     ↓
Keep numerical columns
     ↓
One-hot encode
     ↓
Align columns
     ↓
X_train / X_valid ready
     ↓
XGBoost
```

---

# 11. What Is Gradient Boosting?

Gradient boosting is an **ensemble learning technique**.

Instead of relying on one model, it builds a sequence of models.

Each new model tries to improve the mistakes made by the previous ensemble.

Conceptually:

```
Model 1
   ↓
Errors
   ↓
Model 2 learns from errors
   ↓
Errors
   ↓
Model 3 learns from errors
   ↓
...
   ↓
Final prediction
```

The models are added sequentially.

---

# 12. What Is XGBoost?

XGBoost stands for:

> **Extreme Gradient Boosting**

It is a popular implementation of gradient boosting.

It is especially powerful for:

- tabular data
- structured datasets
- regression
- classification
- Kaggle competitions

The exercise uses:

```
XGBRegressor
```

because the target:

```
SalePrice
```

is continuous.

---

# 13. XGBRegressor

Import:

```
from xgboost import XGBRegressor
```

Create a model:

```
my_model = XGBRegressor(
    random_state=0
)
```

Then train it:

```
my_model.fit(
    X_train,
    y_train
)
```

---

# 14. Step 1 — Build the First Model

The first exercise asks you to create an XGBoost model using default parameters.

The required model is:

```
my_model_1 = XGBRegressor(
    random_state=0
)
```

Then:

```
my_model_1.fit(
    X_train,
    y_train
)
```

This becomes your **baseline model**.

---

# 15. Why Is the First Model a Baseline?

The first model gives you a reference point.

Suppose its MAE is:

```
MAE = 250,000
```

Now you can experiment with different parameters.

If another model gives:

```
MAE = 220,000
```

you know that the second model improved performance.

Without a baseline, you don't have a clear comparison.

---

# 16. Step 1 Part B — Generate Predictions

Once the model is trained:

```
predictions_1 = my_model_1.predict(
    X_valid
)
```

The model takes:

```
X_valid
```

and produces:

```
predictions_1
```

These are predicted house prices.

---

# 17. Step 1 Part C — Calculate MAE

Import:

```
from sklearn.metrics import mean_absolute_error
```

Calculate:

```
mae_1 = mean_absolute_error(
    predictions_1,
    y_valid
)
```

Then:

```
print(
    "Mean Absolute Error:",
    mae_1
)
```

---

# 18. Understanding the MAE Calculation

MAE measures the average absolute difference between:

```
Actual Price
    vs
Predicted Price
```

Formula:

```
MAE = average(|actual - predicted|)
```

For example:

Actual:

```
500,000
```

Predicted:

```
450,000
```

Error:

```
50,000
```

The absolute error is:

```
50,000
```

Lower MAE means better predictions.

---

# 19. Complete Step 1

The complete logic is:

```
X_train
   ↓
XGBRegressor
   ↓
fit()
   ↓
trained model
   ↓
predict(X_valid)
   ↓
predictions
   ↓
mean_absolute_error()
   ↓
MAE
```

Code:

```
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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
    "Mean Absolute Error:",
    mae_1
)
```

---

# 20. Step 2 — Improve the Model

Now the exercise asks you to improve the baseline model.

You are specifically encouraged to experiment with parameters such as:

```
n_estimators
learning_rate
```

The model must achieve:

```
MAE_2 < MAE_1
```

Remember:

```
Lower MAE = Better
```

---

# 21. `n_estimators`

`n_estimators` controls the number of boosting rounds/models added to the ensemble.

In this exercise:

```
n_estimators=1000
```

means the model can build up to 1000 boosting stages.

Generally:

```
Too few
   ↓
May underfit

Too many
   ↓
May overfit / increase computation
```

The ideal value depends on the dataset and other parameters.

---

# 22. `learning_rate`

`learning_rate` controls how strongly each new model contributes to the final prediction.

A smaller learning rate means:

> Each boosting step makes a smaller correction.

This often allows you to use more estimators.

A common relationship is:

```
Smaller learning_rate
        +
More n_estimators
```

---

# 23. Step 2 Solution Used in the Exercise

The notebook uses:

```
my_model_2 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

Then:

```
my_model_2.fit(
    X_train,
    y_train
)
```

Predictions:

```
predictions_2 = my_model_2.predict(
    X_valid
)
```

MAE:

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

The purpose is to obtain a lower MAE than the baseline model.

---

# 24. Why This Configuration Can Improve the Model

The baseline uses mostly default parameters.

The second model changes:

```
n_estimators
learning_rate
```

to allow the boosting process to make many smaller improvements.

Instead of:

```
fewer → larger updates
```

we use:

```
many → smaller updates
```

This can result in better validation performance.

---

# 25. Complete Step 2

```
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
    "Mean Absolute Error:",
    mae_2
)
```

The objective:

```
mae_2 < mae_1
```

---

# 26. Step 3 — Break the Model

This step is intentionally different.

Instead of improving the model, you are asked to create a model that performs **worse** than the original baseline.

The condition is:

```
MAE_3 > MAE_1
```

This helps you understand the effect of model parameters.

---

# 27. Step 3 Solution

The notebook uses:

```
my_model_3 = XGBRegressor(
    n_estimators=1
)
```

Then:

```
my_model_3.fit(
    X_train,
    y_train
)
```

Predictions:

```
predictions_3 = my_model_3.predict(
    X_valid
)
```

MAE:

```
mae_3 = mean_absolute_error(
    predictions_3,
    y_valid
)
```

---

# 28. Why Does `n_estimators=1` Make the Model Worse?

Gradient boosting improves the ensemble by adding models sequentially.

With:

```
n_estimators=1
```

you are giving the model only one boosting stage.

There is very little opportunity to progressively correct errors.

Therefore, the model can underfit the data.

Conceptually:

```
n_estimators=1

    ↓

Very few boosting stages

    ↓

Limited learning

    ↓

Higher validation error
```

---

# 29. Complete Step 3

```
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
    "Mean Absolute Error:",
    mae_3
)
```

The objective:

```
mae_3 > mae_1
```

---

# 30. Comparing the Three Models

The exercise creates three models.

### Model 1 — Baseline

```
XGBRegressor(
    random_state=0
)
```

Purpose:

```
Establish baseline performance.
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
Improve validation MAE.
```

Expected:

```
MAE_2 < MAE_1
```

---

### Model 3 — Worse

```
XGBRegressor(
    n_estimators=1
)
```

Purpose:

```
Demonstrate poor parameter selection.
```

Expected:

```
MAE_3 > MAE_1
```

---

# 🧠 Parameter Intuition

Think of gradient boosting like repeatedly correcting mistakes.

```
First model
    ↓
Makes mistakes
    ↓
New model corrects mistakes
    ↓
More mistakes corrected
    ↓
Another model
    ↓
More corrections
    ↓
Final ensemble
```

`n_estimators` controls approximately how many correction stages are added.

---

# 31. Important Parameter Relationship

Two parameters from this exercise are particularly important:

```
n_estimators
learning_rate
```

They work together.

### Higher `n_estimators`

More boosting stages.

Potential benefit:

```
More learning
```

Potential problem:

```
Overfitting / slower training
```

---

### Lower `learning_rate`

Each stage has a smaller contribution.

Potential benefit:

```
More controlled learning
```

Potential cost:

```
Usually requires more estimators
```

---

# 32. Common Configuration Pattern

A common idea is:

```
learning_rate ↓
n_estimators ↑
```

For example:

```
learning_rate=0.1
n_estimators=500
```

versus:

```
learning_rate=0.05
n_estimators=1000
```

The second configuration takes smaller steps but more steps.

The exact best combination must be determined experimentally.

---

# 33. XGBoost vs Random Forest

Both are ensemble methods, but they work differently.

### Random Forest

Builds many trees independently and combines their predictions.

Conceptually:

```
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┼→ Average
Tree 4 ─┤
Tree 5 ─┘
```

---

### Gradient Boosting

Builds models sequentially.

Conceptually:

```
Model 1
   ↓
Correct errors
   ↓
Model 2
   ↓
Correct errors
   ↓
Model 3
   ↓
...
   ↓
Final prediction
```

---

# 34. Why XGBoost Is Powerful

XGBoost is particularly effective on **tabular/structured data**.

Examples:

- housing prices
- customer churn
- credit risk
- fraud detection
- sales prediction
- competition datasets

It is one of the important models to know for practical machine learning on structured data.

---

# ⚠️ Common Mistakes

## Mistake 1 — Using the wrong estimator

For continuous house prices, use:

```
XGBRegressor
```

not:

```
XGBClassifier
```

---

## Mistake 2 — Forgetting to fit the model

Creating:

```
XGBRegressor(...)
```

does not train it.

You still need:

```
model.fit(X_train, y_train)
```

---

## Mistake 3 — Predicting with training data when evaluating

For this exercise, evaluation is performed using:

```
X_valid
```

and:

```
y_valid
```

---

## Mistake 4 — Thinking higher MAE is better

For MAE:

```
Lower = Better
```

---

## Mistake 5 — Confusing `n_estimators`

It means the number of boosting stages/estimators.

It does not mean:

```
number of input features
```

---

## Mistake 6 — Assuming more estimators always improve the model

More estimators can help, but too many can contribute to overfitting and longer training.

---

# 🔁 When NOT to Use This Exact Setup

This exercise uses:

```
XGBRegressor
```

for a regression problem.

Do not blindly use this exact estimator for every problem.

For classification, you would use an appropriate classifier such as:

```
XGBClassifier
```

Also remember that the exercise uses a simplified preprocessing approach with Pandas `get_dummies()`. In a real production workflow, a properly designed preprocessing pipeline is generally preferable.

---

# 🧪 Code Cheat Sheet

## Basic XGBoost

```
from xgboost import XGBRegressor

model = XGBRegressor(
    random_state=0
)

model.fit(
    X_train,
    y_train
)
```

---

## Prediction

```
predictions = model.predict(
    X_valid
)
```

---

## MAE

```
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    predictions,
    y_valid
)
```

---

## Tune number of estimators

```
model = XGBRegressor(
    n_estimators=1000
)
```

---

## Tune learning rate

```
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)
```

---

# 🎤 Interview Revision

### What is gradient boosting?

An ensemble technique that builds models sequentially, with later models working to reduce the errors of the existing ensemble.

### What is XGBoost?

XGBoost, or Extreme Gradient Boosting, is a high-performance implementation of gradient boosting.

### What is `XGBRegressor`?

An XGBoost model used for regression problems.

### What does `n_estimators` control?

The number of boosting stages/estimators used in the ensemble.

### What does `learning_rate` control?

How strongly each new boosting stage contributes to the ensemble.

### What happens if `n_estimators` is too small?

The model may underfit because it doesn't have enough boosting stages to learn the patterns.

### Why was `n_estimators=1` used in Step 3?

To intentionally create a poorly performing model and demonstrate the effect of insufficient boosting stages.

### Why was `n_estimators=1000, learning_rate=0.05` used in Step 2?

To experiment with many smaller boosting updates and obtain better validation performance than the baseline.

### What metric is used?

```
Mean Absolute Error (MAE)
```

### Is lower or higher MAE better?

```
Lower is better.
```

---

# 🧠 Exercise 4 — One-Minute Revision

```
Gradient Boosting
      ↓
Sequential ensemble
      ↓
New models correct previous errors
      ↓
XGBoost
      ↓
XGBRegressor
      ↓
Train on X_train / y_train
      ↓
Predict X_valid
      ↓
Calculate MAE
      ↓
Baseline
n_estimators → default
      ↓
Improve
n_estimators=1000
learning_rate=0.05
      ↓
Make worse
n_estimators=1
```

Remember:

```
Lower MAE = Better
```

and:

```
n_estimators ↑
    → more boosting stages

learning_rate ↓
    → smaller contribution per stage
    → often paired with more estimators
```

---

# ⭐ Core Lesson

> **XGBoost is a powerful gradient-boosting algorithm for tabular data. Its performance depends strongly on hyperparameters such as `n_estimators` and `learning_rate`, so we should evaluate different configurations on validation data rather than blindly using defaults.**
