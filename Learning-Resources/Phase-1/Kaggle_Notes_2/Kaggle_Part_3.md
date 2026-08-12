# Intermediate Machine Learning — Kaggle Course Notes

# Part 3 — Gradient Boosting, XGBoost & Data Leakage

---

# 🔹 39. Ensemble Methods

## 📖 Concept

An **ensemble method** combines predictions from multiple models instead of relying on only one model.

We already saw one example in Intro to ML:

**Random Forest**

```
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┼──→ Combined Prediction
Tree 4 ──┤
Tree 5 ──┘
```

Random Forest generally works by building many decision trees and averaging their predictions.

Gradient boosting is another type of ensemble method.

---

# 🔹 40. Gradient Boosting

## 📖 Concept

**Gradient boosting** builds an ensemble **sequentially**.

Instead of creating many independent trees and averaging them, gradient boosting repeatedly adds new models that try to improve the mistakes made by the current ensemble.

Think of it as:

```
Start with a simple model
        ↓
Find what it gets wrong
        ↓
Add another model to improve those errors
        ↓
Find remaining errors
        ↓
Add another model
        ↓
      Repeat
```

Each new model tries to make the overall ensemble better.

---

## 🧠 Simple Intuition

Imagine predicting house prices.

The first model makes rough predictions:

```
Actual:     500k
Prediction: 400k
```

Error:

```
100k
```

The next model focuses on correcting this type of error.

Then another model corrects what remains.

Eventually:

```
Model 1
   +
Model 2
   +
Model 3
   +
Model 4
   ↓
Better prediction
```

---

# 🔹 41. How Gradient Boosting Works

The course describes gradient boosting as a repeated cycle.

### Step 1 — Start with an initial model

The first model may make relatively poor predictions.

That's okay.

---

### Step 2 — Make predictions

The current ensemble predicts the target values.

---

### Step 3 — Calculate the loss

We measure how wrong those predictions are using a loss function.

For example:

```
Mean Squared Error
```

---

### Step 4 — Build another model

A new model is trained to reduce the existing loss.

The new model is specifically added to improve the ensemble.

---

### Step 5 — Add the new model

Its predictions are added to the current ensemble.

---

### Step 6 — Repeat

The process continues.

```
Model 1
   ↓
Model 2
   ↓
Model 3
   ↓
Model 4
   ↓
  ...
```

Each additional model attempts to improve the overall predictions.

---

# 🔹 42. Why Is It Called "Gradient" Boosting?

The term **gradient** refers to using gradient-based optimization to determine how the new model should improve the existing ensemble.

You don't need to memorize the mathematical derivation for this course.

For practical understanding, remember:

> Gradient boosting adds models sequentially to reduce the current prediction error.

---

# 🔹 43. XGBoost

## 📖 Concept

**XGBoost** stands for:

**Extreme Gradient Boosting**

It is an implementation of gradient boosting designed with additional features focused on:

- Performance
- Speed
- Practical usability

It is extremely popular for **tabular data**.

---

## ⚙️ Practical Usage

XGBoost is commonly useful for datasets containing structured/tabular features such as:

- Customer information
- Financial data
- Sales data
- House prices
- Credit applications
- Business metrics

It is particularly well known in Kaggle competitions.

---

# 🔹 44. XGBRegressor

For regression problems, the course uses:

```
XGBRegressor
```

---

## 🧪 Basic Code

```
from xgboost import XGBRegressor

my_model = XGBRegressor()

my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
```

---

## 🧠 Code Explanation

### `XGBRegressor()`

Creates an XGBoost regression model.

### `fit()`

Trains the model.

### `predict()`

Generates predictions for new data.

The workflow is therefore similar to the scikit-learn models you've already learned.

---

# 🔹 45. Evaluating XGBoost

## 🧪 Code

```
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)

print(
    "Mean Absolute Error: "
    + str(mean_absolute_error(
        predictions,
        y_valid
    ))
)
```

---

## 🧠 Important

For regression:

```
Lower MAE = Better
```

The course's initial XGBoost model produced an MAE of approximately:

```
241,042
```

The important lesson isn't memorizing this exact number.

The important lesson is:

> XGBoost can provide strong performance on tabular datasets, but its performance depends significantly on its parameters.

---

# 🔹 46. XGBoost Parameters

Several parameters can significantly affect:

- Accuracy
- Training speed
- Overfitting
- Underfitting

The course focuses on four important parameters:

1. `n_estimators`
2. `early_stopping_rounds`
3. `learning_rate`
4. `n_jobs`

---

# 🔹 47. `n_estimators`

## 📖 Concept

`n_estimators` controls the number of models/boosting rounds in the ensemble.

Think:

```
n_estimators = number of boosting iterations
```

Higher value:

```
More models
```

Lower value:

```
Fewer models
```

---

## ⚠️ Too Few Estimators

If the number is too low:

```
Too few boosting rounds
      ↓
Model doesn't learn enough
      ↓
Underfitting
```

Both training and validation predictions can be inaccurate.

---

## ⚠️ Too Many Estimators

If the number is too high:

```
Too many boosting rounds
      ↓
Model becomes too specialized
      ↓
Overfitting
```

Training performance may become very good while performance on unseen data gets worse.

---

## 🧪 Example

```
my_model = XGBRegressor(
    n_estimators=500
)

my_model.fit(
    X_train,
    y_train
)
```

---

## 🧠 Memory Trick

```
n_estimators
      ↓
How many boosting models?
```

---

# 🔹 48. `early_stopping_rounds`

## 📖 Concept

Instead of manually trying to guess the perfect number of estimators, we can allow XGBoost to stop when validation performance stops improving.

This is called **early stopping**.

---

## 🧠 Intuition

Suppose:

```
Round 100 → improves
Round 101 → improves
Round 102 → improves
Round 103 → no improvement
Round 104 → no improvement
Round 105 → no improvement
Round 106 → no improvement
Round 107 → no improvement
```

If:

```
early_stopping_rounds = 5
```

the model can stop after five consecutive rounds without improvement.

---

## 🧪 Code

```
my_model = XGBRegressor(
    n_estimators=500
)

my_model.fit(
    X_train,
    y_train,
    early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)
```

---

## 🧠 Important Parameters

### `eval_set`

Tells XGBoost which data to use for monitoring performance.

Here:

```
X_valid
y_valid
```

are used.

### `early_stopping_rounds=5`

Allows five consecutive rounds without improvement before stopping.

---

## ⚠️ Important

You need validation data to monitor performance.

Therefore, early stopping requires an evaluation set.

---

# 🔹 49. `learning_rate`

## 📖 Concept

The learning rate controls how strongly each new model contributes to the overall ensemble.

Instead of:

```
New prediction added at full strength
```

we can think of it as:

```
New prediction × learning_rate
```

before adding it.

---

## 🧠 Intuition

Suppose:

```
learning_rate = 1
```

Each new model has a very large influence.

If:

```
learning_rate = 0.05
```

each model contributes much less.

Therefore, the model generally needs more boosting rounds to learn.

---

# 🔹 50. Learning Rate and Number of Estimators

These parameters are closely connected.

Generally:

```
Smaller learning_rate
        +
Larger n_estimators
        ↓
More gradual learning
```

This can produce better models, but training takes longer.

---

## 🧪 Example

```
my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05
)

my_model.fit(
    X_train,
    y_train,
    early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)
```

---

## 🧠 Easy Memory Trick

### `learning_rate`

```
How much does each new model contribute?
```

### `n_estimators`

```
How many models are added?
```

---

# 🔹 51. `n_jobs`

## 📖 Concept

`n_jobs` controls parallelism.

It can allow XGBoost to use multiple CPU cores during training.

---

## 🧪 Example

```
my_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4
)
```

---

## ⚙️ Practical Usage

Useful when:

- Dataset is large
- Training takes significant time
- Multiple CPU cores are available

---

## ⚠️ Important

Increasing `n_jobs` generally improves **training speed**, not model quality.

It is a performance optimization.

Don't confuse:

```
Faster training
```

with:

```
Better model
```

---

# 🔹 52. Important XGBoost Parameter Relationships

| Parameter               | Controls                   | Too Low                         | Too High                   |
| ----------------------- | -------------------------- | ------------------------------- | -------------------------- |
| `n_estimators`          | Number of boosting rounds  | Underfitting                    | Overfitting                |
| `learning_rate`         | Contribution of each model | Slower learning                 | Can learn too aggressively |
| `early_stopping_rounds` | When training stops        | May stop too early if too small | Allows more rounds         |
| `n_jobs`                | CPU parallelism            | Slower training                 | More CPU usage             |

---

# 🔥 XGBoost Mental Model

Think:

```
XGBoost
   ↓
Sequential trees
   ↓
Each new tree improves previous errors
   ↓
Many trees combined
   ↓
Strong prediction
```

Key controls:

```
n_estimators
     ↓
Number of trees/rounds

learning_rate
     ↓
Contribution of each tree

early_stopping_rounds
     ↓
Stop when validation stops improving

n_jobs
     ↓
Training parallelism
```

---

# 🔹 53. Random Forest vs Gradient Boosting

Both are ensemble methods, but they work differently.

## Random Forest

Builds many trees and combines their predictions.

Conceptually:

```
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┼──→ Average
Tree 4 ──┤
Tree 5 ──┘
```

Trees are generally built independently.

---

## Gradient Boosting

Builds trees sequentially.

```
Tree 1
  ↓
Correct errors
  ↓
Tree 2
  ↓
Correct remaining errors
  ↓
Tree 3
  ↓
  ...
```

---

## 🧠 Easy Interview Difference

**Random Forest:**

> Many trees contribute independently and their predictions are combined.

**Gradient Boosting:**

> Trees are added sequentially, with each new tree trying to improve the current ensemble.

---

# 🔹 54. Data Leakage

## 📖 Concept

**Data leakage** occurs when information is available to the model during training but would **not actually be available when making real-world predictions**.

This causes the model to appear much better than it really is.

---

## 🚨 Why Leakage Is Dangerous

A leaked model may show:

```
Excellent validation score
        ↓
Looks like a great model
        ↓
Deploy model
        ↓
Real-world performance collapses
```

Therefore, leakage can be much more dangerous than simply having a low-quality model.

---

# 🔹 55. Two Main Types of Leakage

The course identifies:

1. **Target leakage**
2. **Train-test contamination**

These are different problems.

---

# 🔹 56. Target Leakage

## 📖 Concept

Target leakage occurs when a feature contains information that would **not be available at the time the prediction is supposed to be made**.

The key question is:

> "Would this information actually be available when I need to make the prediction?"

Not:

> "Does this feature correlate strongly with the target?"

---

# 🔹 57. Target Leakage Example

Suppose we want to predict:

```
Will this patient get pneumonia?
```

Target:

```
got_pneumonia
```

Feature:

```
took_antibiotic_medicine
```

The dataset might show:

```
got_pneumonia = True
took_antibiotic_medicine = True
```

This looks like a useful predictor.

But think about the timeline.

Usually:

```
Patient gets pneumonia
        ↓
Doctor gives antibiotics
```

Therefore:

```
took_antibiotic_medicine
```

may be information created **after** the target event.

It would not be available when we need to predict whether the patient will get pneumonia.

That's target leakage.

---

# 🔥 The Most Important Target-Leakage Question

Always ask:

> **Was this feature available before the prediction target was determined?**

If the answer is no:

```
Remove the feature.
```

---

# 🔹 58. Target Leakage Can Produce Amazing Scores

This is what makes leakage dangerous.

The leaked feature can have an extremely strong relationship with the target.

Therefore the model may produce:

```
Very high accuracy
Very low error
Excellent validation score
```

Yet the model can still be useless in production.

---

# 🔹 59. How to Prevent Target Leakage

Think about **time**.

Ask:

```
What information existed
at prediction time?
```

Exclude anything that:

- Was created after the target
- Was updated after the target
- Directly depends on the target
- Is only known after the event you're trying to predict

---

# 🔹 60. Train-Test Contamination

## 📖 Concept

The second type of leakage happens when validation/test data influences the training process.

This is called **train-test contamination**.

Remember:

```
Training data
    ↓
Learn model/preprocessing
```

Validation data should remain unseen during fitting.

---

# 🔹 61. Example of Train-Test Contamination

Suppose you have:

```
X_train
X_valid
```

Before splitting, you perform preprocessing using the entire dataset.

For example:

```
imputer.fit_transform(X)
```

and only afterward:

```
train_test_split(...)
```

The imputer has now learned information from:

```
Training data
   +
Validation data
```

That means validation data influenced preprocessing.

This contaminates the validation process.

---

# 🔥 Correct Approach

First split:

```
Raw Data
   ↓
Train / Validation
   ↓
Fit preprocessing on Train
   ↓
Transform Train
   ↓
Transform Validation
   ↓
Train Model
```

---

# 🔹 62. Why This Matters

Validation is supposed to answer:

> "How well does my model perform on data it hasn't seen?"

If preprocessing has already learned information from validation data, the validation set isn't truly unseen anymore.

Therefore:

```
Validation should remain unseen
during fitting.
```

---

# 🔹 63. Pipelines Help Prevent Train-Test Contamination

This connects directly to the previous chapter.

Instead of manually doing:

```
Imputer.fit_transform(all_data)
```

we put preprocessing inside a pipeline.

Then cross-validation can apply preprocessing correctly within each training/validation split.

Conceptually:

```
Fold
  ↓
Training portion
  ↓
Fit preprocessing
  ↓
Transform training portion
  ↓
Transform validation portion
  ↓
Train model
  ↓
Validate
```

This is one of the major practical reasons to use pipelines.

---

# 🔹 64. Detecting Suspicious Model Performance

The course gives an important practical clue.

If you get an unexpectedly amazing result, don't immediately celebrate.

Ask:

> "Could there be leakage?"

For example:

```
98% accuracy
```

might be excellent.

But if the problem normally produces much lower performance, investigate.

A surprisingly high score can be a warning sign.

---

# 🔹 65. Target Leakage Detection Example

The course uses a credit-card application dataset.

The target is:

```
card
```

where:

```
1 = application accepted
0 = application rejected
```

The dataset contains features such as:

- `reports`
- `age`
- `income`
- `share`
- `expenditure`
- `owner`
- `selfempl`
- `dependents`
- `months`
- `majorcards`
- `active`

---

# 🔹 66. Initial Model

The course uses cross-validation with a Random Forest classifier.

## 🧪 Code

```
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

my_pipeline = make_pipeline(
    RandomForestClassifier(
        n_estimators=100
    )
)

cv_scores = cross_val_score(
    my_pipeline,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print(cv_scores.mean())
```

---

## 📊 Result

The model achieved approximately:

```
0.981
```

or:

```
98.1% accuracy
```

The course points out that such a high score is suspicious because extremely high accuracy is uncommon in many real-world problems.

So we investigate.

---

# 🔹 67. Finding Suspicious Features

One suspicious feature was:

```
expenditure
```

We need to understand what this variable actually represents.

The analysis compares expenditure for:

```
Card holders
Non-card holders
```

---

## 🧪 Code

```
expenditures_cardholders = X.expenditure[y]

expenditures_noncardholders = X.expenditure[~y]

print(
    (expenditures_noncardholders == 0).mean()
)

print(
    (expenditures_cardholders == 0).mean()
)
```

---

## 📊 Course Result

Approximately:

```
Non-card holders with zero expenditure:
100%

Card holders with zero expenditure:
2%
```

This is an enormous relationship.

That explains why the model performs so well.

But it also raises a serious question:

> What exactly does `expenditure` represent?

---

# 🔹 68. Why `expenditure` Is Leaky

The likely explanation is that `expenditure` refers to expenditure on the credit card being applied for.

Think about the timeline:

```
Apply for card
    ↓
Decision about approval
    ↓
Card is issued
    ↓
Expenditure occurs
```

If expenditure happens after approval, it cannot be used to predict approval.

Therefore:

```
expenditure
```

is a target-leaking feature.

---

# 🔹 69. Other Suspicious Features

The course also identifies:

```
share
active
majorcards
```

as potentially concerning.

`share` is partially determined by `expenditure`, so it can also carry leakage.

`active` and `majorcards` are less certain from the available descriptions, but the course recommends being cautious when the meaning of a feature cannot be confirmed.

---

# 🔹 70. Removing Leaky Features

The course removes:

```
expenditure
share
active
majorcards
```

---

## 🧪 Code

```
potential_leaks = [
    'expenditure',
    'share',
    'active',
    'majorcards'
]

X2 = X.drop(
    potential_leaks,
    axis=1
)

cv_scores = cross_val_score(
    my_pipeline,
    X2,
    y,
    cv=5,
    scoring='accuracy'
)

print(cv_scores.mean())
```

---

## 📊 Result

Accuracy drops from approximately:

```
0.981
```

to:

```
0.831
```

At first, this looks disappointing.

But the second model is actually more useful.

Why?

Because it is based on information that can realistically exist when the prediction is made.

---

# 🔥 The Critical Lesson

A lower score can be **better** if the higher score was caused by leakage.

Remember:

```
98% with leakage
    ❌
```

is worse than:

```
83% without leakage
    ✅
```

if the 83% model actually generalizes to real-world predictions.

---

# 🔹 71. Leakage vs Overfitting

These concepts are related but different.

## Overfitting

The model learns patterns that don't generalize well.

```
Training performance → Excellent
Validation performance → Poor
```

---

## Data Leakage

The model receives information it should not have.

```
Training performance → Excellent
Validation performance → Potentially excellent
Production performance → Poor
```

---

## 🧠 Key Difference

Overfitting is mainly about:

```
Model complexity
```

Leakage is mainly about:

```
Incorrect information entering the modeling process
```

---

# 🔹 72. Leakage Prevention Checklist

Before training a model, ask:

### Question 1

Is every feature available at prediction time?

### Question 2

Was any feature created after the target event?

### Question 3

Does any feature directly depend on the target?

### Question 4

Did validation/test data influence preprocessing?

### Question 5

Did I fit an imputer/encoder/scaler using validation or test data?

### Question 6

Am I performing feature engineering before splitting in a way that allows validation information to influence the transformation?

### Question 7

Is the model performance suspiciously high?

If yes:

```
Investigate for leakage.
```

---

# 🔥 73. Complete Intermediate ML Workflow

The entire course can now be connected.

```
Raw Data
    ↓
Explore Data
    ↓
Identify Missing Values
    ↓
Impute Missing Values
    ↓
Identify Categorical Variables
    ↓
Encode Categories
    ↓
ColumnTransformer
    ↓
Pipeline
    ↓
Model
    ↓
Cross-Validation
    ↓
Tune Model
    ↓
Check for Leakage
    ↓
Final Model
```

---

# 🔹 74. Model Choices Covered in the Course

The course builds on the models from Intro ML and introduces gradient boosting.

### Decision Tree

Simple and interpretable.

### Random Forest

Many trees combined.

### Gradient Boosting

Trees added sequentially to improve previous predictions.

### XGBoost

A powerful and optimized gradient boosting implementation.

---

# 🔥 75. Random Forest vs XGBoost

| Property            | Random Forest                  | XGBoost                                         |
| ------------------- | ------------------------------ | ----------------------------------------------- |
| Ensemble            | Yes                            | Yes                                             |
| Trees               | Many                           | Many                                            |
| Tree relationship   | Generally independent          | Sequential                                      |
| Main idea           | Combine many trees             | Correct previous errors                         |
| Tabular data        | Excellent                      | Excellent                                       |
| Tuning              | Often works well with defaults | Parameters can strongly affect performance      |
| Training            | Can be parallelized            | Can use parallelism                             |
| Overfitting control | Ensemble averaging             | Learning rate, estimators, early stopping, etc. |

---

# 🔹 76. Most Important Parameters to Remember

## Random Forest

```
n_estimators
```

Controls the number of trees.

---

## XGBoost

### `n_estimators`

Number of boosting rounds/models.

### `learning_rate`

Contribution of each new model.

### `early_stopping_rounds`

Stops training when validation performance stops improving.

### `n_jobs`

Controls parallel processing.

---

# 🧠 One-Shot Revision — Entire Course

## Missing Values

```
Drop
Impute
Impute + Missing Indicator
```

Best approach?

```
Test and compare.
```

---

## Categorical Variables

```
Drop
Ordinal Encoding
One-Hot Encoding
```

Decision:

```
Natural order?
   ↓
Yes → Ordinal
No  → One-Hot
```

---

## Pipelines

```
Preprocessing
     +
   Model
     ↓
  Pipeline
```

Benefits:

- Cleaner
- Safer
- Easier to deploy
- Better for cross-validation

---

## Cross-Validation

```
Split into folds
     ↓
Train/validate repeatedly
     ↓
Get multiple scores
     ↓
Average scores
```

Use especially when:

- Dataset is small
- Model training isn't excessively expensive

---

## Gradient Boosting

```
Initial model
     ↓
Find errors
     ↓
New model improves errors
     ↓
Repeat
     ↓
Strong ensemble
```

---

## XGBoost

```
XGBoost
   ↓
Gradient Boosting
   ↓
Sequential models
   ↓
Strong tabular-data performance
```

Important parameters:

```
n_estimators
learning_rate
early_stopping_rounds
n_jobs
```

---

## Data Leakage

Two major types:

```
Target Leakage
      +
Train-Test Contamination
```

### Target leakage

Feature contains information unavailable at prediction time.

### Train-test contamination

Validation/test data influences preprocessing or training.

---

# 🔥 77. Critical Code Patterns

## Imputation

```
imputer.fit_transform(X_train)

imputer.transform(X_valid)
```

Remember:

```
Train → fit + transform
Valid → transform
```

---

## One-Hot Encoding

```
encoder.fit_transform(X_train[categorical_cols])

encoder.transform(X_valid[categorical_cols])
```

---

## Pipeline

```
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_valid)
```

---

## Cross-Validation

```
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring='neg_mean_absolute_error'
)

mae_scores = -scores

average_mae = mae_scores.mean()
```

---

## XGBoost

```
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=4
)

model.fit(
    X_train,
    y_train,
    early_stopping_rounds=5,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)
```

---

# 🎯 78. Interview Questions You Should Be Able to Answer

### Missing Values

**What is imputation?**

Replacing missing values with estimated values.

**Why might imputation be better than dropping a column?**

Because the column may contain valuable information even if some values are missing.

---

### Categorical Variables

**What is one-hot encoding?**

Representing each category with its own binary feature.

**When would you use ordinal encoding?**

When the categories have a meaningful natural order.

---

### Pipelines

**Why use a pipeline?**

To combine preprocessing and modeling into one reproducible workflow and reduce preprocessing errors.

---

### Cross-Validation

**Why is cross-validation better than a single validation split?**

It evaluates the model across multiple subsets, producing a more reliable performance estimate.

---

### Gradient Boosting

**How does gradient boosting differ from random forest?**

Random forest combines many trees that are generally built independently, while gradient boosting builds models sequentially so later models improve the errors of earlier models.

---

### XGBoost

**What does `learning_rate` control?**

How strongly each newly added model contributes to the ensemble.

**What does `n_estimators` control?**

The number of boosting rounds/models.

**Why use early stopping?**

To stop adding models when validation performance stops improving.

---

### Data Leakage

**What is target leakage?**

Using information in a feature that would not be available when the prediction is actually made.

**What is train-test contamination?**

Allowing validation/test data to influence preprocessing or model fitting.

**Why is leakage dangerous?**

It can produce extremely good validation scores while causing poor real-world performance.

---

# 🏆 Final Mental Model

The biggest lesson of Intermediate ML is:

> **Good ML is not just choosing a powerful algorithm. It is building a correct end-to-end process.**

Think:

```
Real-World Data
      ↓
Handle Missing Values
      ↓
Encode Categories
      ↓
Build Pipeline
      ↓
Validate Properly
      ↓
Use Strong Models
      ↓
Check for Leakage
      ↓
Trust the Result
```

---

# 🔥 Final Rules to Remember

1. **Never fit preprocessing on validation/test data.**
2. **Use `fit_transform()` only for training data.**
3. **Use `transform()` for validation/test data.**
4. **Use pipelines when preprocessing becomes complicated.**
5. **Use `ColumnTransformer` when different column types need different preprocessing.**
6. **Use cross-validation when a single validation split may be unreliable.**
7. **Lower MAE is better for MAE-based regression evaluation.**
8. **Small learning rates often require more boosting rounds.**
9. **Early stopping helps find when additional boosting rounds stop helping.**
10. **Always investigate suspiciously high model performance.**
11. **Think about the timeline of information when checking for target leakage.**
12. **A lower score without leakage can be much more valuable than a higher leaked score.**

---

# 🏁 END — INTERMEDIATE MACHINE LEARNING COURSE NOTES

The course has now taken us from:

```
Basic ML
   ↓
Real-world messy data
   ↓
Preprocessing
   ↓
Pipelines
   ↓
Cross-validation
   ↓
Gradient Boosting
   ↓
XGBoost
   ↓
Leakage Prevention
```

The next stage is the **hands-on exercise notebooks**, where these concepts are applied to actual datasets.
