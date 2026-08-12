# Kaggle Intermediate Machine Learning

# Exercise 6 — Cross-Validation

---

## 🎯 Exercise Overview

This exercise teaches you how to use **cross-validation to tune a model parameter**.

The model used is:

```
RandomForestRegressor
```

The parameter being tuned is:

```
n_estimators
```

The evaluation metric is:

```
Mean Absolute Error (MAE)
```

The exercise follows this process:

```
Dataset
   ↓
Pipeline
   ↓
Cross-validation
   ↓
Try different n_estimators
   ↓
Compare MAE
   ↓
Select best value
```

The final best value in this exercise is:

```
n_estimators = 200
```

---

# 1. Dataset Preparation

The exercise uses the:

**Housing Prices Competition for Kaggle Learn Users**

dataset.

The target is:

```
SalePrice
```

---

# 2. Load the Data

The training and test datasets are loaded using Pandas.

```
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

test_data = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)
```

---

# 3. Separate the Target

Rows without a target value are removed:

```
train_data.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)
```

Then:

```
y = train_data.SalePrice
```

The target column is removed from the features:

```
train_data.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)
```

So:

```
X → Input features

y → SalePrice
```

---

# 4. Select Numerical Features

For simplicity, this exercise drops categorical variables.

Numerical columns are identified with:

```
numeric_cols = [
    cname
    for cname in train_data.columns
    if train_data[cname].dtype in [
        'int64',
        'float64'
    ]
]
```

Then:

```
X = train_data[numeric_cols].copy()

X_test = test_data[numeric_cols].copy()
```

Therefore, the model works only with numerical features.

---

# 5. Build the Pipeline

The exercise combines preprocessing and modeling into one pipeline.

The preprocessing step is:

```
SimpleImputer()
```

The model is:

```
RandomForestRegressor()
```

Code:

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    (
        'preprocessor',
        SimpleImputer()
    ),
    (
        'model',
        RandomForestRegressor(
            n_estimators=50,
            random_state=0
        )
    )
])
```

---

# 6. Why Use `SimpleImputer()`?

The dataset contains missing values.

Random Forest cannot directly handle those missing values in this exercise.

So:

```
SimpleImputer()
```

fills the missing values before the model receives the data.

Pipeline:

```
Raw X
  ↓
SimpleImputer
  ↓
Completed X
  ↓
RandomForestRegressor
```

---

# 7. Why Use a Pipeline?

Without a pipeline, you would need to manually:

1. Split data
2. Fit imputer
3. Transform training data
4. Transform validation data
5. Train model
6. Predict
7. Repeat preprocessing for every cross-validation fold

With a pipeline:

```
Pipeline(
    preprocessing
    +
    model
)
```

Cross-validation can handle the complete workflow automatically.

This is especially important because preprocessing should be performed correctly inside each cross-validation fold.

---

# 8. Cross-Validation Setup

The exercise uses:

```
cross_val_score()
```

Code:

```
from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(
    my_pipeline,
    X,
    y,
    cv=5,
    scoring='neg_mean_absolute_error'
)
```

Then:

```
print(
    "Average MAE score:",
    scores.mean()
)
```

---

# 9. Why `-1 *`?

Scikit-learn follows a convention:

> Higher scoring values are considered better.

But MAE works in the opposite direction:

```
Lower MAE = Better
```

Therefore scikit-learn provides:

```
neg_mean_absolute_error
```

The returned values are negative.

For example:

```
-100000
```

actually represents:

```
MAE = 100000
```

So we multiply by:

```
-1
```

to get the normal positive MAE.

---

# 10. Exercise Goal

The purpose of the exercise is not simply to calculate one cross-validation score.

Instead, you will use cross-validation to determine:

> **How many trees should the Random Forest contain?**

The parameter is:

```
n_estimators
```

---

# Step 1 — Write `get_score()`

The first task is to create a function:

```
get_score(n_estimators)
```

The function must:

1. Create a pipeline
2. Use `SimpleImputer()`
3. Use `RandomForestRegressor()`
4. Set `n_estimators` from the function argument
5. Use `random_state=0`
6. Perform 3-fold cross-validation
7. Calculate MAE
8. Return the average MAE

---

# 11. The Function

```
def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[
        (
            'preprocessor',
            SimpleImputer()
        ),
        (
            'model',
            RandomForestRegressor(
                n_estimators,
                random_state=0
            )
        )
    ])

    scores = -1 * cross_val_score(
        my_pipeline,
        X,
        y,
        cv=3,
        scoring='neg_mean_absolute_error'
    )

    return scores.mean()
```

---

# 12. Understand the Function

When you call:

```
get_score(100)
```

the function creates:

```
RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
```

Then it performs:

```
3-fold cross-validation
```

and returns:

```
Average MAE
```

---

# 13. Why Make a Function?

Instead of repeatedly writing:

```
create model
create pipeline
cross-validation
calculate average
```

you can simply write:

```
get_score(50)
```

or:

```
get_score(100)
```

or:

```
get_score(200)
```

This makes hyperparameter experimentation much easier.

---

# 14. Step 2 — Test Different Values

Now the exercise asks you to test eight values:

```
50
100
150
200
250
300
350
400
```

The values are generated using:

```
range(1, 9)
```

and:

```
50 * i
```

---

# 15. Store Results in a Dictionary

The exercise uses:

```
results = {}
```

Then:

```
for i in range(1, 9):
    results[50 * i] = get_score(50 * i)
```

This produces a dictionary conceptually like:

```
{
    50:  MAE,
    100: MAE,
    150: MAE,
    200: MAE,
    250: MAE,
    300: MAE,
    350: MAE,
    400: MAE
}
```

---

# 16. Why Use a Dictionary?

The dictionary stores:

```
n_estimators → MAE
```

This makes it easy to compare the performance of different parameter values.

For example:

```
50  →  190000
100 →  180000
150 →  175000
200 →  170000
```

Now we can identify the parameter producing the lowest MAE.

---

# 17. Complete Step 2

```
results = {}

for i in range(1, 9):
    results[50 * i] = get_score(
        50 * i
    )

print(results)
```

---

# 18. Visualizing the Results

The exercise then plots the results.

```
import matplotlib.pyplot as plt

plt.plot(
    list(results.keys()),
    list(results.values())
)

plt.show()
```

The x-axis represents:

```
n_estimators
```

The y-axis represents:

```
Average MAE
```

---

# 19. Why Plot the Results?

A graph makes it easier to see how model performance changes as:

```
n_estimators
```

changes.

Conceptually:

```
MAE
 │
 │\
 │ \
 │  \
 │   \__
 │      \__
 │          /
 │         /
 └──────────────── n_estimators
```

We want the lowest point.

---

# 20. Step 3 — Find the Best Parameter

The exercise asks:

> Which `n_estimators` value appears to produce the best model?

The answer is:

```
200
```

So:

```
n_estimators_best = 200
```

---

# 21. Why Is 200 the Best?

Because among the tested values:

```
50
100
150
200
250
300
350
400
```

the model with:

```
n_estimators = 200
```

gave the best cross-validation performance according to the exercise.

Remember:

```
Best MAE = Lowest MAE
```

---

# 22. Complete Exercise Workflow

The entire exercise can be remembered as:

```
X, y
  ↓
Pipeline
  ↓
SimpleImputer
  ↓
RandomForestRegressor
  ↓
Cross-validation
  ↓
get_score(n_estimators)
  ↓
Test:
  50
  100
  150
  200
  250
  300
  350
  400
  ↓
Compare MAE
  ↓
Plot results
  ↓
Best = 200
```

---

# 🧠 Important Concept — Hyperparameter

`n_estimators` is a **hyperparameter**.

It is a setting chosen before/during model training rather than something the model learns directly from the training data.

For Random Forest:

```
n_estimators
```

controls the number of trees.

---

# 23. Parameter vs Hyperparameter

### Parameter

Learned from data.

Example:

```
Tree split values
```

### Hyperparameter

Chosen by us.

Example:

```
n_estimators=200
```

We use validation or cross-validation to decide which hyperparameter values work best.

---

# 24. Cross-Validation for Hyperparameter Tuning

The exercise demonstrates a basic form of hyperparameter tuning.

Process:

```
Choose parameter
     ↓
Try value 1
     ↓
Cross-validation
     ↓
Record MAE
     ↓
Try value 2
     ↓
Cross-validation
     ↓
Record MAE
     ↓
...
     ↓
Choose best value
```

This is a simple and practical tuning strategy.

---

# 25. Why Not Just Use One Validation Set?

A single validation split can depend somewhat on which observations happened to land in the validation set.

Cross-validation uses multiple splits.

For example:

```
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
```

Each time, the model is trained on the remaining folds.

The resulting scores are averaged.

This gives a more reliable estimate.

---

# 26. Why Does This Exercise Use `cv=3`?

The `get_score()` function specifically uses:

```
cv=3
```

So each candidate `n_estimators` value is evaluated using three folds.

The earlier demonstration uses:

```
cv=5
```

but the actual tuning function uses:

```
cv=3
```

This reduces computation while still providing multiple validation measurements.

---

# 27. Why Is `random_state=0` Important?

The exercise sets:

```
random_state=0
```

for the Random Forest.

This makes the results reproducible.

If you run the same code again, you should get consistent results.

---

# 28. Important Difference: `cv` vs `n_estimators`

These are easy to confuse.

### `cv`

Controls:

```
Number of cross-validation folds
```

Example:

```
cv=3
```

means:

```
3 folds
```

---

### `n_estimators`

Controls:

```
Number of Random Forest trees
```

Example:

```
n_estimators=200
```

means:

```
200 trees
```

---

# 29. Common Mistake

Don't think:

```
cv=200
```

means 200 trees.

No.

`cv` belongs to cross-validation.

`n_estimators` belongs to Random Forest.

---

# 30. Exercise Code — Full Version

```
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Load data
train_data = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

test_data = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)

# Remove rows without target
train_data.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)

# Separate target
y = train_data.SalePrice

train_data.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)

# Select numerical columns
numeric_cols = [
    cname
    for cname in train_data.columns
    if train_data[cname].dtype in [
        'int64',
        'float64'
    ]
]

X = train_data[numeric_cols].copy()

X_test = test_data[numeric_cols].copy()

# Function for cross-validation score
def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[
        (
            'preprocessor',
            SimpleImputer()
        ),
        (
            'model',
            RandomForestRegressor(
                n_estimators,
                random_state=0
            )
        )
    ])

    scores = -1 * cross_val_score(
        my_pipeline,
        X,
        y,
        cv=3,
        scoring='neg_mean_absolute_error'
    )

    return scores.mean()

# Test different values
results = {}

for i in range(1, 9):
    results[50 * i] = get_score(
        50 * i
    )

print(results)

# Best value from the exercise
n_estimators_best = 200
```

---

# 31. What You Should Understand From This Exercise

Don't just memorize:

```
n_estimators_best = 200
```

That value belongs to this particular exercise.

The important lesson is the **process**:

```
Don't guess the hyperparameter.
```

Instead:

```
Try several values
      ↓
Evaluate using cross-validation
      ↓
Compare scores
      ↓
Select the best value
```

On another dataset, the best value could be:

```
100
```

or:

```
500
```

or:

```
1000
```

---

# 32. What Happens If You Increase `n_estimators`?

Generally:

```
More trees
   ↓
More computation
```

Increasing the number of trees can improve Random Forest performance up to a point.

But more trees do not automatically guarantee a better model.

You should evaluate the actual performance.

---

# 33. Cross-Validation + Pipeline

This exercise combines two important ideas from earlier chapters:

```
Pipeline
    +
Cross-validation
```

The pipeline handles:

```
Missing values
    ↓
Model training
```

Cross-validation handles:

```
Multiple train/validation splits
    ↓
Reliable performance estimate
```

Together:

```
Pipeline
   +
Cross-validation
   ↓
Safer model evaluation
```

---

# ⚠️ Common Mistakes

### Mistake 1 — Using training MAE instead of CV MAE

The exercise is about evaluating different parameter values using cross-validation.

---

### Mistake 2 — Forgetting the negative MAE convention

Scikit-learn uses:

```
neg_mean_absolute_error
```

So use:

```
-1 * scores
```

to convert it back to normal MAE.

---

### Mistake 3 — Forgetting to average the scores

Cross-validation returns multiple scores.

The exercise wants:

```
scores.mean()
```

---

### Mistake 4 — Testing only one parameter value

The purpose of the exercise is to compare multiple values.

---

### Mistake 5 — Assuming 200 is universally optimal

The exercise finds:

```
200
```

for this dataset and tested range.

It is not a universal Random Forest rule.

---

### Mistake 6 — Confusing `cv` with `n_estimators`

Remember:

```
cv
→ number of validation folds

n_estimators
→ number of trees
```

---

# 🔁 When NOT to Use This Exact Approach

This exercise uses manual testing:

```
50
100
150
...
400
```

For larger hyperparameter searches, manually writing every value becomes inefficient.

A better option is:

```
GridSearchCV
```

which can automatically evaluate combinations of hyperparameters.

The Kaggle exercise specifically points to `GridSearchCV()` as the next step toward more systematic hyperparameter optimization.

---

# 🧪 Code Syntax Cheat Sheet

## Create Random Forest

```
model = RandomForestRegressor(
    n_estimators=200,
    random_state=0
)
```

---

## Create Pipeline

```
pipeline = Pipeline(steps=[
    (
        'preprocessor',
        SimpleImputer()
    ),
    (
        'model',
        RandomForestRegressor(
            n_estimators=200,
            random_state=0
        )
    )
])
```

---

## Cross-validation

```
scores = -1 * cross_val_score(
    pipeline,
    X,
    y,
    cv=3,
    scoring='neg_mean_absolute_error'
)
```

---

## Average MAE

```
average_mae = scores.mean()
```

---

## Manual Parameter Search

```
results = {}

for n in [50, 100, 150, 200]:
    results[n] = get_score(n)
```

---

# 🎤 Interview Revision

### What is cross-validation?

A technique that evaluates a model using multiple train/validation splits to obtain a more reliable estimate of model performance.

### Why use cross-validation?

Because one validation split can give a noisy or unlucky estimate of model performance.

### What is a fold?

One of the subsets created when the dataset is divided for cross-validation.

### What does `cv=3` mean?

The data is divided into three folds and each fold is used as validation once.

### What is `n_estimators` in Random Forest?

The number of decision trees in the Random Forest.

### Is `n_estimators` a parameter or hyperparameter?

It is a hyperparameter selected by the practitioner.

### Why use a pipeline with cross-validation?

It ensures preprocessing such as imputation is performed properly within the cross-validation process.

### Why is `neg_mean_absolute_error` negative?

Scikit-learn's scoring convention treats larger scores as better, so loss metrics such as MAE are represented negatively.

### How do you get normal MAE?

```
-1 * scores
```

### How did the exercise select the best `n_estimators`?

It tested several values, calculated cross-validation MAE for each, and selected the value with the lowest average MAE.

### What was the best value in this exercise?

```
n_estimators = 200
```

---

# 🧠 Exercise 6 — One-Minute Revision

```
CROSS-VALIDATION
      ↓
More reliable model evaluation
      ↓
Combine with Pipeline
      ↓
SimpleImputer
      ↓
RandomForestRegressor
      ↓
Tune n_estimators
      ↓
Test:
50 → MAE
100 → MAE
150 → MAE
200 → MAE
250 → MAE
300 → MAE
350 → MAE
400 → MAE
      ↓
Compare MAEs
      ↓
Lowest MAE
      ↓
Best parameter
      ↓
200
```

Remember:

```
cv
→ number of folds

n_estimators
→ number of trees

MAE
→ lower is better

cross-validation
→ evaluate multiple splits

hyperparameter tuning
→ try values and select the best-performing one
```

---

# ⭐ Core Lesson

> **Cross-validation can be used not only to measure model quality, but also to choose better hyperparameters. Instead of guessing a value such as `n_estimators`, evaluate multiple candidates with cross-validation and select the value that gives the lowest validation error.**
