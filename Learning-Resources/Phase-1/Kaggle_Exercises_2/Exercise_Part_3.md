# Kaggle Intermediate Machine Learning

# Exercise 3 — Cross-Validation

---

## 🎯 Exercise Overview

This exercise teaches you how to use **cross-validation** to evaluate and tune a machine learning model.

You will:

- Build a machine learning pipeline
- Use `SimpleImputer`
- Use `RandomForestRegressor`
- Apply `cross_val_score()`
- Create a reusable scoring function
- Test different values of `n_estimators`
- Compare model performance
- Visualize the results
- Select the best parameter value

The main goal is:

> **Use cross-validation to select a suitable value for a model parameter.**

---

# 1. Dataset

This exercise continues using the **Housing Prices Competition for Kaggle Learn Users** dataset.

The target variable is:

```
SalePrice
```

The exercise uses only numerical columns.

---

# 2. Loading the Data

The training and test data are loaded:

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

Rows where `SalePrice` is missing are removed:

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

The target is removed from the feature dataset:

```
train_data.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)
```

So:

```
X → Features
y → Target
```

---

# 4. Select Numerical Columns

The exercise keeps only numerical variables.

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

Categorical variables are intentionally excluded to keep the exercise focused on cross-validation.

---

# 5. Why Cross-Validation?

Normally, we might split our dataset into:

```
Training Data
      ↓
   Model
      ↓
Validation Data
      ↓
  Evaluation
```

But there is a problem.

A single validation split can sometimes give an unreliable estimate of model performance.

The result can depend on exactly which observations ended up in the validation set.

Cross-validation addresses this by repeatedly splitting the data into different training and validation portions.

---

# 6. Understanding K-Fold Cross-Validation

Suppose:

```
cv = 5
```

The dataset is divided into 5 folds.

Conceptually:

```
Fold 1 → Validation
Folds 2-5 → Training
```

Then:

```
Fold 2 → Validation
Folds 1,3,4,5 → Training
```

Then:

```
Fold 3 → Validation
Others → Training
```

And so on.

Every observation gets a chance to be part of the validation set.

---

# 🧠 Why Is This Useful?

Instead of getting one score:

```
MAE = 18,000
```

you get several scores:

```
Fold 1 → MAE
Fold 2 → MAE
Fold 3 → MAE
Fold 4 → MAE
Fold 5 → MAE
```

Then calculate the average:

```
Average MAE
    =
(Fold1 + Fold2 + Fold3 + Fold4 + Fold5) / 5
```

This generally gives a more reliable estimate of model performance than relying on a single split.

---

# 7. Building the Pipeline

The exercise uses a pipeline containing two steps:

1. Missing-value preprocessing
2. Random Forest model

Import:

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
```

Create the pipeline:

```
my_pipeline = Pipeline(
    steps=[
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
    ]
)
```

---

# 8. Understanding the Pipeline

The pipeline is:

```
Raw Data
   ↓
SimpleImputer
   ↓
RandomForestRegressor
   ↓
Predictions
```

The imputer handles missing values before the Random Forest receives the data.

This is especially useful with cross-validation because the preprocessing can be applied separately inside each fold.

---

# 9. Why Use a Pipeline With Cross-Validation?

This is a very important concept.

Suppose we manually impute the entire dataset before cross-validation.

Then information from the validation folds could influence the preprocessing.

That can cause **data leakage**.

A pipeline avoids this problem by making preprocessing part of the model workflow.

Conceptually:

```
Fold 1
   ↓
Fit imputer on training portion
   ↓
Transform validation portion
   ↓
Train model
   ↓
Evaluate
```

Then the same process happens independently for the other folds.

---

# 10. Using `cross_val_score()`

Import:

```
from sklearn.model_selection import cross_val_score
```

The exercise uses:

```
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

# 11. Why Is There a `-1`?

This is an important scikit-learn detail.

The exercise wants to measure:

```
Mean Absolute Error
```

For MAE:

```
Lower = Better
```

But scikit-learn's scoring convention treats higher scores as better.

Therefore scikit-learn provides:

```
neg_mean_absolute_error
```

The returned values are negative.

For example:

```
-18000
-17000
-19000
```

The exercise multiplies them by `-1`:

```
-1 * scores
```

giving:

```
18000
17000
19000
```

Now we can interpret them normally:

```
Lower MAE = Better
```

---

# 12. Understanding `cv=5`

This:

```
cv=5
```

means the exercise uses **5-fold cross-validation**.

The dataset is divided into five parts.

Each part gets used as validation once.

---

# 13. Step 1 — Write a Useful Function

The main task begins with creating:

```
get_score()
```

The function receives:

```
n_estimators
```

This parameter controls the number of trees in the Random Forest.

---

## Function

```
def get_score(n_estimators):

    my_pipeline = Pipeline(
        steps=[
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
        ]
    )

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

# 14. Understanding `get_score()`

The function performs several operations.

### Step 1

Receive the number of trees:

```
n_estimators
```

### Step 2

Create a pipeline:

```
SimpleImputer
    ↓
RandomForestRegressor
```

### Step 3

Perform 3-fold cross-validation:

```
cv=3
```

### Step 4

Calculate MAE for every fold.

### Step 5

Convert negative MAE into positive MAE:

```
-1 *
```

### Step 6

Return the average MAE:

```
scores.mean()
```

---

# 15. Why Use a Function?

Without a function, you would have to repeat the same code for every value:

```
50 trees
100 trees
150 trees
200 trees
...
```

Instead:

```
get_score(50)

get_score(100)

get_score(150)

get_score(200)
```

This makes experimentation much easier.

---

# 16. Step 2 — Test Different Parameter Values

The exercise wants to test eight values for `n_estimators`:

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

and multiplied by:

```
50
```

---

# 17. Storing the Results

The exercise uses a dictionary:

```
results = {}
```

Then:

```
for i in range(1, 9):
    results[50 * i] = get_score(50 * i)
```

Finally:

```
print(results)
```

The dictionary structure is:

```
{
    number_of_trees: average_MAE
}
```

For example:

```
{
    50:  ...,
    100: ...,
    150: ...,
    200: ...,
    ...
}
```

---

# 🧠 Why Use a Dictionary?

It keeps the relationship between the parameter and its performance score.

For example:

```
results[200]
```

means:

> What was the average MAE when `n_estimators = 200`?

This makes it easy to compare the tested values.

---

# 18. Step 2 — Visualizing the Results

The exercise uses Matplotlib.

```
import matplotlib.pyplot as plt

%matplotlib inline
```

Then:

```
plt.plot(
    list(results.keys()),
    list(results.values())
)

plt.show()
```

---

# 19. Understanding the Graph

The graph has:

### X-axis

Number of trees:

```
n_estimators
```

### Y-axis

Average MAE from cross-validation.

The objective is:

```
Find the lowest point
```

because:

```
Lower MAE = Better
```

---

# 20. What Happens When We Increase `n_estimators`?

`n_estimators` controls how many decision trees are inside the Random Forest.

For example:

```
n_estimators=50
```

means:

```
50 trees
```

while:

```
n_estimators=400
```

means:

```
400 trees
```

Increasing the number of trees can improve model stability, but it also increases computation.

Therefore, we don't simply assume:

```
More trees = Always better
```

Instead, we test different values.

---

# 21. Step 3 — Find the Best Parameter

The exercise asks:

> Which value of `n_estimators` appears to give the best performance?

The solution is:

```
n_estimators_best = 200
```

So the selected value is:

```
n_estimators = 200
```

---

# 22. Why Is 200 Considered Best?

The exercise compares the average MAE for the tested values.

Because:

```
Lower MAE = Better
```

we select the value that produces the lowest validation error among the tested values.

The exercise identifies:

```
200
```

as the best value.

---

# 23. Complete Exercise Logic

The entire exercise can be understood as:

```
Load data
    ↓
Separate X and y
    ↓
Keep numerical columns
    ↓
Build Pipeline
    ↓
SimpleImputer
    ↓
RandomForestRegressor
    ↓
Cross-validation
    ↓
Calculate average MAE
    ↓
Create get_score()
    ↓
Test different n_estimators
    ↓
Store results
    ↓
Visualize results
    ↓
Choose best parameter
    ↓
n_estimators = 200
```

---

# 24. Important Concepts From Exercise 3

## Cross-Validation

A technique for evaluating a model using multiple train/validation splits.

---

## Fold

One portion of the dataset used as the validation set during a particular round of cross-validation.

---

## `cv`

Controls the number of cross-validation folds.

Example:

```
cv=5
```

means:

```
5-fold cross-validation
```

---

## `cross_val_score()`

Scikit-learn function used to evaluate a model using cross-validation.

Basic pattern:

```
cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring=...
)
```

---

## `Pipeline`

Combines preprocessing and model training into a single workflow.

Example:

```
Pipeline(
    steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(...))
    ]
)
```

---

## `n_estimators`

Number of trees in the Random Forest.

Example:

```
RandomForestRegressor(
    n_estimators=200
)
```

means:

```
Random Forest
    ↓
200 decision trees
```

---

# 25. Cross-Validation vs Train/Validation Split

### Single validation split

```
Dataset
   ↓
┌──────────┬────────────┐
│ Training │ Validation │
└──────────┴────────────┘
```

You get one validation score.

---

### Cross-validation

```
Dataset
   ↓
┌────┬────┬────┬────┬────┐
│ F1 │ F2 │ F3 │ F4 │ F5 │
└────┴────┴────┴────┴────┘
```

Validation rotates:

```
Round 1 → F1 validation
Round 2 → F2 validation
Round 3 → F3 validation
Round 4 → F4 validation
Round 5 → F5 validation
```

Then:

```
Average all scores
```

---

# 26. Advantages of Cross-Validation

### 1. More reliable evaluation

The result isn't dependent on one particular validation split.

### 2. Better use of available data

Every observation gets used for both training and validation across different folds.

### 3. Useful for model selection

You can compare:

```
Model A
Model B
Model C
```

using the same cross-validation procedure.

### 4. Useful for hyperparameter tuning

For example:

```
n_estimators = 50
n_estimators = 100
n_estimators = 150
n_estimators = 200
```

and select the best-performing value.

---

# 27. Disadvantages of Cross-Validation

Cross-validation is not free.

If you use:

```
cv=5
```

the model is trained multiple times.

If you test:

```
8 different n_estimators values
```

you perform many model training operations.

Therefore:

```
More reliable evaluation
       +
More computation
```

This is why cross-validation can be expensive for large datasets or complex models.

---

# 28. Hyperparameter Tuning

The exercise is actually introducing **hyperparameter optimization**.

A hyperparameter is a model setting that is chosen before training.

For Random Forest:

```
n_estimators
```

is a hyperparameter.

We test:

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

Then choose the best-performing value.

---

# 29. Grid Search

The exercise briefly introduces **Grid Search** as the next step beyond manually testing one parameter.

Instead of manually doing:

```
get_score(50)
get_score(100)
get_score(150)
...
```

you can use:

```
GridSearchCV()
```

to search through combinations of parameters.

Conceptually:

```
Parameter values
      ↓
GridSearchCV
      ↓
Cross-validation
      ↓
Compare combinations
      ↓
Best parameters
```

---

# 30. Manual Tuning vs GridSearchCV

### Manual approach

```
for value in parameter_values:
    score = get_score(value)
```

You control everything yourself.

### Grid Search

```
GridSearchCV(...)
```

Scikit-learn automates the search over parameter combinations.

The exercise recommends learning Grid Search as the next step for hyperparameter optimization.

---

# ⚠️ Common Mistakes

## Mistake 1 — Forgetting the negative MAE

Scikit-learn uses:

```
neg_mean_absolute_error
```

Therefore:

```
scores = -1 * scores
```

is used to convert it back to normal MAE.

---

## Mistake 2 — Choosing the highest MAE

For MAE:

```
Lower = Better
```

So choose:

```
minimum MAE
```

not maximum.

---

## Mistake 3 — Confusing `n_estimators` with tree depth

`n_estimators` means:

```
Number of trees
```

It does not mean:

```
Maximum depth of each tree
```

---

## Mistake 4 — Thinking more trees automatically means better

More trees can help, but the exercise specifically tests different values because the best parameter should be determined empirically.

---

## Mistake 5 — Forgetting the pipeline

When preprocessing is required, putting preprocessing and modeling into a pipeline is especially useful with cross-validation.

---

# 🧪 Key Code to Remember

## Create pipeline

```
my_pipeline = Pipeline(
    steps=[
        ('preprocessor', SimpleImputer()),
        (
            'model',
            RandomForestRegressor(
                n_estimators=50,
                random_state=0
            )
        )
    ]
)
```

---

## Cross-validation

```
scores = -1 * cross_val_score(
    my_pipeline,
    X,
    y,
    cv=5,
    scoring='neg_mean_absolute_error'
)
```

---

## Average score

```
scores.mean()
```

---

## Test multiple values

```
results = {}

for i in range(1, 9):
    results[50 * i] = get_score(50 * i)
```

---

## Select best parameter

```
n_estimators_best = 200
```

---

# 🎤 Interview Revision

### What is cross-validation?

A technique that evaluates a model across multiple train/validation splits instead of relying on one fixed validation split.

### What is a fold?

One subset of the dataset used as validation data during a particular cross-validation iteration.

### What does `cv=5` mean?

Five-fold cross-validation.

### Why use cross-validation?

To obtain a more reliable estimate of model performance and help compare models or hyperparameters.

### What does `n_estimators` mean in Random Forest?

The number of decision trees in the forest.

### Why multiply the result of `cross_val_score()` by `-1`?

Because scikit-learn represents MAE as `neg_mean_absolute_error`, following its convention that larger scores are better.

### For MAE, is higher or lower better?

Lower is better.

### What parameter was tuned in this exercise?

```
n_estimators
```

### Which values were tested?

```
50, 100, 150, 200, 250, 300, 350, 400
```

### Which value was selected?

```
200
```

### What can automate hyperparameter search?

```
GridSearchCV()
```

---

# 🧠 Exercise 3 — One-Minute Revision

```
Cross-validation
      ↓
Split data into folds
      ↓
Train + validate repeatedly
      ↓
Calculate score for every fold
      ↓
Average the scores
      ↓
Use average score for comparison
      ↓
Tune hyperparameters
      ↓
Test n_estimators
      ↓
50 → 100 → 150 → ... → 400
      ↓
Compare MAE
      ↓
Lower MAE = Better
      ↓
Best value = 200
```

---

# ⭐ Core Lesson

> **Cross-validation gives a more reliable estimate of model performance by evaluating the model across multiple train/validation splits. It can also be used to tune hyperparameters by testing different parameter values and selecting the one that performs best.**

The key pattern to remember is:

```
Pipeline
    ↓
Cross-validation
    ↓
Average validation score
    ↓
Compare hyperparameters
    ↓
Select the best value
```
