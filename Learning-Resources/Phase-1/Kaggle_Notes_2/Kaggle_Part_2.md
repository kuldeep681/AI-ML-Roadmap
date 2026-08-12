# Intermediate Machine Learning — Kaggle Course Notes

# Part 2 — Pipelines & Cross-Validation

---

# 🔹 16. Pipelines

## 📖 Concept

As preprocessing becomes more complicated, ML code can become messy.

For example, suppose your dataset contains:

- Numerical columns
- Missing numerical values
- Categorical columns
- Missing categorical values

Without a pipeline, you might have to manually:

1. Find numerical columns
2. Impute numerical values
3. Find categorical columns
4. Impute categorical values
5. One-hot encode categorical values
6. Combine everything
7. Train the model
8. Repeat the same preprocessing for validation/test data

This creates many opportunities for mistakes.

A **Pipeline** bundles preprocessing and the model together so they can be treated as one object.

---

## 🧠 Mental Model

Without a pipeline:

```
Raw Data
   ↓
Imputation
   ↓
Encoding
   ↓
Combine Data
   ↓
Model
   ↓
Prediction
```

With a pipeline:

```
Raw Data
   ↓
Pipeline
   ↓
Prediction
```

The pipeline automatically performs the required preprocessing before the model makes predictions.

---

# 🔹 17. Why Pipelines Are Useful

The Kaggle course highlights four major benefits.

## 1. Cleaner Code

Instead of manually managing every preprocessing step, the pipeline keeps them together.

---

## 2. Fewer Bugs

It reduces the chance of:

- Forgetting preprocessing
- Applying different preprocessing to train and validation data
- Processing columns incorrectly
- Accidentally changing the wrong DataFrame

---

## 3. Easier to Productionize

The same preprocessing logic can be attached to the model.

When new data arrives:

```
New Data
   ↓
Pipeline
   ↓
Preprocessing
   ↓
Model
   ↓
Prediction
```

You don't need to manually repeat every preprocessing step.

---

## 4. Better Model Validation

Pipelines become especially valuable when using **cross-validation**.

This is important because preprocessing needs to be performed correctly inside each validation split.

---

# 🔹 18. ColumnTransformer

## 📖 Concept

Different types of columns often need different preprocessing.

For example:

### Numerical data

```
Rooms
Bathroom
Landsize
YearBuilt
```

May need:

```
Missing-value imputation
```

### Categorical data

```
Type
Method
Regionname
```

May need:

```
Missing-value imputation
      ↓
One-hot encoding
```

`ColumnTransformer` allows us to apply different transformations to different groups of columns.

---

## 🧠 Mental Model

```
Dataset
   │
   ├── Numerical columns
   │       ↓
   │    Imputation
   │
   └── Categorical columns
           ↓
       Imputation
           ↓
       One-Hot Encoding
```

`ColumnTransformer` combines these branches.

---

# 🔹 19. Building Preprocessing Steps

## Numerical Transformer

The course uses:

```
numerical_transformer = SimpleImputer(
    strategy='constant'
)
```

This replaces missing numerical values with a constant value.

---

## Categorical Transformer

Categorical data requires two steps:

1. Fill missing values
2. One-hot encode categories

The course combines those steps into another pipeline:

```
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(
            strategy='most_frequent'
        )),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore'
        ))
    ]
)
```

---

## 🧠 Why `most_frequent`?

For categorical data, the course fills missing values using the most frequently occurring category.

For example:

```
House
Unit
House
NaN
House
```

The most frequent value is:

```
House
```

So the missing value becomes:

```
House
```

---

# 🔹 20. Combining Preprocessing with ColumnTransformer

## 🧪 Code

```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(
    strategy='constant'
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(
            strategy='most_frequent'
        )),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore'
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)
```

---

## 🧠 Code Explanation

### `numerical_transformer`

Defines preprocessing for numerical columns.

### `categorical_transformer`

Defines preprocessing for categorical columns.

It itself contains a small pipeline:

```
Imputation
    ↓
One-Hot Encoding
```

### `ColumnTransformer`

Connects each preprocessing method to the appropriate columns.

---

# 🔹 21. Adding the Model

Once preprocessing is defined, define the model normally.

The course uses Random Forest:

```
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
```

---

# 🔹 22. Complete Pipeline

Now combine:

```
Preprocessor
     +
Model
```

using `Pipeline`.

---

## 🧪 Code

```
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)
```

---

## 🧠 Pipeline Structure

```
Raw X
  ↓
preprocessor
  ↓
Clean numerical + encoded categorical data
  ↓
Random Forest
  ↓
Prediction
```

---

# 🔹 23. Training the Pipeline

## 🧪 Code

```
my_pipeline.fit(X_train, y_train)
```

That's it.

The pipeline automatically:

1. Fits the preprocessing using training data
2. Transforms training data
3. Fits the model

---

# 🔹 24. Predicting with the Pipeline

## 🧪 Code

```
preds = my_pipeline.predict(X_valid)
```

The pipeline automatically:

1. Applies the previously learned preprocessing
2. Transforms validation data
3. Sends transformed data to the model
4. Produces predictions

---

## 🧠 Why This Is Better

Without a pipeline, you might have to remember:

```
preprocess X_valid
then predict
```

With a pipeline:

```
pipeline.predict(X_valid)
```

The preprocessing is automatically applied.

---

# 🔹 25. Evaluating the Pipeline

## 🧪 Code

```
from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(
    y_valid,
    preds
)

print('MAE:', score)
```

---

## 📊 Course Result

The example achieved approximately:

```
MAE = 160,679
```

This shows how combining proper preprocessing with a model can improve performance.

---

# ⚠️ Pipeline Rules to Remember

### Rule 1

Fit preprocessing only using training data.

### Rule 2

Don't manually preprocess validation/test data differently.

### Rule 3

Keep preprocessing and modeling together when possible.

### Rule 4

Pipelines are particularly valuable when preprocessing becomes complicated.

---

# 🔥 Quick Revision — Pipelines

```
ColumnTransformer
    ↓
Different preprocessing
for different columns
    ↓
Pipeline
    ↓
Model
    ↓
Prediction
```

---

# 🔹 26. Cross-Validation

## 📖 Concept

In Intro to ML, we used:

```
Training data
      +
Validation data
```

The validation set gave us an estimate of model performance.

But there is a problem.

The result depends partly on **which rows happened to be placed in the validation set**.

---

# 🔹 27. Problem with a Single Validation Set

Imagine:

```
Dataset = 5,000 rows
```

You use:

```
80% → Training
20% → Validation
```

So:

```
4,000 training rows
1,000 validation rows
```

Suppose Model A performs very well on those 1,000 rows.

Does that guarantee Model A would perform equally well on another 1,000 rows?

No.

The validation score contains some randomness because it depends on the particular validation sample.

---

## 🧠 Extreme Example

Imagine only one validation row exists.

You compare two models based on their prediction for that one row.

The result could be mostly due to luck.

Therefore:

```
One validation split
    ↓
One estimate of performance
```

can sometimes be noisy.

---

# 🔹 28. What Is Cross-Validation?

**Cross-validation** runs the modeling process multiple times using different parts of the dataset as validation data.

The dataset is divided into several parts called **folds**.

For example:

```
5 folds
```

means the dataset is divided into:

```
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

---

# 🔹 29. Five-Fold Cross-Validation

Suppose we have:

```
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

We perform five experiments.

### Experiment 1

```
Validation → Fold 1
Training   → Folds 2,3,4,5
```

### Experiment 2

```
Validation → Fold 2
Training   → Folds 1,3,4,5
```

### Experiment 3

```
Validation → Fold 3
Training   → Folds 1,2,4,5
```

### Experiment 4

```
Validation → Fold 4
Training   → Folds 1,2,3,5
```

### Experiment 5

```
Validation → Fold 5
Training   → Folds 1,2,3,4
```

Every row becomes validation data exactly once.

---

# 🔹 30. Why Cross-Validation Helps

Instead of asking:

```
"How good is my model
 on this particular validation split?"
```

we get several estimates:

```
Score 1
Score 2
Score 3
Score 4
Score 5
```

Then we can calculate:

```
Average Score
```

This gives us a more reliable estimate of model performance.

---

# 🔹 31. Cross-Validation Tradeoff

Cross-validation is more reliable, but it requires more computation.

With five folds:

```
Model trained/evaluated ≈ 5 times
```

Therefore:

### Small dataset

Cross-validation is often worth using.

### Large dataset

A single validation set may be sufficient because:

- There is already plenty of training data
- The validation estimate is less sensitive to which rows are selected
- Cross-validation would take more computation

---

# 🔥 Practical Rule from the Course

If your model takes only a couple of minutes or less to train, it is probably worth considering cross-validation.

There is no universal numerical threshold for what counts as a "small" dataset.

---

# 🔹 32. Cross-Validation with a Pipeline

Pipelines and cross-validation work extremely well together.

The course creates a pipeline containing:

```
Imputation
   ↓
Random Forest
```

---

## 🧪 Code

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(
    steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(
            n_estimators=50,
            random_state=0
        ))
    ]
)
```

---

# 🔹 33. `cross_val_score()`

Scikit-learn provides:

```
cross_val_score()
```

to perform cross-validation.

---

## 🧪 Code

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

---

## 🧠 Code Explanation

### `my_pipeline`

The complete preprocessing + model workflow.

### `X`

Features.

### `y`

Target.

### `cv=5`

Use five folds.

### `scoring`

Defines the evaluation metric.

Here:

```
neg_mean_absolute_error
```

---

# 🔹 34. Why Is MAE Negative?

This is a common source of confusion.

Normally:

```
MAE = positive error
```

Lower MAE is better.

But scikit-learn follows a convention where scoring functions are designed so:

```
Higher score = Better
```

Therefore it represents MAE as:

```
Negative MAE
```

For example:

```
-200,000
-250,000
```

The higher score is:

```
-200,000
```

which corresponds to the lower actual MAE:

```
200,000
```

The course therefore multiplies by `-1`:

```
scores = -1 * cross_val_score(...)
```

to convert the results back to normal MAE values.

---

# 🔹 35. Cross-Validation Scores

The course example produced approximately:

```
301,629
303,164
287,298
236,062
260,383
```

Each number represents the MAE from one fold.

---

## 🧪 Average Score

```
print(scores.mean())
```

The course obtained approximately:

```
277,707
```

This becomes the overall cross-validation MAE.

---

# 🔹 36. Why Average the Scores?

Suppose:

```
Fold 1 → 300k
Fold 2 → 280k
Fold 3 → 250k
Fold 4 → 290k
Fold 5 → 270k
```

The individual scores tell us how the model behaved on different subsets.

The average gives us one number that can be used to compare models.

---

# 🔹 37. Cross-Validation vs Train/Validation Split

| Single Validation           | Cross-Validation                       |
| --------------------------- | -------------------------------------- |
| One validation set          | Multiple validation folds              |
| Faster                      | Slower                                 |
| More dependent on one split | More reliable estimate                 |
| Good for large datasets     | Especially useful for smaller datasets |
| One model evaluation        | Multiple model evaluations             |

---

# 🔥 When Should You Use Cross-Validation?

## Prefer Cross-Validation When:

- Dataset is relatively small
- You have enough computation
- You are comparing several models
- You are making many modeling decisions
- You want a more reliable performance estimate

---

## Prefer a Single Validation Set When:

- Dataset is very large
- Model training is expensive
- Cross-validation would take too long
- A single validation set already provides a reliable estimate

---

# 🔹 38. Cross-Validation + Pipeline = Powerful Combination

This is one of the most important concepts from these chapters.

Instead of:

```
Preprocess
   ↓
Split
   ↓
Train
   ↓
Validate
   ↓
Repeat manually
```

we can use:

```
Pipeline
   +
Cross-Validation
```

The pipeline ensures preprocessing is performed correctly within the modeling process.

---

# ⚠️ Critical Interview Point

Cross-validation does **not** magically prevent data leakage.

Preprocessing must still be handled correctly.

This is one reason pipelines are so valuable.

If preprocessing is performed incorrectly before cross-validation, information can leak between folds.

---

# 🧠 Part 2 One-Shot Revision

## Pipeline

```
Preprocessing + Model
        ↓
     Pipeline
```

Benefits:

- Cleaner code
- Fewer bugs
- Easier deployment
- Works well with cross-validation

---

## ColumnTransformer

Used when different columns require different preprocessing.

```
Numerical → Imputation

Categorical → Imputation → One-Hot
```

---

## Cross-Validation

```
Dataset
   ↓
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
   ↓
Average Score
```

---

# 🎯 Interview Recall

### What is a pipeline?

A way to bundle preprocessing and modeling steps into a single workflow.

### Why use a pipeline?

To make code cleaner, reduce preprocessing mistakes, simplify productionization, and work effectively with cross-validation.

### What does `ColumnTransformer` do?

It applies different preprocessing operations to different groups of columns.

### What is cross-validation?

A technique that repeatedly trains and validates a model on different subsets of the dataset.

### Why use cross-validation?

To obtain a more reliable estimate of model performance than relying on one validation split.

### What does `cv=5` mean?

Use five folds for cross-validation.

### Why is MAE negative in `cross_val_score()`?

Because scikit-learn's scoring convention treats higher scores as better, so MAE is represented as negative MAE.

---

# 🔥 Core Mental Model

Intermediate ML so far:

```
Raw Dataset
    ↓
Missing Values
    ↓
Imputation
    ↓
Categorical Variables
    ↓
Encoding
    ↓
ColumnTransformer
    ↓
Pipeline
    ↓
Cross-Validation
    ↓
Reliable Model Evaluation
```

The next part moves into one of the most powerful tabular ML techniques covered by this course:

**Gradient Boosting and XGBoost.**
