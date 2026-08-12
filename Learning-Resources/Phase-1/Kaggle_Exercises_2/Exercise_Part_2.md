# Kaggle Intermediate Machine Learning

# Exercise 2 — Missing Values

---

## 🎯 Exercise Overview

This exercise applies the concepts from the **Missing Values** chapter to the Kaggle Housing Prices dataset.

The exercise teaches you how to:

- Inspect missing values
- Calculate the amount of missing data
- Decide whether dropping columns is reasonable
- Remove columns containing missing values
- Use `SimpleImputer`
- Compare different missing-value strategies using MAE
- Add missing-value indicator columns
- Prepare training, validation, and test data
- Train a Random Forest model
- Generate predictions
- Create a Kaggle submission

---

# 1. Dataset Setup

The exercise uses the **Housing Prices Competition for Kaggle Learn Users** dataset.

Two datasets are loaded:

```
train.csv
test.csv
```

The training dataset contains the target:

```
SalePrice
```

The test dataset does not contain `SalePrice`.

---

## Loading the Data

```
import pandas as pd
from sklearn.model_selection import train_test_split

X_full = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

X_test_full = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)
```

---

## Separating the Target

Rows with missing target values are removed:

```
X_full.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)
```

The target is then extracted:

```
y = X_full.SalePrice
```

And removed from the feature dataset:

```
X_full.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)
```

So now:

```
X_full → input features
y      → target
```

---

# 2. Selecting Numerical Features

To keep the exercise focused on missing values, only numerical predictors are used.

```
X = X_full.select_dtypes(
    exclude=['object']
)

X_test = X_test_full.select_dtypes(
    exclude=['object']
)
```

This removes categorical/text columns.

The resulting training data contains:

```
36 columns
```

---

# 3. Train/Validation Split

The training data is divided into training and validation sets.

```
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)
```

### Result

The training portion contains:

```
1168 rows
```

The validation portion contains the remaining 20%.

### Why split?

The model should not be evaluated only on the data it was trained on.

Instead:

```
Training data
      ↓
   Learn

Validation data
      ↓
   Evaluate
```

---

# 4. Preliminary Investigation

The first task is to investigate the missing values.

```
print(X_train.shape)
```

Output:

```
(1168, 36)
```

This means:

```
1168 rows
36 columns
```

---

## Counting Missing Values

The exercise uses:

```
missing_val_count_by_column = X_train.isnull().sum()

print(
    missing_val_count_by_column[
        missing_val_count_by_column > 0
    ]
)
```

Output:

```
LotFrontage    212
MasVnrArea       6
GarageYrBlt     58
```

So only three columns contain missing values.

---

# 5. Exercise Question — Preliminary Investigation

The exercise asks three questions.

### Question 1

How many rows are in the training data?

Answer:

```
1168
```

---

### Question 2

How many columns contain missing values?

Answer:

```
3
```

Those columns are:

```
LotFrontage
MasVnrArea
GarageYrBlt
```

---

### Question 3

How many missing entries exist in total?

Calculate:

```
212 + 6 + 58
```

Therefore:

```
276
```

---

# 🧠 What This Tells Us

The missing values are relatively limited.

The largest amount of missing data is:

```
LotFrontage → 212 / 1168
```

This is less than 20% of the training rows.

Therefore, completely deleting these columns would throw away a significant amount of potentially useful information.

This suggests that **imputation may be worth trying**.

However, the exercise later demonstrates an important lesson:

> Your intuition about which strategy should work best does not guarantee the validation results.

---

# 6. Scoring Function

To compare different preprocessing approaches, the exercise defines a common scoring function.

```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(
    X_train,
    X_valid,
    y_train,
    y_valid
):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=0
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_valid
    )

    return mean_absolute_error(
        y_valid,
        preds
    )
```

---

## What Does `score_dataset()` Do?

It performs three operations.

### Step 1 — Create Random Forest

```
RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
```

### Step 2 — Train

```
model.fit(
    X_train,
    y_train
)
```

### Step 3 — Predict and calculate MAE

```
preds = model.predict(X_valid)

mean_absolute_error(
    y_valid,
    preds
)
```

The same function is used to compare different missing-value approaches.

This makes the comparison fair.

---

# 7. Step 2 — Drop Columns With Missing Values

The first approach is simple:

> If a column contains missing values, remove the entire column.

---

## Finding Columns With Missing Values

```
cols_with_missing = [
    col
    for col in X_train.columns
    if X_train[col].isnull().any()
]
```

The result is:

```
[
    'LotFrontage',
    'MasVnrArea',
    'GarageYrBlt'
]
```

---

## Removing the Columns

Training data:

```
reduced_X_train = X_train.drop(
    cols_with_missing,
    axis=1
)
```

Validation data:

```
reduced_X_valid = X_valid.drop(
    cols_with_missing,
    axis=1
)
```

---

# 🧠 Why Drop the Same Columns?

The training and validation datasets must have the same feature structure.

Correct:

```
X_train
   ↓
Remove A, B, C

X_valid
   ↓
Remove A, B, C
```

Incorrect:

```
X_train
   ↓
Remove A, B, C

X_valid
   ↓
Keep A, B, C
```

The model needs consistent features.

---

# 8. Exercise 2 — Drop Columns Result

The exercise calculates:

```
score_dataset(
    reduced_X_train,
    reduced_X_valid,
    y_train,
    y_valid
)
```

Result:

```
MAE = 17837.82570776256
```

Approximately:

```
MAE ≈ 17,838
```

### Interpretation

The model's predictions are off by roughly $17,838 on average on this validation set.

Remember:

```
Lower MAE = Better
```

---

# 9. Step 3 — Imputation

Instead of removing columns, we can replace missing values.

This process is called:

**Imputation**

The exercise uses:

```
SimpleImputer
```

Import:

```
from sklearn.impute import SimpleImputer
```

Create the imputer:

```
my_imputer = SimpleImputer()
```

---

# 10. Applying Imputation

Training data:

```
imputed_X_train = pd.DataFrame(
    my_imputer.fit_transform(X_train)
)
```

Validation data:

```
imputed_X_valid = pd.DataFrame(
    my_imputer.transform(X_valid)
)
```

---

# ⭐ Important: `fit_transform()` vs `transform()`

This is one of the most important concepts in the exercise.

### Training data

Use:

```
fit_transform()
```

Because the imputer must:

1. Learn the replacement values
2. Apply those replacement values

So:

```
fit_transform()
    =
fit + transform
```

---

### Validation data

Use:

```
transform()
```

Because the replacement values have already been learned from the training data.

Do not fit the imputer again on validation data.

---

# 11. Why Not `fit_transform(X_valid)`?

Suppose the imputer calculates the mean using validation data.

Then information from the validation set has influenced preprocessing.

That violates the basic principle that validation data should remain unseen during training/preprocessing.

Correct:

```
Training
    ↓
fit_transform()

Validation
    ↓
transform()

Test
    ↓
transform()
```

---

# 12. Restoring Column Names

`fit_transform()` returns an array, so the DataFrame column names are lost.

Therefore the exercise restores them:

```
imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns
```

Now the processed DataFrames have the correct feature names.

---

# 13. Exercise 3 — Imputation Result

The exercise calculates:

```
score_dataset(
    imputed_X_train,
    imputed_X_valid,
    y_train,
    y_valid
)
```

Result:

```
MAE = 18062.894611872147
```

Approximately:

```
MAE ≈ 18,063
```

---

# 14. Comparing Drop vs Imputation

| Approach        |       MAE |
| --------------- | --------: |
| Drop columns    | 17,837.83 |
| Mean imputation | 18,062.89 |

Therefore:

```
Drop columns
    ↓
Better validation MAE
```

This is interesting because the tutorial suggested that imputation would often perform better.

But in this exercise:

```
Drop columns performed better.
```

---

# 🧠 Important Lesson

Never assume:

> "Imputation is always better than dropping."

The correct approach is:

```
Try different approaches
        ↓
Evaluate them
        ↓
Compare validation performance
        ↓
Choose based on evidence
```

---

# 15. Step 3 Part B — Missing Indicators

The exercise then tests an extension of imputation.

The idea is:

> Replace the missing value AND tell the model that the original value was missing.

---

## Copy the Data

```
X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()
```

Copies are created so that the original datasets are not modified.

---

# 16. Creating Missing Indicator Columns

The exercise loops through all columns containing missing values.

```
for col in cols_with_missing:

    X_train_plus[
        col + '_was_missing'
    ] = X_train_plus[col].isnull()

    X_valid_plus[
        col + '_was_missing'
    ] = X_valid_plus[col].isnull()
```

For example:

```
GarageYrBlt
```

gets an additional column:

```
GarageYrBlt_was_missing
```

---

# 17. What Does the Indicator Contain?

Suppose:

```
GarageYrBlt
```

contains:

```
2005
NaN
1998
NaN
```

The new column becomes conceptually:

```
GarageYrBlt_was_missing

False
True
False
True
```

So the model receives two pieces of information:

```
GarageYrBlt
    +
Was GarageYrBlt originally missing?
```

---

# 🧠 Why Can This Help?

Sometimes missingness itself has meaning.

For example:

```
GarageYrBlt = missing
```

could potentially indicate:

```
House has no garage
```

The missing indicator gives the model a chance to learn that relationship.

---

# 18. Impute the Extended Dataset

After adding the missing indicators, the original missing values still need to be handled.

```
my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(
    my_imputer.fit_transform(
        X_train_plus
    )
)

imputed_X_valid_plus = pd.DataFrame(
    my_imputer.transform(
        X_valid_plus
    )
)
```

Again:

```
Training → fit_transform()

Validation → transform()
```

---

# 19. Restore Column Names

```
imputed_X_train_plus.columns = (
    X_train_plus.columns
)

imputed_X_valid_plus.columns = (
    X_valid_plus.columns
)
```

---

# 20. Exercise 3 Part B Result

The exercise produces:

```
MAE = 18148.417180365297
```

Approximately:

```
MAE ≈ 18,148
```

Comparison:

| Approach                       |       MAE |
| ------------------------------ | --------: |
| Drop columns                   | 17,837.83 |
| Mean imputation                | 18,062.89 |
| Imputation + missing indicator | 18,148.42 |

Therefore, in this particular exercise:

```
Drop columns
    ↓
Best MAE
```

---

# 21. Why Did Dropping Perform Better?

This is an important exercise question.

There are several possible explanations supported by the exercise:

### Reason 1 — Dataset noise

The difference may partly be caused by noise in the dataset.

### Reason 2 — Mean imputation may not be the best strategy

The mean may not be the most suitable replacement for these particular columns.

For example:

```
GarageYrBlt
```

represents the year a garage was built.

If the value is missing, it could potentially mean that the house does not have a garage.

Simply replacing the missing value with the mean may not represent the underlying meaning.

---

# 22. Other Possible Imputation Strategies

The exercise suggests that other approaches could be tested.

For example:

### Fill with zero

```
SimpleImputer(
    strategy='constant',
    fill_value=0
)
```

But for something like `GarageYrBlt`, zero may be a poor representation.

### Fill with most frequent value

```
SimpleImputer(
    strategy='most_frequent'
)
```

### Fill with median

```
SimpleImputer(
    strategy='median'
)
```

The important point is:

> The best imputation strategy depends on the meaning and distribution of the feature.

---

# 23. Step 4 — Generate Test Predictions

The final step allows you to choose **any missing-value strategy**.

The exercise only requires that:

- Training data has no missing values
- Validation data has no missing values
- Training and validation have the same number of columns
- Number of training rows matches `y_train`
- Number of validation rows matches `y_valid`

---

# 24. Final Preprocessing Used in the Exercise

The completed notebook uses median imputation:

```
from sklearn.impute import SimpleImputer

final_imputer = SimpleImputer(
    strategy='median'
)
```

Then:

```
final_X_train = pd.DataFrame(
    final_imputer.fit_transform(
        X_train
    )
)

final_X_valid = pd.DataFrame(
    final_imputer.transform(
        X_valid
    )
)
```

Column names are restored:

```
final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns
```

---

# 25. Why Median?

The final exercise chooses:

```
strategy='median'
```

instead of the default mean.

The median can be more robust when data contains extreme values.

Example:

```
10
20
30
10000
```

The mean is heavily affected by `10000`.

The median remains close to the central values.

---

# 26. Train the Random Forest

The model is:

```
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
```

Train it:

```
model.fit(
    final_X_train,
    y_train
)
```

---

# 27. Validate the Final Model

Generate predictions:

```
preds_valid = model.predict(
    final_X_valid
)
```

Calculate MAE:

```
mean_absolute_error(
    y_valid,
    preds_valid
)
```

The exercise gets:

```
MAE = 17791.59899543379
```

Approximately:

```
MAE ≈ 17,792
```

---

# 28. Comparing the Final Result

The final median-imputation model achieves:

```
≈ 17,792 MAE
```

This is slightly better than the simple column-dropping result:

```
≈ 17,838 MAE
```

So although mean imputation performed worse than dropping columns, the chosen **median imputation** performs slightly better.

This demonstrates why testing different preprocessing strategies matters.

---

# 29. Preprocessing the Test Data

The test dataset must be processed using the same imputer.

```
final_X_test = pd.DataFrame(
    final_imputer.transform(
        X_test
    )
)
```

Notice:

```
transform()
```

not:

```
fit_transform()
```

The imputer was already fitted using training data.

---

# 30. Generate Test Predictions

```
preds_test = model.predict(
    final_X_test
)
```

Now:

```
X_test
   ↓
Median imputation
   ↓
Trained Random Forest
   ↓
preds_test
```

---

# 31. Creating the Submission File

The exercise creates:

```
output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': preds_test
})
```

Then:

```
output.to_csv(
    'submission.csv',
    index=False
)
```

This produces:

```
submission.csv
```

which can be submitted to Kaggle.

---

# 🧪 Complete Exercise Code

Here is the important end-to-end pattern from the exercise.

```
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
X_full = pd.read_csv(
    '../input/train.csv',
    index_col='Id'
)

X_test_full = pd.read_csv(
    '../input/test.csv',
    index_col='Id'
)

# Separate target
X_full.dropna(
    axis=0,
    subset=['SalePrice'],
    inplace=True
)

y = X_full.SalePrice

X_full.drop(
    ['SalePrice'],
    axis=1,
    inplace=True
)

# Keep numerical predictors
X = X_full.select_dtypes(
    exclude=['object']
)

X_test = X_test_full.select_dtypes(
    exclude=['object']
)

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)

# Median imputation
final_imputer = SimpleImputer(
    strategy='median'
)

final_X_train = pd.DataFrame(
    final_imputer.fit_transform(X_train),
    columns=X_train.columns
)

final_X_valid = pd.DataFrame(
    final_imputer.transform(X_valid),
    columns=X_valid.columns
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

model.fit(
    final_X_train,
    y_train
)

# Validation
preds_valid = model.predict(
    final_X_valid
)

print(
    mean_absolute_error(
        y_valid,
        preds_valid
    )
)

# Test preprocessing
final_X_test = pd.DataFrame(
    final_imputer.transform(X_test),
    columns=X_test.columns
)

# Test predictions
preds_test = model.predict(
    final_X_test
)

# Submission
output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': preds_test
})

output.to_csv(
    'submission.csv',
    index=False
)
```

---

# 🔑 Important Syntax to Remember

## Find missing columns

```
cols_with_missing = [
    col
    for col in X_train.columns
    if X_train[col].isnull().any()
]
```

---

## Drop missing columns

```
X_train.drop(
    cols_with_missing,
    axis=1
)
```

---

## Create imputer

```
SimpleImputer(
    strategy='median'
)
```

---

## Fit on training data

```
imputer.fit_transform(X_train)
```

---

## Transform validation/test

```
imputer.transform(X_valid)

imputer.transform(X_test)
```

---

## Add missing indicator

```
X_train[col + '_was_missing'] = (
    X_train[col].isnull()
)
```

---

# ⚠️ Common Mistakes

### Mistake 1 — Fitting on validation data

Wrong:

```
imputer.fit_transform(X_valid)
```

Correct:

```
imputer.transform(X_valid)
```

---

### Mistake 2 — Fitting a separate imputer for test data

Wrong:

```
test_imputer = SimpleImputer()
test_imputer.fit_transform(X_test)
```

Correct:

```
final_imputer.transform(X_test)
```

---

### Mistake 3 — Forgetting the target

`SalePrice` is `y`, not a feature.

```
y = X_full.SalePrice
```

---

### Mistake 4 — Removing different columns from train and validation

Both datasets need compatible features.

---

### Mistake 5 — Assuming one imputation method is universally best

The exercise itself demonstrates this.

Mean imputation:

```
≈ 18,063 MAE
```

Dropping columns:

```
≈ 17,838 MAE
```

Median imputation:

```
≈ 17,792 MAE
```

The best choice depends on the dataset and validation results.

---

# 🎤 Interview Revision

### What is imputation?

Replacing missing values with estimated or chosen replacement values.

### What is `SimpleImputer`?

A scikit-learn transformer used to replace missing values.

### Why use `fit_transform()` on training data?

Because the preprocessing rule must first be learned from training data and then applied.

### Why only `transform()` on validation/test data?

Because the preprocessing rule has already been learned from training data.

### What is a missing indicator?

A feature that tells the model whether the original value was missing.

### Why might missingness itself be useful?

Because the fact that a value is missing can sometimes contain information about the observation.

### Is imputation always better than dropping columns?

No. The exercise demonstrates that dropping columns can outperform mean imputation.

### Why can median imputation be useful?

Median is less sensitive to extreme values than mean.

### What metric is used?

Mean Absolute Error.

### Is higher MAE better?

No.

```
Lower MAE = Better
```

---

# 🧠 Exercise 2 — One-Minute Revision

```
Dataset
   ↓
Separate SalePrice
   ↓
Keep numerical features
   ↓
Train/validation split
   ↓
Inspect missing values
   ↓
3 columns have missing values
   ↓
276 missing entries
   ↓
Try different approaches
   ↓
┌──────────────┬──────────────────┬──────────────────────┐
│ Drop columns │ Mean imputation   │ Impute + indicator   │
│ 17,837.83    │ 18,062.89        │ 18,148.42            │
└──────────────┴──────────────────┴──────────────────────┘
   ↓
Try median imputation
   ↓
≈ 17,791.60 MAE
   ↓
Train Random Forest
   ↓
Transform test data
   ↓
Predict SalePrice
   ↓
Create submission.csv
```

---

# ⭐ Core Lesson

> **Missing-value handling is not about blindly choosing one technique. Inspect the data, try reasonable approaches, evaluate them on validation data, and choose the approach that actually works best.**

And the most important preprocessing rule:

> **Fit preprocessing on training data. Transform validation and test data using the already-fitted preprocessing object.**
