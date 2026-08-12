# Intermediate Machine Learning — Kaggle Course Notes

# Part 1 — Missing Values & Categorical Variables

---

# 🔹 1. Introduction to Intermediate Machine Learning

## 📖 Concept

The Intro to Machine Learning course taught us the basic ML workflow:

```
Data → Features → Model → Validation → Improvement
```

Intermediate ML focuses on the problems that appear when working with **real-world datasets**.

Real datasets are rarely perfectly clean.

They commonly contain:

- Missing values
- Categorical/text values
- Different types of features
- Complex preprocessing requirements
- Limited data for validation
- Risk of data leakage

This course teaches practical techniques to handle these problems.

---

## 🎯 What This Course Covers

The main topics are:

1. Missing values
2. Categorical variables
3. Pipelines
4. Cross-validation
5. Gradient boosting / XGBoost
6. Data leakage

These techniques are especially useful for **tabular datasets** stored in Pandas DataFrames.

---

## ⚙️ Practical Usage

These problems appear constantly in real ML projects.

For example, a house-price dataset might contain:

- `Rooms` → numerical
- `Price` → target
- `YearBuilt` → numerical but potentially missing
- `Type` → categorical
- `Regionname` → categorical
- `BuildingArea` → missing for many houses

A model cannot simply consume this raw data.

We need preprocessing.

---

## 🧠 Important Connection to Intro ML

In Intro ML, we learned:

```
X → Model → Prediction
```

Intermediate ML adds preprocessing:

```
Raw X
   ↓
Preprocessing
   ↓
Model
   ↓
Prediction
```

Later, pipelines will allow us to combine these steps cleanly.

---

# 🔹 2. Missing Values

## 📖 Concept

A **missing value** means that a particular feature has no recorded value for a particular observation.

For example:

```
Rooms    Bathroom    BuildingArea
3        2           150
4        2           NaN
2        1           90
```

`NaN` means the value is missing.

---

## 🤔 Why Do Values Go Missing?

Missing data can happen for many reasons.

Examples:

- Information was not collected
- A person didn't provide an answer
- A feature was not applicable
- Data collection failed
- Historical data is incomplete

Example from the course:

A two-bedroom house may not have a value for the size of a third bedroom.

That doesn't necessarily mean the dataset is broken.

It may simply mean the feature does not apply.

---

## ⚠️ Why Are Missing Values a Problem?

Many ML libraries, including common scikit-learn models, cannot directly work with missing values.

So before training the model, we usually need to:

- Remove missing data, or
- Fill the missing values

The process of filling missing values is called **imputation**.

---

# 🔹 3. Three Approaches to Missing Values

There are three main approaches covered in the course.

---

## Approach 1 — Drop Columns with Missing Values

### 📖 Concept

Remove any feature/column that contains missing values.

Example:

```
Before:

Rooms    BuildingArea    YearBuilt
3        150              1990
4        NaN              2000
2        90               NaN

After:

Rooms
```

The columns containing missing values are completely removed.

---

## ⚙️ Practical Usage

This is the simplest approach.

It can be reasonable when:

- Only a small amount of useful information exists in the column
- The column isn't important
- The dataset has many other useful features

---

## 🧪 Code

```
cols_with_missing = [
    col for col in X_train.columns
    if X_train[col].isnull().any()
]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
```

---

## 🧠 Code Explanation

### `X_train[col].isnull()`

Checks which values are missing.

### `.any()`

Checks whether that column contains at least one missing value.

### `cols_with_missing`

Stores the names of columns containing missing values.

### `drop(..., axis=1)`

Removes those columns.

`axis=1` means:

```
Remove columns
```

while:

```
axis=0
```

means:

```
Remove rows
```

---

## ⚠️ Important

The **same columns must be removed from training and validation data**.

Don't do this:

```
X_train = X_train.drop(...)
X_valid = X_valid.drop(different_columns)
```

Both datasets must have matching features.

---

## ❌ Problem with This Approach

Suppose:

```
10,000 rows
1 important column
1 missing value
```

Dropping the entire column throws away almost all of that useful information.

Therefore, dropping columns can waste valuable data.

---

# 🔹 4. Imputation

## 📖 Concept

Instead of deleting a column, we **fill the missing values**.

This process is called **imputation**.

The simplest example is replacing missing values with the column's mean.

Example:

```
Values:

100
120
NaN
140

Mean:

120

After imputation:

100
120
120
140
```

---

## ⚙️ Practical Usage

Imputation is usually a better starting point when:

- The column contains useful information
- Only some values are missing
- Removing the column would lose important information

---

## 🧪 Code

```
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(
    my_imputer.fit_transform(X_train)
)

imputed_X_valid = pd.DataFrame(
    my_imputer.transform(X_valid)
)
```

---

## 🧠 Very Important: `fit_transform()` vs `transform()`

This is a critical ML pattern.

### Training data

```
fit_transform(X_train)
```

The imputer:

1. Learns the required statistic from training data
2. Applies the transformation

### Validation data

```
transform(X_valid)
```

The imputer:

1. Uses the statistic already learned from training data
2. Applies it to validation data

We **do not fit the imputer separately on validation data**.

---

## 🔥 Remember This

```
Training data:
fit + transform

Validation/Test data:
transform only
```

This principle becomes extremely important when we discuss **data leakage**.

---

## ⚠️ Important

`SimpleImputer` removes the original column names when converted back to a DataFrame in the example.

Therefore the course restores them:

```
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

---

# 🔹 5. Adding Missing-Value Indicators

## 📖 Concept

Sometimes the fact that a value was missing is itself useful information.

Example:

Suppose `BuildingArea` is missing.

That missingness might indicate something about the property.

Instead of simply replacing the missing value, we can create another feature:

```
BuildingArea_was_missing
```

Example:

```
BuildingArea    BuildingArea_was_missing

150             False
200             False
180             False
180             True
```

The missing value is still imputed, but the model also knows that the original value was missing.

---

## ⚙️ Practical Usage

Useful when:

- Missingness itself may contain information
- Rows with missing values may behave differently
- The reason for missing data may be related to the prediction

---

## 🧪 Code

```
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(
    my_imputer.fit_transform(X_train_plus)
)

imputed_X_valid_plus = pd.DataFrame(
    my_imputer.transform(X_valid_plus)
)

imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
```

---

## 🧠 What This Does

Suppose the original feature is:

```
YearBuilt
```

We add:

```
YearBuilt_was_missing
```

So the model receives:

```
YearBuilt
YearBuilt_was_missing
```

The first feature contains the imputed value.

The second feature tells the model whether that value was originally missing.

---

## ⚠️ Important

This technique **does not always improve performance**.

The model should be evaluated to determine whether the additional information actually helps.

---

# 🔹 6. Comparing the Three Missing-Value Approaches

The Melbourne example produced approximately:

| Approach                        |     MAE |
| ------------------------------- | ------: |
| Drop columns                    | 183,550 |
| Imputation                      | 178,166 |
| Imputation + missing indicators | 178,928 |

Remember:

```
Lower MAE = Better
```

Therefore, for this particular dataset:

```
Imputation
    ↓
Best of the three
```

---

## 🧠 Why Did Imputation Win?

The training data contained:

```
10,864 rows
12 columns
```

Missing values existed in:

```
Car            49
BuildingArea   5,156
YearBuilt      4,307
```

Dropping columns would completely remove:

- `Car`
- `BuildingArea`
- `YearBuilt`

That would discard a lot of potentially useful information.

Imputation allows the model to keep those features.

---

# 🔹 7. Choosing a Missing-Value Strategy

Use this mental model:

### Drop columns

```
Missing values
      ↓
Column not important
      ↓
   Drop
```

### Imputation

```
Missing values
      ↓
Column useful
      ↓
  Fill values
```

### Imputation + indicator

```
Missing values
      ↓
Missingness may itself matter
      ↓
Fill + add indicator
```

---

## 🔁 When NOT to Use Simple Dropping

Avoid blindly dropping columns when:

- The column is important
- Only a small percentage is missing
- The dataset is small
- You would lose substantial information

---

## 🔁 When NOT to Assume Imputation Is Perfect

Imputation creates an estimate.

The filled value is usually **not the true original value**.

Therefore:

```
Imputation ≠ recovering the original data
```

It is simply a practical way to allow the model to use the feature.

---

# 🔹 8. Categorical Variables

## 📖 Concept

A **categorical variable** contains a limited set of categories.

Examples:

```
Car Brand:
Honda
Toyota
Ford

Color:
Red
Green
Yellow

House Type:
House
Unit
Townhouse
```

These are different from numerical variables such as:

```
Age = 25
Price = 500000
Rooms = 3
```

---

# 🔹 9. Ordinal vs Nominal Variables

This distinction is extremely important.

## Ordinal Variable

Categories have a meaningful order.

Example:

```
Never
   ↓
Rarely
   ↓
Most days
   ↓
Every day
```

There is a clear ranking.

---

## Nominal Variable

Categories do **not** have a meaningful order.

Example:

```
Red
Yellow
Green
```

There is no meaningful statement such as:

```
Red < Yellow < Green
```

---

## 🔥 Easy Memory Trick

### Ordinal

```
Categories have ORDER
```

### Nominal

```
Categories have NO ORDER
```

---

# 🔹 10. Three Approaches to Categorical Variables

The course covers:

1. Drop categorical columns
2. Ordinal encoding
3. One-hot encoding

---

# 🔹 11. Approach 1 — Drop Categorical Variables

## 📖 Concept

Simply remove columns containing categorical data.

---

## 🧪 Code

```
drop_X_train = X_train.select_dtypes(
    exclude=['object']
)

drop_X_valid = X_valid.select_dtypes(
    exclude=['object']
)
```

---

## 🧠 Explanation

`select_dtypes()` allows us to select columns based on their data type.

Here:

```
exclude=['object']
```

means:

```
Keep everything except object/text columns.
```

---

## ⚙️ Practical Usage

This is acceptable when:

- Categorical columns contain little useful information
- The dataset contains many useful numerical features
- Simplicity is more important than extracting every possible signal

---

## ⚠️ Problem

You may throw away useful information.

Therefore, this is often the simplest but weakest approach.

---

# 🔹 12. Approach 2 — Ordinal Encoding

## 📖 Concept

Ordinal encoding converts categories into numbers.

Example:

```
Never      → 0
Rarely     → 1
Most days  → 2
Every day  → 3
```

The model can now work with numerical values.

---

## 🧪 Code

```
from sklearn.preprocessing import OrdinalEncoder

label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

ordinal_encoder = OrdinalEncoder()

label_X_train[object_cols] = (
    ordinal_encoder.fit_transform(X_train[object_cols])
)

label_X_valid[object_cols] = (
    ordinal_encoder.transform(X_valid[object_cols])
)
```

---

## 🧠 Code Explanation

### `OrdinalEncoder()`

Creates an encoder that converts categories to integers.

### `fit_transform()`

Learns category mappings from training data and transforms it.

### `transform()`

Uses the mappings learned from training data on validation data.

Again:

```
Train → fit_transform()

Validation/Test → transform()
```

---

## ⚠️ Critical Limitation

Ordinal encoding can accidentally introduce an ordering.

For example:

```
Red → 0
Yellow → 1
Green → 2
```

The numbers may make the model think:

```
Green > Yellow > Red
```

But there is no real ranking.

This is why ordinal encoding is most natural when the categories actually have an order.

---

# 🔹 13. Approach 3 — One-Hot Encoding

## 📖 Concept

One-hot encoding creates a separate binary column for each category.

Suppose:

```
Color
```

contains:

```
Red
Yellow
Green
```

One-hot encoding creates:

```
Color_Red
Color_Yellow
Color_Green
```

Example:

```
Original:

Color
Red
Green
Yellow

Encoded:

Red   Yellow   Green
 1      0        0
 0      0        1
 0      1        0
```

---

## 🧠 Why One-Hot Encoding Is Useful

It does **not** create an artificial ordering.

For example:

```
Red = [1,0,0]
Yellow = [0,1,0]
Green = [0,0,1]
```

No category is considered numerically greater than another.

---

## ⚙️ Practical Usage

One-hot encoding is especially useful for **nominal variables**.

Examples:

- City
- Color
- Car brand
- House type
- Region

---

## 🧪 Code

```
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(
    handle_unknown='ignore',
    sparse=False
)

OH_cols_train = pd.DataFrame(
    OH_encoder.fit_transform(X_train[object_cols])
)

OH_cols_valid = pd.DataFrame(
    OH_encoder.transform(X_valid[object_cols])
)
```

---

## 🧠 `handle_unknown='ignore'`

This is an important practical option.

Imagine training data contains:

```
Delhi
Mumbai
Bangalore
```

But validation data contains:

```
Chennai
```

The encoder hasn't seen `Chennai` during training.

Without appropriate handling, this can cause an error.

With:

```
handle_unknown='ignore'
```

the encoder safely handles unseen categories.

---

## 🔹 Combining Encoded and Numerical Features

After encoding, we need to combine:

- Original numerical columns
- New one-hot columns

First remove the original categorical columns:

```
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
```

Then combine:

```
OH_X_train = pd.concat(
    [num_X_train, OH_cols_train],
    axis=1
)

OH_X_valid = pd.concat(
    [num_X_valid, OH_cols_valid],
    axis=1
)
```

---

# 🔹 14. Ordinal Encoding vs One-Hot Encoding

| Feature           | Ordinal Encoding      | One-Hot Encoding           |
| ----------------- | --------------------- | -------------------------- |
| Output            | Integers              | Binary columns             |
| Assumes ordering  | Yes / can imply order | No                         |
| Best for          | Ordinal variables     | Nominal variables          |
| Number of columns | Same                  | Can increase significantly |
| Example           | Low, Medium, High     | Red, Blue, Green           |

---

## 🔥 Easy Decision Rule

Ask:

### "Does the category have a natural order?"

If **YES**:

```
Ordinal Encoding
```

If **NO**:

```
One-Hot Encoding
```

---

# 🔹 15. High-Cardinality Categorical Variables

## 📖 Concept

A categorical feature is **high-cardinality** when it contains many unique categories.

Example:

```
City
```

might contain:

```
5 categories → low cardinality
```

while:

```
ZIP Code
```

might contain:

```
10,000 categories → high cardinality
```

---

## ⚠️ One-Hot Encoding Problem

One-hot encoding creates a column for every category.

If a feature contains thousands of unique values, this can create thousands of columns.

That can:

- Increase memory usage
- Increase computation
- Make the dataset unnecessarily large

The Kaggle course notes that one-hot encoding generally does not perform well when a categorical variable has a large number of values, with **more than about 15 categories** given as a rough guideline in this lesson.

---

# 🔥 Missing Values + Categorical Variables — One-Shot Revision

## Missing Values

Three approaches:

```
Drop columns
    ↓
Impute
    ↓
Impute + missing indicator
```

### Default practical choice

```
Imputation
```

But always validate the result.

---

## Categorical Variables

Three approaches:

```
Drop
    ↓
Ordinal Encoding
    ↓
One-Hot Encoding
```

### Decision rule

```
Has meaningful order?
      │
   YES → Ordinal
      │
    NO → One-Hot
```

---

# 🧠 Critical Code Patterns to Remember

## Missing Values

```
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()

X_train = pd.DataFrame(
    imputer.fit_transform(X_train)
)

X_valid = pd.DataFrame(
    imputer.transform(X_valid)
)
```

---

## Ordinal Encoding

```
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

X_train[cat_cols] = encoder.fit_transform(
    X_train[cat_cols]
)

X_valid[cat_cols] = encoder.transform(
    X_valid[cat_cols]
)
```

---

## One-Hot Encoding

```
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown='ignore'
)

train_encoded = encoder.fit_transform(
    X_train[cat_cols]
)

valid_encoded = encoder.transform(
    X_valid[cat_cols]
)
```

---

# ⚠️ Most Important Rules From Part 1

1. Never blindly delete useful columns with missing values.
2. Imputation is usually a strong baseline.
3. Fit preprocessing on training data only.
4. Transform validation/test data using the already-fitted preprocessing.
5. Ordinal encoding is appropriate when categories have meaningful order.
6. One-hot encoding avoids artificial ordering.
7. One-hot encoding can become expensive with high-cardinality features.
8. Always compare approaches using validation performance.
9. Lower MAE means better regression performance.
10. Preprocessing is part of the ML pipeline, not an afterthought.

---

# 🎯 Interview Recall

### What is imputation?

Replacing missing values with estimated values so the model can use the feature.

### Why shouldn't we always drop missing columns?

Because the column may contain valuable information even if only some values are missing.

### Ordinal vs nominal?

Ordinal has a meaningful order; nominal does not.

### Ordinal encoding vs one-hot encoding?

Ordinal encoding represents categories as integers and can imply ordering. One-hot encoding creates separate binary columns and does not imply ordering.

### Why use `handle_unknown='ignore'`?

To prevent errors when validation/test data contains categories not seen during training.

### Why use `fit_transform()` on training data but `transform()` on validation data?

The preprocessing parameters must be learned only from training data. Using validation data to learn them can cause data leakage.

---

# 🔥 Part 1 Mental Model

Real-world data is messy.

```
Raw Data
   ↓
Missing Values?
   ↓
Impute / Drop
   ↓
Categorical Variables?
   ↓
Encode
   ↓
Clean Features
   ↓
Model
```

The next major improvement is to **combine all these preprocessing operations with the model itself**.

That is where **Pipelines** become important.
