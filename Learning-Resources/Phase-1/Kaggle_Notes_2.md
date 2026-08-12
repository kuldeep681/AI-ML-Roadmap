# Intermediate Machine Learning — Kaggle Course Notes

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
