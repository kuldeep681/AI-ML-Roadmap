# Kaggle Intermediate Machine Learning

# Exercise 1 — Introduction

---

## 🎯 Exercise Goal

This exercise applies the basic machine learning workflow from the Intro to Machine Learning course to the **Ames Housing** dataset.

The main objective is to practice:

- Loading training and test data
- Separating features and target
- Selecting useful features
- Splitting data into training and validation sets
- Building Random Forest models
- Comparing different model configurations
- Using Mean Absolute Error (MAE)
- Selecting the best model
- Training the final model
- Making predictions on test data
- Creating a Kaggle submission

---

# 1. Understanding the Dataset

The exercise uses two datasets:

- `train.csv`
- `test.csv`

The training dataset contains the target:

```
SalePrice
```

The test dataset does not contain `SalePrice`.

The model must learn the relationship between house characteristics and the selling price.

### Basic ML structure

```
X → Input features
y → Target
```

For this exercise:

```
X → House characteristics
y → SalePrice
```

---

# 2. Selecting the Prediction Target

The target is the value that we want the model to predict.

```
y = X_full.SalePrice
```

Here:

- `X_full` contains the original training data.
- `SalePrice` is the value we want to predict.
- `y` stores the target values.

### 🧠 Remember

In supervised machine learning:

```
X = features / inputs
y = target / output
```

For a house-price problem:

```
X = house information
y = house price
```

---

# 3. Selecting Features

The exercise does not use every column in the dataset.

Instead, a smaller set of features is selected:

```
features = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd'
]
```

Then:

```
X = X_full[features].copy()
X_test = X_test_full[features].copy()
```

### Selected Features

| Feature        | Meaning                         |
| -------------- | ------------------------------- |
| `LotArea`      | Size of the property lot        |
| `YearBuilt`    | Year the house was built        |
| `1stFlrSF`     | First-floor square footage      |
| `2ndFlrSF`     | Second-floor square footage     |
| `FullBath`     | Number of full bathrooms        |
| `BedroomAbvGr` | Number of bedrooms above ground |
| `TotRmsAbvGrd` | Total rooms above ground        |

### Why select features?

Using fewer features makes the initial model easier to understand and keeps the exercise focused on the modeling workflow.

In real projects, feature selection can be much more systematic.

---

# 4. Train/Validation Split

The data is divided into training and validation portions:

```
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)
```

### What happens?

```
Original dataset
      ↓
┌─────┴─────┐
↓           ↓
80%         20%
↓           ↓
```

Training Validation

The model learns from:

```
X_train
y_train
```

The model is evaluated using:

```
X_valid
y_valid
```

### Why?

The validation data represents data that the model did not use during training.

This gives us a better estimate of how the model generalizes.

---

# 5. Random Forest Models

The exercise compares several Random Forest configurations.

The goal is not to assume that one configuration is always best.

Instead:

```
Build multiple candidates
        ↓
Evaluate each candidate
        ↓
Compare validation MAE
        ↓
Select the best model
```

---

## Model 1

```
model_1 = RandomForestRegressor(
    n_estimators=50,
    random_state=0
)
```

### Important parameter

`n_estimators=50`

Means the Random Forest contains 50 trees.

---

## Model 2

```
model_2 = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)
```

This increases the number of trees to 100.

---

## Model 3

```
model_3 = RandomForestRegressor(
    n_estimators=100,
    criterion='absolute_error',
    random_state=0
)
```

This uses:

```
criterion='absolute_error'
```

The exercise tests whether changing the criterion improves validation performance.

---

## Model 4

```
model_4 = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=20,
    random_state=0
)
```

Important parameter:

```
min_samples_split=20
```

This controls how many samples must be present in a node before that node can be split.

---

## Model 5

```
model_5 = RandomForestRegressor(
    n_estimators=100,
    max_depth=7,
    random_state=0
)
```

Important parameter:

```
max_depth=7
```

This limits how deep the individual decision trees can grow.

---

# 6. Understanding the Random Forest Parameters

## `n_estimators`

Controls the number of trees in the forest.

Example:

```
n_estimators=100
```

means:

```
Random Forest
   ├── Tree 1
   ├── Tree 2
   ├── Tree 3
   ├── ...
   └── Tree 100
```

More trees can improve stability, although more trees also increase training time.

---

## `max_depth`

Controls the maximum depth of each tree.

A small value:

```
max_depth=3
```

creates relatively shallow trees.

A larger value allows trees to learn more complex patterns.

### Important

Increasing depth can make the model more flexible but can also increase the risk of overfitting.

---

## `min_samples_split`

Controls how many samples are required before an internal node can be split.

For example:

```
min_samples_split=20
```

means a node generally needs at least 20 samples before it can be considered for splitting.

Increasing this value makes the tree less willing to create very small branches.

---

## `criterion`

Determines the function used to measure the quality of splits.

The exercise tests:

```
criterion='absolute_error'
```

This is related to the use of absolute error when evaluating possible splits.

---

# 7. Creating a Model-Scoring Function

Instead of repeating the same code for every model, the exercise uses a function.

```
def score_model(
    model,
    X_t=X_train,
    X_v=X_valid,
    y_t=y_train,
    y_v=y_valid
):
    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)
```

### What does this function do?

It performs three major operations.

### Step 1 — Train

```
model.fit(X_t, y_t)
```

The model learns patterns from the training data.

### Step 2 — Predict

```
preds = model.predict(X_v)
```

The trained model predicts prices for validation houses.

### Step 3 — Evaluate

```
mean_absolute_error(y_v, preds)
```

The predictions are compared with the actual validation prices.

---

# 8. Mean Absolute Error

## 📖 Concept

MAE stands for:

**Mean Absolute Error**

It measures the average absolute difference between:

```
Actual value
vs
Predicted value
```

For example:

```
Actual price:       300,000
Predicted price:    280,000
```

Absolute error:

```
20,000
```

The MAE averages these absolute errors across all validation examples.

### Interpretation

If:

```
MAE = 25,000
```

we can roughly interpret this as:

> The model's predictions are off by about 25,000 on average.

---

## ⚠️ Important

For MAE:

```
Lower = Better
```

So when comparing models:

```
Model with lowest MAE
            ↓
    Better validation performance
```

---

# 9. Comparing the Models

The models are collected together:

```
models = [
    model_1,
    model_2,
    model_3,
    model_4,
    model_5
]
```

Then each model is evaluated.

```
for i in range(0, len(models)):
    mae = score_model(models[i])

    print(
        "Model %d MAE: %d"
        % (i + 1, mae)
    )
```

The process is:

```
Model 1 → Train → Predict → MAE

Model 2 → Train → Predict → MAE

Model 3 → Train → Predict → MAE

Model 4 → Train → Predict → MAE

Model 5 → Train → Predict → MAE
```

Then the MAE values are compared.

---

# 10. Selecting the Best Model

The exercise uses validation performance to select the best candidate.

The selected model is:

```
best_model = model_3
```

The important lesson is **not**:

> Model 3 is always the best Random Forest.

The correct lesson is:

> Among the candidate configurations tested in this exercise, Model 3 produced the best validation performance.

This is an example of **model selection using validation data**.

---

# 11. Why Do We Compare Multiple Models?

Machine learning models have hyperparameters.

Examples:

```
n_estimators
max_depth
min_samples_split
criterion
```

Different configurations can produce different predictions.

Therefore, instead of assuming:

```
"More trees = automatically better"
```

we test different configurations.

### General workflow

```
Candidate models
      ↓
    Train
      ↓
   Validate
      ↓
  Calculate MAE
      ↓
   Compare
      ↓
 Select winner
```

---

# 12. Training the Selected Model

After selecting the best candidate:

```
best_model = model_3
```

The selected model can then be trained on the available training data.

```
my_model = best_model

my_model.fit(X, y)
```

Here:

- `X` contains the selected features.
- `y` contains `SalePrice`.

The validation data was used to choose the model.

After the model configuration has been selected, the final model can use the available training data before generating test predictions.

---

# 13. Predicting Test Data

The test dataset contains houses for which we need predictions.

```
preds_test = my_model.predict(X_test)
```

The flow is:

```
X_test
   ↓
Trained model
   ↓
Predicted SalePrice
```

The result is an array containing predicted prices.

---

# 14. Creating the Submission DataFrame

The Kaggle competition expects the house ID and predicted sale price.

```
output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': preds_test
})
```

The resulting DataFrame has the structure:

| Id       |       SalePrice |
| -------- | --------------: |
| House ID | Predicted price |
| House ID | Predicted price |
| House ID | Predicted price |

---

# 15. Saving the Submission

```
output.to_csv(
    'submission.csv',
    index=False
)
```

This creates:

```
submission.csv
```

### Why `index=False`?

The pandas DataFrame has its own index.

That index should not become an extra column in the Kaggle submission.

Therefore:

```
index=False
```

is used.

---

# 🧪 Complete Modeling Pattern

The important syntax from the exercise can be remembered as:

```
X = features
y = target

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

model.fit(X_train, y_train)

predictions = model.predict(X_valid)

mae = mean_absolute_error(
    y_valid,
    predictions
)

print(mae)
```

Then after selecting the best model:

```
model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': predictions
})

output.to_csv(
    'submission.csv',
    index=False
)
```

---

# ⚠️ Common Mistakes

## Mistake 1 — Evaluating on training data

Do not use the training predictions as the main basis for model selection.

Bad evaluation:

```
predictions = model.predict(X_train)
```

Better:

```
predictions = model.predict(X_valid)
```

The validation set gives a more realistic estimate of performance on unseen data.

---

## Mistake 2 — Choosing the highest MAE

Wrong:

```
Highest MAE = Best model
```

Correct:

```
Lowest MAE = Best model
```

---

## Mistake 3 — Including the target inside X

`SalePrice` is the target.

It should not be included among the features used to predict itself.

---

## Mistake 4 — Confusing validation and test data

Remember:

```
Training data
    ↓
Used to learn

Validation data
    ↓
Used to compare/select models

Test data
    ↓
Used for final predictions
```

---

# 🎤 Interview Revision

### What is `X`?

The input features used by the model.

### What is `y`?

The target value the model is trying to predict.

### Why split the dataset?

To evaluate the model on data that was not used during training.

### What is MAE?

Mean Absolute Error measures the average absolute difference between actual and predicted values.

### Is a lower or higher MAE better?

Lower MAE is better.

### What does `n_estimators` mean?

The number of trees in a Random Forest.

### What does `max_depth` control?

The maximum depth of individual trees.

### What does `min_samples_split` control?

The minimum number of samples required for a node to be considered for splitting.

### Why compare multiple models?

Because different hyperparameter configurations can produce different validation performance.

### Why use `random_state`?

It makes operations involving randomness reproducible, allowing you to obtain consistent results between runs.

---

# 🔁 30-Second Revision

Remember the exercise as:

```
Load data
    ↓
Select X and y
    ↓
Select useful features
    ↓
Split into train/validation
    ↓
Create multiple Random Forest models
    ↓
Train each model
    ↓
Calculate validation MAE
    ↓
Choose lowest MAE
    ↓
Train selected model
    ↓
Predict test data
    ↓
Create submission.csv
```

---

# ⭐ Key Takeaways

1. `X` contains features and `y` contains the target.
2. Validation data is used to compare candidate models.
3. MAE measures average absolute prediction error.
4. Lower MAE means better validation performance.
5. Random Forest has important hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split`.
6. Model selection should be based on validation performance rather than assumptions.
7. After selecting the model, train the final model and generate predictions for the test data.
8. Kaggle submissions require the expected column names and format.

---

# 🧠 One-Line Memory Trick

**Select → Split → Train → Validate → Compare → Select Best → Retrain → Predict → Submit**
