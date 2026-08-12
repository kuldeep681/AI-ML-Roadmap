# Kaggle Feature Engineering — Exercise Notes
## Exercise 6: Target Encoding

This note is based on the Kaggle exercise notebook `exercise-target-encoding.ipynb`.

---

## 1. What This Exercise Is About

This is the final exercise in the Feature Engineering course.

It applies **target encoding** to categorical features in the Ames housing dataset.

The exercise focuses on three major ideas:

1. Choosing categorical features for target encoding.
2. Applying M-estimate encoding with a separate encoding split.
3. Understanding how target encoding can overfit and leak target information.

The most important warning in this exercise is:

> Target encoding uses the target variable, so leakage prevention is essential.

---

# Part 1 — Dataset and Evaluation

## 2. Dataset

The notebook uses the Ames housing dataset.

The target is:

`SalePrice`

The task is regression.

---

## 3. Model and Metric

The notebook uses:

`XGBRegressor`

for model scoring.

The scoring function performs:

- Categorical factorization
- 5-fold cross-validation
- Negative mean squared log error internally
- Conversion to RMSLE

The final metric is:

`RMSLE`

or:

`Root Mean Squared Log Error`

Lower is better.

---

# Part 2 — Choosing Features for Target Encoding

## 4. Why Choose Particular Categorical Features?

The notebook first examines:

`df.select_dtypes(["object"]).nunique()`

This counts the number of unique categories in each categorical feature.

The exercise emphasizes that categorical features with many categories are often good candidates for target encoding.

Why?

Because one-hot encoding a high-cardinality feature can create a very wide dataset.

Target encoding can instead represent a category using a numerical statistic derived from the target.

---

## 5. Checking Category Frequencies

The notebook then examines:

`df["SaleType"].value_counts()`

This shows how often each category occurs.

This is important because rare categories can create unreliable target statistics.

For example:

`Category A → many observations`

is generally more reliable than:

`Category B → one observation`

when estimating the typical target value for that category.

---

## 6. What Makes a Good Candidate?

When selecting a target-encoded feature, consider:

- Number of unique categories.
- Frequency of each category.
- Whether the category has a meaningful relationship with the target.
- Whether one-hot encoding would become too large.
- Whether the feature has enough observations to produce useful statistics.

The notebook's intended candidates are categorical features where target encoding can provide useful information without creating an unnecessarily wide feature matrix.

---

# Part 3 — The Encoding Split

## 7. Why We Need a Separate Encoding Dataset

Target encoding directly uses:

`SalePrice`

to calculate the encoded value.

That creates a leakage risk.

If the same observations are used to calculate their own target-derived encodings and then train the model, the model can indirectly see the answers.

To avoid this, the notebook creates an **encoding split**.

---

## 8. Creating the Encoding Split

The notebook takes 20% of the data:

`X_encode = df.sample(frac=0.20, random_state=0)`

Then extracts:

`y_encode = X_encode.pop("SalePrice")`

The remaining 80% becomes the training portion:

`X_pretrain`

and:

`y_train`

Conceptually:

`Full dataset`

↓

`20% encoding data`

+

`80% model-training data`

The encoder learns the category-to-target relationship from the encoding data.

It then transforms the training data.

---

## 9. Why This Prevents a Particular Kind of Leakage

The encoder does not learn category statistics from the same observations it is transforming for model training.

The model therefore does not receive an encoding that was calculated directly from each training row's own target.

This is the key purpose of the split.

---

# Part 4 — M-Estimate Encoding

## 10. What Is M-Estimate Encoding?

The notebook uses:

`MEstimateEncoder`

from:

`category_encoders`

The basic idea of target encoding is:

`Category`

↓

`Target statistic`

↓

`Numerical encoded value`

For regression, a common statistic is the category's target mean.

For example:

`Neighborhood A → average SalePrice`

`Neighborhood B → average SalePrice`

and so on.

M-estimate encoding adds **smoothing** to make estimates more reliable, especially for rare categories.

---

## 11. The Smoothing Parameter `m`

The exercise allows experimentation with:

`m`

The parameter controls how strongly category-specific statistics are smoothed toward the overall target statistic.

Conceptually:

`Large amount of category data`

→

More confidence in the category statistic.

`Very little category data`

→

Less confidence in the category statistic.

Smoothing pulls unreliable category estimates toward the global target level.

---

## 12. Applying the Encoder

The notebook's example uses:

`Neighborhood`

with:

`m = 1.0`

The encoder is fitted on:

`X_encode`

and:

`y_encode`

Then the training split is transformed.

The important workflow is:

`Encoding data`

↓

`Fit encoder`

↓

`Learn category-target statistics`

↓

`Transform model-training data`

---

# Part 5 — Understanding the Encoded Feature

## 13. Comparing the Encoding With the Target

The notebook plots:

- The distribution of `y_train`.
- The distribution of the encoded feature.

The purpose is to visually ask:

> Does the encoded feature contain information about `SalePrice`?

If the encoded values show a useful relationship with the target distribution, the encoding may be informative.

But visual similarity alone is not enough.

---

# Part 6 — Comparing Model Performance

## 14. Baseline Score

The notebook first evaluates the original dataset.

Conceptually:

`Original features`

↓

`XGBRegressor`

↓

`5-fold CV`

↓

`Baseline RMSLE`

---

## 15. Score With Target Encoding

It then evaluates the encoded training set.

Conceptually:

`Encoded features`

↓

`XGBRegressor`

↓

`5-fold CV`

↓

`Encoded RMSLE`

The comparison is:

`Baseline RMSLE`

vs.

`Target-encoded RMSLE`

Remember:

**Lower RMSLE is better.**

---

## 16. Why Encoding Can Make the Score Worse

The notebook explicitly warns that depending on which feature you choose, the encoded model may perform significantly worse.

One reason is:

> Some information was used as the separate encoding set instead of being available for ordinary model training.

So there is a trade-off:

`Better category information`

versus:

`Less data available for direct model training`

Target encoding is not automatically beneficial.

It must be validated.

---

# Part 7 — The Overfitting Demonstration

## 17. Why the Notebook Creates a `Count` Feature

The exercise creates an intentionally uninformative feature:

`Count = 0, 1, 2, 3, 4, ...`

This feature should have no real relationship with:

`SalePrice`

It is deliberately constructed to demonstrate what happens when target encoding is done incorrectly.

---

## 18. The Dangerous Workflow

The notebook fits and transforms the encoder on the **same dataset**.

Conceptually:

`Full dataset`

↓

`Create Count`

↓

`Fit target encoder using SalePrice`

↓

`Transform same rows`

↓

`Train XGBoost`

This is dangerous because the encoded value for each observation is influenced by its own target information.

---

# Part 8 — Why the Model Can Almost Perfectly Fit

## 19. The Key Mechanism

Suppose each observation has a nearly unique category.

The target encoder calculates a target-derived value for each category.

If the encoder sees the same observation's target when creating its encoding, the resulting encoded value can carry information that is extremely close to the target itself.

Then XGBoost receives:

`Encoded Count`

which has been contaminated with:

`SalePrice`

information.

The model can exploit this leaked information.

That is why the notebook produces an almost perfect score despite using a feature that was intentionally designed to be uninformative.

---

## 20. Why This Is Not Real Predictive Power

The model has not discovered a genuine relationship between:

`Count`

and:

`SalePrice`

Instead:

`SalePrice`

was allowed to influence the feature representation.

The model is therefore effectively being given information about the answer.

This is **target leakage**.

---

# Part 9 — The Smoothing Experiment

## 21. Experimenting With `m`

The notebook asks you to experiment with:

`m = 0`

`m = 1`

`m = 5`

`m = 50`

The purpose is to observe how smoothing affects the encoded values and the resulting model performance.

The exact score depends on the experiment and execution environment.

The conceptual lesson is more important:

> Smoothing can reduce the effect of unreliable category estimates, but smoothing does not magically make a fundamentally leaky workflow safe.

If the encoder is fitted using the same target information that should remain unseen, leakage can still occur.

---

# Part 10 — Why Rare Categories Are Dangerous

Suppose:

`Category A`

has:

`1000 observations`

and:

`Category B`

has:

`2 observations`

The average target for Category A is based on substantial evidence.

The average target for Category B is much less reliable.

Without smoothing:

`Category B → noisy target estimate`

With smoothing:

`Category B → pulled toward global target average`

This reduces the influence of unreliable estimates.

---

# Part 11 — Target Encoding vs Ordinary Encoding

Target encoding is different from:

### One-Hot Encoding

Creates binary columns for categories.

It does not use the target.

### Ordinal Encoding

Maps categories to numbers such as:

`A → 0`

`B → 1`

`C → 2`

The numbers do not necessarily contain target information.

### Target Encoding

Maps categories using target-derived information.

For example:

`Neighborhood A → estimated target level`

This makes target encoding supervised.

---

# Part 12 — Why High Cardinality Is a Good Use Case

Suppose a categorical variable has:

`1000 unique categories`

One-hot encoding could create roughly:

`1000 columns`

Target encoding can often represent the same categorical variable with:

`1 numerical column`

This can make the feature representation much more compact.

Examples of high-cardinality features include:

- Zipcode
- City
- Product ID
- Merchant ID

But high cardinality alone does not guarantee that target encoding will help.

---

# Part 13 — Safe Target-Encoding Workflow

A leakage-aware conceptual workflow is:

`Dataset`

↓

`Separate target`

↓

`Create appropriate encoding data`

↓

`Learn category → target relationship`

↓

`Apply smoothing`

↓

`Transform model data`

↓

`Train`

↓

`Validate`

The exact implementation can become more sophisticated when cross-validation is used.

In production-quality workflows, target encoding should be constructed within the appropriate training folds so validation targets do not influence validation encodings.

---

# Part 14 — Important Difference Between Encoding Data and Model Data

The notebook uses:

`X_encode`

for learning the encoding.

It uses:

`X_pretrain`

for training the predictive model.

This separation is important.

The encoder is effectively a model itself.

It learns information from:

`Category + Target`

Therefore, it must be treated as part of the preprocessing pipeline and protected from leakage.

---

# Part 15 — Important Code Concepts

### `select_dtypes(["object"])`

Selects categorical/object columns.

### `.nunique()`

Counts unique categories.

### `.value_counts()`

Counts observations in each category.

### `sample(frac=0.20, random_state=0)`

Creates the 20% encoding split.

### `.pop("SalePrice")`

Separates the target.

### `MEstimateEncoder(...)`

Creates an M-estimate target encoder.

### `encoder.fit(...)`

Learns the encoding from the encoding dataset.

### `encoder.transform(...)`

Applies the learned encoding to another dataset.

### `encoder.fit_transform(...)`

Fits and transforms on the same data.

This last operation is exactly what the overfitting demonstration uses to show the danger of target leakage.

---

# Part 16 — Why `fit_transform()` Can Be Dangerous Here

For ordinary unsupervised preprocessing, fitting and transforming the same training data can be normal.

But target encoding is different because it uses the target.

Therefore:

`fit_transform(X, y)`

can be dangerous when the encoder is used on the same observations whose target values are being used to create their own encoded features.

The problem is not the method name itself.

The problem is:

> The target information used to create the feature overlaps with the target information the model is supposed to predict.

---

# Part 17 — The Distribution Plot

The notebook compares:

`SalePrice`

with:

`Encoded Count`

The distributions become almost identical in the intentionally leaky experiment.

This is another warning sign.

A feature created from target information can become suspiciously predictive.

---

# Part 18 — Common Mistakes

## Mistake 1 — Encoding before the train/validation split

If target statistics are calculated using the full dataset, validation targets can influence the features.

That is leakage.

## Mistake 2 — Fitting the encoder and model on the same target-derived information

This can cause severe overfitting.

## Mistake 3 — Assuming smoothing solves leakage

Smoothing reduces variance and can help rare categories.

It does not make a leaky data pipeline safe.

## Mistake 4 — Assuming high-cardinality means target encoding is always better

It is simply a strong use case to investigate.

## Mistake 5 — Ignoring rare categories

Rare categories can produce unstable target estimates.

## Mistake 6 — Ignoring unknown categories

A production model can encounter categories that were not present when the encoder was fitted.

The encoding strategy must handle them safely.

---

# Part 19 — Exercise Mental Model

Remember the exercise as:

`Categorical feature`

↓

`Check cardinality`

↓

`Check category frequency`

↓

`Choose candidate`

↓

`Create encoding split`

↓

`Fit M-Estimate Encoder`

↓

`Apply smoothing`

↓

`Transform model data`

↓

`Compare against baseline`

↓

`Validate`

And remember the warning:

`Target Encoding`

↓

`Uses Target`

↓

`Leakage Risk`

↓

`Separate encoding process from model evaluation`

---

# 20. What You Should Be Able to Do

After this exercise, you should be able to:

- Explain target encoding.
- Explain why high-cardinality categories are good candidates.
- Inspect category counts.
- Understand M-estimate encoding.
- Understand the smoothing parameter `m`.
- Create an encoding split.
- Fit an encoder on separate data.
- Transform training data.
- Compare target encoding with a baseline.
- Explain why target encoding can overfit.
- Explain target leakage.
- Understand why fitting and transforming on the same target-derived data is dangerous.
- Understand why smoothing does not eliminate leakage.
- Understand the role of cross-validation in safe evaluation.

---

# 21. Final Takeaway

Target encoding can turn:

`Categorical value`

into:

`Numerical target-informed feature`

This can be extremely useful, especially for high-cardinality categories.

But it comes with a major responsibility:

> The target must not leak into the feature representation in a way that gives the model access to information it would not have at prediction time.

The most important sequence to remember is:

`Choose category`

↓

`Separate encoding data`

↓

`Learn target statistics`

↓

`Smooth`

↓

`Transform`

↓

`Train`

↓

`Validate without leakage`

The deliberately broken `Count` experiment exists to make this lesson unforgettable:

> An almost perfect validation score can be evidence of leakage rather than evidence of an excellent feature.
