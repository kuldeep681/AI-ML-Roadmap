# Kaggle — Feature Engineering

## Part 1 — What Is Feature Engineering?

## 1. What Is Feature Engineering?

**Feature engineering** is the process of creating or transforming features so that the data is better suited to the machine learning problem.

The goal is not simply to create more columns.

The goal is to create **better representations of information already present in the data** so that a machine learning model can learn useful relationships more effectively.

### Main Goals

- Improve predictive performance
- Reduce computational or data requirements
- Improve interpretability of results

> The goal of feature engineering is to make your data better suited to the problem at hand.

---

## 2. Real-World Example — Apparent Temperature

Consider measurements such as:

- Air temperature
- Humidity
- Wind speed

These are quantities that can be measured directly.

However, what we may actually care about is **how hot or cold it feels to a person**.

Measures such as **heat index** and **wind chill** combine directly measured quantities to represent perceived temperature more meaningfully.

This is similar to feature engineering.

The important idea is that a new representation can be more useful for the problem than the raw measurements themselves.

---

## 3. Guiding Principle of Feature Engineering

The most important principle is:

> **For a feature to be useful, it must have a relationship to the target that your model is able to learn.**

Different models can learn different types of relationships.

For example, a linear model is designed to learn linear relationships.

If the actual relationship between a feature and the target is not linear, the model may perform poorly unless we transform the feature into a representation that makes the relationship easier to learn.

Therefore, when performing feature engineering, ask:

> **What relationship exists in the data that my model might have difficulty learning?**

Then consider whether a transformation can expose that relationship.

---

## 4. Example — Length and Area

Suppose we want to predict the area of a square plot of land.

The relationship is:

**Area = Length²**

If our dataset contains only `Length`, a linear model cannot naturally represent this squared relationship.

We can create a new feature:

    df["Area"] = df["Length"] ** 2

The information was already present in `Length`.

We simply represented it in a form that is more useful to the model.

---

## 5. Why Feature Transformations Can Be Powerful

A transformation applied to a feature effectively becomes part of the model's representation of the problem.

If the model cannot naturally learn a relationship, we can sometimes transform the data so that the model can learn it.

This is why feature engineering can provide a high return on the time invested.

As you develop features, think about:

> **What information could my model use to achieve better performance?**

---

## 6. Domain Knowledge

Feature engineering becomes especially useful when we understand the real-world problem.

Domain knowledge can help us identify:

- Ratios
- Totals
- Differences
- Interactions
- Meaningful transformations
- Relationships between variables

For example, if a dataset contains:

- Income
- Debt

we might create:

**Debt-to-Income Ratio = Debt / Income**

The important point is not simply performing a mathematical operation.

The important point is understanding **why that representation could be useful**.

---

## 7. Concrete Strength Example

The course demonstrates feature engineering using a **Concrete** dataset.

The goal is to predict:

`CompressiveStrength`

from different properties of concrete.

Important features include:

- Cement
- BlastFurnaceSlag
- FlyAsh
- Water
- Superplasticizer
- CoarseAggregate
- FineAggregate
- Age

The target is:

`CompressiveStrength`

---

## 8. Establishing a Baseline

Before creating new features, the course first establishes a baseline model.

This gives us a reference point for evaluating our engineered features.

    X = df.copy()
    y = X.pop("CompressiveStrength")

    baseline = RandomForestRegressor(
        criterion="absolute_error",
        random_state=0
    )

    baseline_score = cross_val_score(
        baseline,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    baseline_score = -1 * baseline_score.mean()

    print(f"MAE Baseline Score: {baseline_score:.4}")

The baseline score is:

**MAE = 8.232**

The baseline is important because we need something to compare against after feature engineering.

---

## 9. Why Establish a Baseline?

Without a baseline, we cannot easily determine whether our feature engineering actually helped.

The workflow is:

1. Train the model with the original features.
2. Record its validation score.
3. Create new features.
4. Train the model again.
5. Compare the new score with the baseline.

For MAE:

**Lower is better.**

Therefore:

- Baseline MAE = 8.232
- New MAE = 7.948

means that the new model performed better.

---

## 10. Creating Ratio Features

The course makes an important observation:

> The ratio of ingredients in a recipe can sometimes be more useful than their absolute amounts.

The same reasoning can apply to concrete.

Instead of only giving the model the raw ingredient quantities, we can create ratios that represent relationships between the ingredients.

Three new features are created.

### Fine Aggregate / Coarse Aggregate

    X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]

### Aggregate / Cement

    X["AggCmtRatio"] = (
        X["CoarseAggregate"] + X["FineAggregate"]
    ) / X["Cement"]

### Water / Cement

    X["WtrCmtRatio"] = X["Water"] / X["Cement"]

These features expose relationships between the ingredients rather than only providing their individual quantities.

---

## 11. Evaluating the Engineered Features

The model is trained again using the additional ratio features.

    model = RandomForestRegressor(
        criterion="absolute_error",
        random_state=0
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    score = -1 * score.mean()

    print(f"MAE Score with Ratio Features: {score:.4}")

The resulting score is:

**MAE = 7.948**

### Comparison

| Model          |   MAE |
| -------------- | ----: |
| Baseline       | 8.232 |
| Ratio Features | 7.948 |

Since lower MAE is better, the engineered features improved model performance.

---

## 12. What Did the Ratio Features Teach Us?

The important lesson is not:

> Always create ratios.

The deeper lesson is:

> **Use knowledge about the problem to create features that expose useful relationships in the data.**

No new external information was collected.

We simply transformed existing information:

- Cement
- Water
- FineAggregate
- CoarseAggregate

into more useful representations:

- Water / Cement
- FineAggregate / CoarseAggregate
- Aggregate / Cement

These new representations helped the model make better predictions.

---

## 13. More Features Do Not Automatically Mean a Better Model

Adding features can sometimes:

- Add noise
- Increase computational requirements
- Increase model complexity
- Contribute to overfitting
- Make interpretation more difficult

Therefore:

> **Feature quality is more important than feature quantity.**

The goal is not to create as many features as possible.

The goal is to create useful features.

---

## 14. Feature Engineering Workflow

A practical feature engineering workflow is:

1. Understand the problem.
2. Understand the available data.
3. Identify relationships that might matter.
4. Create meaningful features.
5. Train the model.
6. Validate the model.
7. Compare against the baseline.
8. Keep useful features.
9. Remove features that do not provide meaningful benefit.

Feature engineering is an iterative process.

---

## 15. Baseline → Feature → Validation

A useful mental model is:

**Baseline**

Original Features → Model → Validation Score

**Feature Engineering**

Original Features + Engineered Features → Model → Validation Score

**Comparison**

New Score vs Baseline Score

This allows us to determine whether the feature engineering actually helped.

---

## 16. Feature Engineering Is Not Random Feature Creation

Avoid this approach:

> Create hundreds of random mathematical combinations and hope something works.

Instead:

**Understand the problem**

→ Identify a meaningful relationship

→ Create a feature representing that relationship

→ Validate it

→ Keep it if it provides useful improvement

Feature engineering should have a reason behind it.

---

## 17. Data Leakage Warning

When creating features, make sure the feature uses only information that would actually be available when the prediction is made.

For example, if predicting a future house sale price, we cannot create a feature using the final sale price.

That would be **data leakage**.

Always ask:

> **Would this information actually be available at prediction time?**

---

## 18. Feature Engineering and Model Choice

The usefulness of a feature depends partly on the model being used.

A transformation that is useful for a simple linear model may provide little additional benefit to a more flexible model that can already learn the relationship.

Therefore:

> **A feature is useful to the extent that its relationship with the target is something the chosen model can learn.**

---

## 19. Common Mistakes

### Mistake 1 — Assuming More Features Are Always Better

More features can introduce noise and complexity.

### Mistake 2 — Creating Features Without Understanding the Problem

A mathematically valid feature is not necessarily a useful feature.

### Mistake 3 — Not Establishing a Baseline

Without a baseline, it is difficult to know whether feature engineering actually helped.

### Mistake 4 — Not Validating New Features

New features should be evaluated using an appropriate validation method.

### Mistake 5 — Creating Leaky Features

Never use information that would not be available when making the prediction.

---

## 20. Key Terms

| Term                   | Meaning                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| Feature                | An input variable used by a machine learning model               |
| Feature Engineering    | Creating or transforming features to make data more useful       |
| Feature Transformation | Changing the representation of an existing feature               |
| Feature Creation       | Creating a new feature from existing information                 |
| Feature Interaction    | Combining features to represent a relationship                   |
| Domain Knowledge       | Real-world understanding used to create useful features          |
| Baseline               | Model performance before introducing an improvement              |
| Data Leakage           | Using information that would not be available at prediction time |
| Target                 | The value the model is trying to predict                         |

---

## 21. One-Shot Revision

### What is feature engineering?

Feature engineering is the process of creating or transforming features so that the data is better suited to the machine learning problem.

### Why do we perform it?

- Improve predictive performance
- Reduce computational or data requirements
- Improve interpretability

### Main principle

A useful feature should expose a relationship with the target that the model can learn.

### How can features be created?

- Mathematical transformations
- Ratios
- Sums
- Differences
- Interactions
- Domain-specific transformations

### How do we know whether a feature helps?

Compare:

**Baseline model performance**

with:

**Model performance after feature engineering**

If the new features meaningfully improve validation performance, they may be useful.

---

## 22. Final Mental Model

Remember:

**Understand → Transform → Create → Validate → Keep useful features**

The goal of feature engineering is:

> **Make the data easier for the model to learn from.**

---

## Part 1 — COMPLETE ✅
