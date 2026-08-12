# Kaggle — Feature Engineering

## Part 2 — Chapter 2: Mutual Information

## 1. Introduction

When working with a new dataset, it can be difficult to know which features deserve attention.

A dataset may contain:

- Many numerical features
- Many categorical features
- Features with unclear importance
- Hundreds or thousands of possible variables

A useful first step is to rank features according to how strongly they are associated with the target.

For this purpose, Kaggle introduces **Mutual Information (MI)**.

### Main idea

Mutual Information measures how much knowing one variable tells us about another variable.

For machine learning:

    Feature → Target

MI asks:

> If I know the value of this feature, how much more certain am I about the target?

A feature that provides a lot of information about the target will generally have a higher MI score.

---

# 2. Mutual Information vs Correlation

Mutual Information is somewhat similar to correlation because both can measure relationships between variables.

However, they are not the same.

### Correlation

Correlation is mainly useful for identifying **linear relationships**.

For example:

    X increases
        ↓
    Y also increases approximately linearly

Correlation can detect this relationship well.

But a feature can have a strong nonlinear relationship with the target and still have a weak correlation.

### Mutual Information

Mutual Information can detect:

- Linear relationships
- Nonlinear relationships
- Other forms of statistical dependence

This makes MI particularly useful during early feature exploration.

### Remember

    Correlation
        ↓
    Mainly detects linear relationships

    Mutual Information
        ↓
    Can detect many kinds of relationships

---

# 3. What Does Mutual Information Measure?

MI is based on the idea of **uncertainty**.

Imagine trying to predict house prices.

Before looking at any feature, the possible prices may range widely:

    $100,000
    $200,000
    $300,000
    $400,000
    ...
    $1,000,000+

There is a lot of uncertainty.

Now suppose we learn:

    Exterior Quality = Excellent

The possible price range may become much narrower.

Therefore:

    Knowing Exterior Quality
            ↓
    Less uncertainty about Price
            ↓
    Higher Mutual Information

If knowing a feature tells us almost nothing about the target:

    Knowing Feature
            ↓
    Almost no reduction in uncertainty
            ↓
    Low Mutual Information

---

# 4. Entropy

The uncertainty discussed in Mutual Information is related to a concept called **entropy**.

Entropy comes from information theory.

A simple way to think about entropy is:

> How much uncertainty exists about a variable?

More uncertainty means more information is needed to describe what happened.

Mutual Information measures how much knowing one variable reduces uncertainty about another.

You do not need the mathematical details of entropy to use MI effectively at this stage.

The important mental model is:

    More information from feature
            ↓
    Less uncertainty about target
            ↓
    Higher MI

---

# 5. Interpreting MI Scores

The minimum possible MI score is:

    0.0

When:

    MI = 0

the two variables are independent.

In simple terms:

> Knowing the feature does not tell us anything about the target.

Higher MI means stronger statistical dependence.

There is theoretically no fixed upper limit, although values above approximately 2 are uncommon in many practical datasets.

### Important

Do not interpret MI as a percentage.

For example:

    MI = 0.8

does NOT mean:

    "The feature is 80% important."

MI is a measure of statistical dependence, not an accuracy percentage.

---

# 6. MI Is a Feature Utility Metric

Mutual Information can be used as a **feature utility metric**.

Its purpose is to help answer:

> Which individual features appear promising enough to investigate further?

For example:

    100 Features
          ↓
    Calculate MI
          ↓
    Rank Features
          ↓
    Investigate promising features

This can prevent us from spending too much time investigating completely uninformative features.

---

# 7. MI Does Not Tell the Whole Story

A very important limitation is that MI is a **univariate metric**.

This means it evaluates one feature at a time.

For example:

    Feature A → Target
    Feature B → Target
    Feature C → Target

MI can measure each of these relationships individually.

However, it does not directly identify relationships such as:

    Feature A + Feature B → Target

A feature might appear weak on its own but become very useful when combined with another feature.

Therefore:

> A low MI score does not automatically mean that a feature should be removed.

---

# 8. Feature Interactions

Suppose we have:

    Feature A
    Feature B

Individually:

    A → weak relationship with target
    B → weak relationship with target

But together:

    A + B → strong relationship with target

MI calculated separately for A and B may fail to reveal this interaction.

This is why visualization and domain knowledge remain important.

---

# 9. MI Depends on the Model

Another important point:

> A feature's actual usefulness depends on the model being used.

Suppose a feature contains a nonlinear relationship with the target.

MI may correctly identify that the relationship exists.

But if the model cannot learn that relationship effectively, the feature may not improve the final model much.

Therefore:

    High MI
        ↓
    Feature contains useful information
        ↓
    But model must be able to learn it

Sometimes feature engineering or transformation is needed to make the relationship easier for the model to learn.

---

# 10. Example — Automobile Dataset

The course uses the **1985 Automobile dataset**.

The dataset contains information about cars from the 1985 model year.

The goal is to predict:

    price

using features such as:

- `make`
- `body_style`
- `horsepower`
- `engine_size`
- `curb_weight`
- `fuel_type`
- `highway_mpg`
- `fuel_system`
- `stroke`
- `compression_ratio`
- and other vehicle characteristics

The dataset contains:

    193 cars
    23 explanatory features
    price as the target

---

# 11. Preparing the Data

First, separate the target from the features.

    X = df.copy()
    y = X.pop("price")

Here:

    X = input features
    y = prediction target

---

# 12. Handling Categorical Features for MI

Scikit-learn treats discrete and continuous features differently when calculating MI.

Therefore, we need to tell the algorithm which features are discrete.

Categorical columns can be converted into integer labels.

The course uses:

    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

After this operation, categorical values are represented by integers.

For example:

    Before:

    Toyota
    Ford
    Honda

    After:

    Toyota → 0
    Ford   → 1
    Honda  → 2

The exact integer assignment itself is not important here.

---

# 13. Identifying Discrete Features

After encoding the categorical variables:

    discrete_features = X.dtypes == int

This creates information that tells the MI algorithm which columns should be treated as discrete.

### Important

Do not blindly assume that every integer column is categorical in every dataset.

The goal is to correctly identify whether each feature is:

- Discrete
- Continuous

The course uses the integer dtype as a practical rule for this particular example.

---

# 14. Choosing the MI Function

Scikit-learn provides different MI functions depending on the type of target.

### Regression Target

For a numerical target:

    mutual_info_regression

### Classification Target

For a categorical target:

    mutual_info_classif

In the Automobile example:

    Target = price

Since `price` is numerical, we use:

    mutual_info_regression

---

# 15. Creating the MI Score Function

The course creates a helper function:

    from sklearn.feature_selection import mutual_info_regression

    def make_mi_scores(X, y, discrete_features):
        mi_scores = mutual_info_regression(
            X,
            y,
            discrete_features=discrete_features
        )

        mi_scores = pd.Series(
            mi_scores,
            name="MI Scores",
            index=X.columns
        )

        mi_scores = mi_scores.sort_values(
            ascending=False
        )

        return mi_scores

This function:

1. Calculates MI for every feature.
2. Stores the scores.
3. Associates each score with its feature name.
4. Sorts the features from highest MI to lowest MI.

Then:

    mi_scores = make_mi_scores(
        X,
        y,
        discrete_features
    )

---

# 16. Example MI Results

The course obtains scores similar to:

    curb_weight          1.540126
    highway_mpg          0.951700
    length               0.621566
    fuel_system          0.485085
    stroke               0.389321
    num_of_cylinders     0.330988
    compression_ratio    0.133927
    fuel_type            0.048139

The important lesson is the ranking.

### High-scoring example

    curb_weight

has a strong relationship with:

    price

### Low-scoring example

    fuel_type

has a relatively low individual MI score.

But we should not immediately conclude that `fuel_type` is useless.

---

# 17. Visualizing MI Scores

A bar plot can make the ranking easier to understand.

The general process is:

    MI Scores
        ↓
    Sort features
        ↓
    Create bar chart
        ↓
    Compare feature relationships

Visualization is useful because it gives us a quick overview of which features deserve further investigation.

---

# 18. High MI Example — Curb Weight

The `curb_weight` feature receives a high MI score.

The course then visualizes the relationship:

    curb_weight
          vs
       price

The visualization shows a strong relationship.

This supports the MI result.

This demonstrates an important workflow:

    Calculate MI
        ↓
    Identify promising feature
        ↓
    Visualize relationship
        ↓
    Understand why the feature is useful

---

# 19. Low MI Example — Fuel Type

The `fuel_type` feature has a relatively low MI score.

It would be tempting to conclude:

    Low MI
       ↓
    Feature is useless

But that would be incorrect.

The course investigates the feature through visualization.

It looks at:

    horsepower
         vs
       price

while separating the observations according to:

    fuel_type

The visualization shows that fuel type separates different price trends.

This indicates that `fuel_type` may contribute an **interaction effect**.

---

# 20. Interaction Effects

An interaction occurs when the effect of one feature depends on another feature.

For example:

    horsepower → price

may behave differently for:

    fuel_type = gas

compared with:

    fuel_type = diesel

Therefore:

    horsepower + fuel_type

may contain useful information even if `fuel_type` alone has a low MI score.

### Key lesson

> A feature can have low individual MI and still be useful through interactions.

---

# 21. MI and Feature Engineering

Mutual Information is particularly useful at the beginning of feature development.

A practical workflow is:

    Dataset
        ↓
    Identify target
        ↓
    Calculate MI
        ↓
    Rank features
        ↓
    Visualize promising relationships
        ↓
    Search for interactions
        ↓
    Create useful features
        ↓
    Validate the model

MI helps us decide where to spend our feature-engineering effort.

---

# 22. MI Does Not Automatically Select the Final Features

Do not use this rule:

    High MI → always keep
    Low MI → always remove

Instead:

    High MI
        ↓
    Investigate

    Low MI
        ↓
    Investigate interactions before removing

The final decision should depend on:

- Model performance
- Domain knowledge
- Feature interactions
- Validation results
- Whether the feature is available at prediction time

---

# 23. Mutual Information vs Model Performance

Suppose two features have:

    Feature A → MI = 1.2
    Feature B → MI = 0.2

This tells us that Feature A has a stronger individual relationship with the target.

It does NOT tell us that:

    Feature A will improve model performance by 6×.

MI and model performance are different concepts.

MI is a measure of association.

Model evaluation measures how well the complete model predicts the target.

---

# 24. Important Limitations

## Limitation 1 — Univariate

MI evaluates features individually.

It does not directly identify multi-feature interactions.

## Limitation 2 — Model Dependence

A relationship identified by MI may not be easy for every model to learn.

## Limitation 3 — Low MI Does Not Mean Useless

A feature may become useful when combined with other features.

## Limitation 4 — Visualization Still Matters

A numerical ranking should be followed by investigation.

---

# 25. Practical Mental Model

Think of MI as a **screening tool**.

Imagine having 100 features.

Instead of deeply investigating all 100:

    100 Features
          ↓
    Mutual Information
          ↓
    Feature Ranking
          ↓
    Focus on promising features
          ↓
    Visualize
          ↓
    Engineer
          ↓
    Validate

This makes feature engineering more systematic.

---

# 26. When Should You Use MI?

MI is especially useful when:

- You are starting feature engineering.
- You have many features.
- You don't yet know which features are important.
- You suspect nonlinear relationships.
- You want a model-independent initial feature ranking.
- You want to discover potentially useful relationships.

---

# 27. When Should You Be Careful With MI?

Be careful when:

- A feature has a low MI score.
- Features may interact strongly.
- The model has limited ability to learn certain relationships.
- The data contains leakage.
- The feature representation is poor.
- You are treating MI as a replacement for model validation.

---

# 28. Interview Questions

### What is Mutual Information?

Mutual Information measures the statistical dependency between two variables. In feature engineering, it measures how much information an individual feature provides about the target.

### What is the main advantage of MI over correlation?

MI can detect nonlinear relationships, whereas correlation primarily measures linear relationships.

### What does MI = 0 mean?

It indicates that the feature and target are statistically independent, meaning the feature provides no information about the target by itself.

### Does high MI mean a feature is definitely useful?

No. Its usefulness depends on the model, interactions with other features, and validation performance.

### Is MI multivariate?

No. MI in this feature-ranking context is **univariate**.

### Can a low-MI feature still be useful?

Yes. It may participate in an interaction with another feature.

### Which MI function is used for regression?

    mutual_info_regression

### Which MI function is used for classification?

    mutual_info_classif

---

# 29. Quick Revision

## Mutual Information

**Definition:**

A measure of how much knowing one variable reduces uncertainty about another.

**Purpose in feature engineering:**

Rank individual features according to their relationship with the target.

**Minimum value:**

    0

**Higher MI:**

Stronger statistical dependence.

**Main advantage:**

Can identify nonlinear relationships.

**Major limitation:**

It is a univariate metric and does not directly detect feature interactions.

**Important warning:**

Low MI does not automatically mean a feature is useless.

---

# 30. One-Shot Revision

Remember this sequence:

    New Dataset
          ↓
    Many Features
          ↓
    Calculate Mutual Information
          ↓
    Rank Features
          ↓
    Visualize Relationships
          ↓
    Check Interactions
          ↓
    Create / Transform Features
          ↓
    Validate Model

The most important idea is:

> **Mutual Information tells us how informative an individual feature is about the target, but it is only a starting point for feature engineering.**

---

# 31. Chapter 2 Key Terms

| Term                     | Meaning                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Mutual Information       | Measures statistical dependency between variables                  |
| Entropy                  | A measure related to uncertainty                                   |
| Feature Utility Metric   | A metric used to estimate how useful a feature may be              |
| Univariate               | Considering one feature at a time                                  |
| Interaction              | A relationship where the effect of one feature depends on another  |
| `mutual_info_regression` | MI calculation for numerical targets                               |
| `mutual_info_classif`    | MI calculation for categorical targets                             |
| Discrete Feature         | A feature whose possible values are treated as separate categories |
| Continuous Feature       | A numerical feature treated as a continuous quantity               |

---

# 32. Chapter 2 — Final Takeaway

Mutual Information is a powerful **first-pass feature investigation tool**.

It helps answer:

> "Which individual features appear to contain useful information about my target?"

But the complete feature-engineering process requires more:

    MI
    +
    Visualization
    +
    Domain Knowledge
    +
    Feature Interactions
    +
    Model Validation

That combination is much more reliable than depending on a single feature score.
