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

# Part 2 — Chapter 2: Mutual Information

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

# Part 3 — Chapter 3: Creating Features with Clustering

## 1. Introduction

This chapter introduces **unsupervised learning** as a feature-engineering technique.

Unlike supervised learning, unsupervised learning does not use a target variable.

The goal is instead to discover structure within the features.

For feature engineering, we can use unsupervised algorithms as **feature discovery techniques**.

One important unsupervised learning technique is:

**Clustering**

Clustering means assigning data points to groups based on how similar they are to one another.

A simple mental model is:

    Similar data points
          ↓
       Same group
          ↓
      Cluster

For example, a customer dataset might contain groups of customers with similar:

- Spending behavior
- Income
- Age
- Location

These groups can become useful features for a machine learning model.

---

# 2. Why Use Clustering for Feature Engineering?

Suppose a dataset contains several features describing houses.

The relationship between these features and the target may be complicated.

A simple model may struggle to learn the entire relationship.

Clustering can divide the data into smaller groups where the relationships are easier to learn.

This is essentially a:

**Divide and conquer strategy**

Instead of asking the model to learn:

    One complicated relationship

we can create:

    Cluster 1 → simpler relationship
    Cluster 2 → simpler relationship
    Cluster 3 → simpler relationship
    ...

The model can then learn these smaller relationships separately.

---

# 3. Cluster Labels as Features

After clustering, every data point receives a cluster label.

For example:

| Longitude | Latitude | Cluster |
| --------: | -------: | ------: |
|   -93.619 |   42.054 |       3 |
|   -93.619 |   42.053 |       3 |
|   -93.638 |   42.060 |       1 |
|   -93.602 |   41.988 |       0 |

The new feature is:

    Cluster

This feature tells the model which group the observation belongs to.

---

# 4. Cluster Labels Are Categorical

Although cluster labels are commonly represented as integers:

    0
    1
    2
    3
    ...

these numbers do not necessarily represent an ordering.

For example:

    Cluster 0
    Cluster 1
    Cluster 2

does not mean:

    Cluster 2 > Cluster 1 > Cluster 0

The numbers are simply labels identifying different groups.

Therefore, the `Cluster` feature should generally be treated as a **categorical feature**.

Depending on the model, it may be appropriate to use:

- Label encoding
- One-hot encoding

---

# 5. The Main Idea Behind Cluster Features

Suppose a model is trying to learn:

    Location + Income → House Price

The relationship might be complicated.

Clustering can create:

    Location + Income
           ↓
        Cluster
           ↓
    Cluster becomes a new feature

Now the model can learn different patterns for different clusters.

For example:

    Cluster 0 → lower-income region
    Cluster 1 → middle-income region
    Cluster 2 → high-income region

The exact meaning of each cluster is discovered from the data.

---

# 6. Clustering as Multi-Dimensional Binning

When clustering is applied to one numerical feature, it behaves somewhat like:

**Binning / Discretization**

For example:

    Income
      ↓
    Low
    Medium
    High

When clustering is applied to multiple features, it becomes similar to **multi-dimensional binning**.

For example:

    Latitude
    Longitude
    Income
       ↓
    Clustering
       ↓
    Geographic-economic segments

This allows the model to work with complicated regions of feature space.

---

# 7. K-Means Clustering

There are many clustering algorithms.

The algorithm used in this chapter is:

**K-Means**

K-Means is intuitive and commonly useful for feature engineering.

It measures similarity using ordinary straight-line distance:

**Euclidean distance**

---

# 8. What Does "K" Mean?

The `K` in K-Means represents the number of clusters.

For example:

    K = 3

means the algorithm creates:

    3 clusters

Similarly:

    K = 6

creates:

    6 clusters

You choose the value of `K`.

---

# 9. Centroids

K-Means creates clusters using points called:

**Centroids**

A centroid represents the center of a cluster.

The basic idea is:

    Data Point
         ↓
    Find nearest centroid
         ↓
    Assign point to that cluster

For example:

    Centroid A
         ●

    Centroid B
                 ●

    Centroid C
                         ●

Each observation is assigned to whichever centroid is closest.

---

# 10. How K-Means Works

K-Means follows an iterative process.

### Step 1 — Initialize Centroids

The algorithm starts with a predefined number of centroids.

The number is determined by:

    n_clusters

The initial positions are chosen automatically.

---

### Step 2 — Assign Points

Each data point is assigned to the nearest centroid.

    Point
      ↓
    Calculate distance to centroids
      ↓
    Choose closest centroid
      ↓
    Assign cluster

---

### Step 3 — Move Centroids

After assigning the points, each centroid is moved to better represent the points assigned to it.

The goal is to minimize the total distance between:

    Data points
        and
    Their assigned centroid

---

### Step 4 — Repeat

The algorithm repeats:

    Assign points
          ↓
    Move centroids
          ↓
    Assign points
          ↓
    Move centroids
          ↓
        ...

until:

- The centroids stop moving significantly, or
- The maximum number of iterations is reached.

---

# 11. Important K-Means Parameters

The course focuses on three important parameters:

- `n_clusters`
- `max_iter`
- `n_init`

---

# 12. n_clusters

`n_clusters` determines the number of clusters.

Example:

    KMeans(n_clusters=6)

creates six clusters.

This is generally the most important parameter you need to choose.

Choosing the best value depends on:

- The dataset
- The features
- The target
- The model
- The intended use of the feature

The course recommends treating `n_clusters` like a model hyperparameter and tuning it using validation.

---

# 13. max_iter

`max_iter` controls the maximum number of iterations allowed.

Each iteration involves:

    Assign points
          ↓
    Move centroids

The process continues until convergence or until `max_iter` is reached.

A larger value can be useful for more complicated clustering problems.

However, the default value is often sufficient for ordinary datasets.

---

# 14. n_init

K-Means can produce different results depending on where the centroids are initially placed.

A poor initial position can lead to a poor clustering.

To reduce this problem, K-Means can run the algorithm multiple times with different initializations.

This is controlled by:

    n_init

The algorithm then selects the clustering with the lowest total distance between observations and their centroids.

---

# 15. Why Initialization Matters

Imagine two possible initializations.

### Initialization A

The centroids begin in useful locations.

Result:

    Good clustering

### Initialization B

The centroids begin in poor locations.

Result:

    Poor clustering

Therefore, K-Means repeats the process using different initial centroid positions.

This is one reason `n_init` matters.

---

# 16. K-Means and Convergence

K-Means repeatedly performs:

    1. Assign each point to the nearest centroid
    2. Move each centroid toward the center of its assigned points

Eventually the centroids stop moving significantly.

At this point, the algorithm has converged.

If convergence does not happen before:

    max_iter

is reached, the algorithm stops because the maximum number of iterations has been exhausted.

---

# 17. Voronoi Tessellation

K-Means divides the feature space into regions according to the nearest centroid.

This creates what is called a:

**Voronoi tessellation**

You do not need to memorize the mathematical details for practical use.

The useful idea is:

> The learned regions determine which cluster future observations will belong to.

In other words, K-Means creates boundaries around the centroids.

Each region corresponds to one cluster.

---

# 18. Clustering and Model Learning

The main reason we care about clustering in this course is not simply to discover groups.

We want to use those groups as features.

Suppose:

    Original features
          ↓
       K-Means
          ↓
    Cluster labels
          ↓
    Add Cluster to dataset
          ↓
    Train predictive model

The cluster label can expose structure that the original features did not make easy for the model to learn.

---

# 19. Example — YearBuilt and SalePrice

The course gives an example involving:

    YearBuilt
          ↓
    SalePrice

The relationship can be curved and complicated.

A simple linear model may struggle because it tries to fit one straight-line relationship.

Clustering can divide the observations into groups.

For example:

    Older houses
    Middle-aged houses
    Newer houses

Within each smaller group, the relationship may be closer to linear.

Therefore:

    Complicated relationship
             ↓
          Clustering
             ↓
       Smaller regions
             ↓
      Simpler relationships
             ↓
        Easier to learn

This is the divide-and-conquer idea behind cluster features.

---

# 20. Example — California Housing

The chapter uses the California Housing dataset.

Important features include:

- `MedInc`
- `Latitude`
- `Longitude`

The goal is to create geographic and economic segments.

The target is:

    MedHouseVal

which represents median house value.

---

# 21. Why These Features Are Good Candidates?

`Latitude` and `Longitude` naturally describe geographic position.

`MedInc` describes median income.

Together they can reveal regions that are similar in:

- Geographic location
- Economic conditions

Therefore, they are natural candidates for clustering.

---

# 22. Scaling and K-Means

K-Means is sensitive to feature scale.

It uses Euclidean distance.

Therefore, a feature with very large numerical values can dominate the distance calculation.

For example:

    Feature A:
    values around 0–1

    Feature B:
    values around 0–100000

Feature B may have a much stronger effect on the distance.

Therefore:

> Scaling or normalization can be important before K-Means.

In the California Housing example, the selected features were already considered to be on roughly comparable scales, so the course leaves them as they are.

---

# 23. Creating a Cluster Feature

The course creates six clusters:

    kmeans = KMeans(n_clusters=6)

Then:

    X["Cluster"] = kmeans.fit_predict(X)

This does two things:

1. Fits K-Means to the data.
2. Assigns every observation a cluster label.

The result is stored in:

    X["Cluster"]

---

# 24. Treating Cluster as Categorical

The course then converts the cluster feature into a categorical type:

    X["Cluster"] = X["Cluster"].astype("category")

This reflects the fact that:

    Cluster 0
    Cluster 1
    Cluster 2

are categories rather than ordered numerical values.

---

# 25. Understanding the Geographic Clusters

The course visualizes:

    Longitude
          vs
    Latitude

with cluster labels.

The resulting clusters can reveal geographic patterns.

For example, the algorithm creates separate segments for some higher-income coastal regions.

This is useful because the cluster feature summarizes multiple characteristics into a single categorical feature.

---

# 26. Checking Whether Clusters Are Informative

Creating clusters is not enough.

We need to determine whether the clusters actually provide useful information about the target.

The course examines:

    Cluster
       vs
    MedHouseVal

using box plots.

If the clusters are informative, their target distributions should be meaningfully different.

For example:

    Cluster A
        ↓
    Mostly lower house values

    Cluster B
        ↓
    Mostly medium house values

    Cluster C
        ↓
    Mostly higher house values

If the distributions are well separated, that suggests the clustering has captured useful structure.

---

# 27. Cluster Feature Workflow

The complete process is:

    Select useful features
            ↓
    Check feature scales
            ↓
    Choose K
            ↓
    Fit K-Means
            ↓
    Generate cluster labels
            ↓
    Add labels as a feature
            ↓
    Treat cluster as categorical
            ↓
    Check relationship with target
            ↓
    Validate model performance

---

# 28. How to Choose K

There is no universal value of K that always works.

For example:

    K = 2
    K = 4
    K = 6
    K = 10

may produce very different feature representations.

The best partition depends on:

- The dataset
- The selected features
- The target
- The model
- The objective of the feature

Therefore, treat `n_clusters` as a hyperparameter.

A practical approach is:

    Try different K values
          ↓
    Create cluster feature
          ↓
    Train model
          ↓
    Cross-validation
          ↓
    Compare scores
          ↓
    Keep useful configuration

---

# 29. Important Warning About Cluster Meaning

Do not assume that cluster labels have inherent meaning.

For example:

    Cluster 0

is not automatically "better" or "worse" than:

    Cluster 1

The labels are assigned by the algorithm.

If the algorithm is rerun, the numerical labels may even be assigned differently.

What matters is the structure represented by the clusters.

---

# 30. Clustering Is Unsupervised

One of the most important concepts in this chapter:

K-Means does not use the target when creating the clusters.

It learns from the feature space.

For example:

    Latitude
    Longitude
    MedInc
          ↓
       K-Means
          ↓
       Cluster

The target:

    MedHouseVal

is not used to determine the clusters.

It can be used later to evaluate whether the clusters are useful.

---

# 31. Supervised vs Unsupervised

### Supervised Learning

Uses:

    Features + Target
           ↓
        Learn prediction

Examples:

- Linear regression
- Decision trees
- Random forests
- XGBoost

### Unsupervised Learning

Does not use a target:

    Features
       ↓
    Discover structure

Example:

- K-Means clustering

In feature engineering, we can use this discovered structure as new features for a supervised model.

---

# 32. Advantages of Clustering as Feature Engineering

Clustering can:

- Discover hidden groups
- Capture geographic structure
- Capture customer segments
- Simplify complicated relationships
- Create useful categorical features
- Help models learn local patterns
- Combine information from multiple features

---

# 33. Limitations

### 1. Sensitive to Scale

K-Means uses distance, so features with very different scales can distort the clusters.

### 2. Choosing K

You need to choose the number of clusters.

### 3. Initialization

Different initial centroid positions can lead to different results.

This is why `n_init` is useful.

### 4. Not Every Cluster Is Useful

A clustering can look reasonable but fail to improve the predictive model.

Always validate the resulting feature.

### 5. Depends on the Features

K-Means can only discover patterns present in the features provided to it.

Choosing poor input features can lead to poor clusters.

---

# 34. Practical Example

Suppose we have:

    Customer Age
    Customer Income
    Customer Spending

We could apply K-Means:

    Age
    Income
    Spending
        ↓
      K-Means
        ↓
    Customer Cluster

The resulting feature might represent:

    Cluster 0 → low spending customers
    Cluster 1 → high-income customers
    Cluster 2 → younger high-spending customers

The actual interpretation would depend on the data.

The cluster can then be given to another model as an additional feature.

---

# 35. Feature Engineering Mental Model

Think of clustering as:

    Multiple Features
          ↓
    Discover Similar Groups
          ↓
    Assign Cluster Labels
          ↓
    Cluster becomes a Feature
          ↓
    Model can learn different patterns
          ↓
    Validate improvement

The important point is:

> Clustering converts complicated multi-dimensional structure into a feature that a predictive model can use.

---

# 36. Code Pattern to Remember

Basic K-Means feature creation:

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=6
    )

    X["Cluster"] = kmeans.fit_predict(X)

    X["Cluster"] = X["Cluster"].astype("category")

The important function is:

    fit_predict()

It:

1. Fits the clustering algorithm.
2. Predicts the cluster assignment for each training observation.

---

# 37. One-Shot Revision

Remember K-Means like this:

    K-Means
       ↓
    Choose K
       ↓
    Initialize centroids
       ↓
    Assign points to nearest centroid
       ↓
    Move centroids
       ↓
    Repeat
       ↓
    Converge
       ↓
    Generate cluster labels
       ↓
    Use labels as features

---

# 38. Interview Recall

### What is clustering?

Clustering is an unsupervised learning technique that groups observations according to their similarity.

### What is K-Means?

K-Means is a clustering algorithm that assigns observations to the nearest centroid and repeatedly updates the centroids until the clustering converges.

### What does K represent?

The number of clusters.

### What is a centroid?

A point representing the center of a cluster.

### Why can clustering be useful for feature engineering?

It can expose hidden groups and simplify complicated relationships, allowing a predictive model to learn patterns more easily.

### Is the cluster label numerical or categorical?

Although represented using numbers, cluster labels are conceptually categorical.

### Why is scaling important for K-Means?

K-Means uses Euclidean distance, so features with larger numerical scales can dominate the distance calculation.

### What does `n_init` do?

It allows K-Means to try different initial centroid configurations and choose a better clustering.

### What does `max_iter` control?

The maximum number of iterations allowed for the centroid-assignment and centroid-update process.

---

# 39. Quick Revision Table

| Concept               | Meaning                                     |
| --------------------- | ------------------------------------------- |
| Clustering            | Grouping similar observations               |
| Unsupervised Learning | Learning structure without a target         |
| K-Means               | Distance-based clustering algorithm         |
| K                     | Number of clusters                          |
| Centroid              | Center representing a cluster               |
| `n_clusters`          | Number of clusters                          |
| `max_iter`            | Maximum iterations                          |
| `n_init`              | Number of different initialization attempts |
| Cluster Label         | Group assigned to an observation            |
| Cluster Feature       | Cluster label added to the dataset          |
| Euclidean Distance    | Straight-line distance used by K-Means      |
| Voronoi Tessellation  | Regions defined by nearest centroids        |

---

# 40. Final Takeaway

The main lesson of this chapter is:

> **Clustering can be used as a feature-engineering technique to discover groups in the data and represent those groups as new categorical features.**

The complete idea is:

    Original Features
          ↓
    K-Means Clustering
          ↓
    Discover Similar Groups
          ↓
    Assign Cluster Labels
          ↓
    Add Cluster as Feature
          ↓
    Model Learns Simpler Relationships
          ↓
    Validate Performance

K-Means is not being used here only to "find groups."

Its important role in this course is:

**Discover structure → convert structure into a feature → help the predictive model.**

# Part 4 — Chapter 4: Principal Component Analysis (PCA)

## 1. Introduction

In the previous chapter, we used **clustering** as a model-based feature-engineering technique.

Clustering can be thought of as dividing a dataset into groups based on proximity.

In this chapter, we learn another model-based feature-engineering technique:

**Principal Component Analysis (PCA)**

A useful way to think about PCA is:

> Clustering partitions the data based on proximity, while PCA partitions the variation in the data.

PCA can help us:

- Discover important relationships between features
- Create new features
- Reduce the number of features
- Handle highly correlated features
- Reduce noise
- Detect unusual patterns

---

# 2. What Is PCA?

PCA stands for:

**Principal Component Analysis**

The basic idea is to replace the original features with new features that describe the main directions of variation in the data.

Suppose we have:

    Height
    Diameter

Instead of describing an observation using:

    Height + Diameter

PCA can create new features such as:

    Size
    Shape

These new features represent different ways in which the observations vary.

---

# 3. Axes of Variation

Imagine plotting two features:

    Height
       ↑
       |
       |
       |       •
       |     •
       |   •
       | •
       +----------------→ Diameter

The observations may naturally vary mostly along a particular direction.

PCA attempts to find these important directions of variation.

These directions become the new features.

---

# 4. Principal Components

The new features created by PCA are called:

**Principal Components**

For example:

    Original Features
    ├── Height
    └── Diameter

PCA creates:

    PC1
    PC2

Instead of describing the observations using the original coordinate system, PCA describes them using these new components.

---

# 5. Simple Example — Abalone Dataset

The course uses an **Abalone dataset**.

Two features are considered:

    Height
    Diameter

We can imagine two important ways the abalones differ.

### Size

Large:

    High Height
    High Diameter

Small:

    Low Height
    Low Diameter

### Shape

One shape may have:

    High Height
    Low Diameter

while another may have:

    Low Height
    High Diameter

Therefore, we can create two conceptual components:

    Size
    Shape

---

# 6. Principal Components Are Combinations of Original Features

Principal components are created as weighted combinations of the original features.

For example:

    Size = 0.707 × Height + 0.707 × Diameter

and:

    Shape = 0.707 × Height - 0.707 × Diameter

These are not new information.

They are new representations of the information already present in the original features.

---

# 7. Loadings

The numbers used to combine the original features are called:

**Loadings**

For example:

    Size:
        Height   →  0.707
        Diameter →  0.707

    Shape:
        Height   →  0.707
        Diameter → -0.707

The loadings tell us how strongly each original feature contributes to a component.

---

# 8. Understanding the Sign of Loadings

The signs of the loadings are important.

Suppose:

    Height     → +0.707
    Diameter   → +0.707

Both features have the same sign.

This means they vary in the same direction in that component.

For example:

    Height ↑
    Diameter ↑

contributes strongly to the component.

Now consider:

    Height     → +0.707
    Diameter   → -0.707

The signs are opposite.

This represents a contrast:

    Height ↑
    Diameter ↓

or:

    Height ↓
    Diameter ↑

---

# 9. Explained Variance

PCA also tells us how much variation each principal component captures.

This is called:

**Explained Variance**

Suppose:

    PC1 → 96%
    PC2 → 4%

This means PC1 captures most of the variation in the dataset, while PC2 captures much less.

In the abalone example, the conceptual:

    Size

component captures most of the variation, while:

    Shape

captures much less.

---

# 10. Important Warning About Explained Variance

A component explaining a large amount of variance does **not automatically mean it is the best predictor of the target**.

For example:

    PC1 → 90% variance
    PC2 → 5% variance
    PC3 → 5% variance

PC1 contains most of the variation.

But perhaps:

    PC3 → strongest relationship with target

Therefore:

> Variance and predictive usefulness are not the same thing.

This is an important concept.

---

# 11. PCA as Feature Engineering

There are two main ways to use PCA for feature engineering.

### Method 1 — Descriptive Technique

Use PCA to understand the structure of the data.

The components can reveal:

- Important directions of variation
- Feature relationships
- Contrasts between features
- Potential new features to engineer

For example:

    PCA discovers:
        Size is important

Then you might create:

    Area
    Volume
    Total Size

depending on the domain.

---

### Method 2 — Use Components as Features

The principal components themselves can become features.

Instead of:

    Feature 1
    Feature 2
    Feature 3
    Feature 4

you might use:

    PC1
    PC2
    PC3
    PC4

as inputs to the predictive model.

---

# 12. PCA for Dimensionality Reduction

One of the most important applications of PCA is:

**Dimensionality Reduction**

Suppose we have:

    100 original features

but many of them are highly correlated.

PCA can transform them into:

    PC1
    PC2
    PC3
    ...
    PC100

Some components may contain very little variation.

Those low-variance components can potentially be removed.

For example:

    100 Features
          ↓
        PCA
          ↓
    100 Components
          ↓
    Keep important components
          ↓
    Drop near-zero variance components
          ↓
    Smaller feature set

---

# 13. Multicollinearity

PCA can also help when features are highly correlated.

For example:

    Feature A ↔ Feature B
    Feature A ↔ Feature C
    Feature B ↔ Feature C

There may be a lot of redundant information.

PCA can reorganize this information into components.

This can make the representation easier for some machine learning algorithms to work with.

---

# 14. PCA for Decorrelation

PCA transforms correlated features into components that are uncorrelated with one another.

This can be useful because some algorithms struggle when input features are highly correlated.

Therefore:

    Correlated Features
           ↓
          PCA
           ↓
    Uncorrelated Components

This process is called:

**Decorrelation**

---

# 15. PCA for Noise Reduction

Suppose a dataset contains:

    Useful signal
    +
    Background noise

PCA can sometimes concentrate the important signal into a smaller number of components while leaving some noise in lower-variance components.

We can then remove components containing mostly noise.

Conceptually:

    Original Data
         ↓
        PCA
         ↓
    Important Components
         +
    Low-information Components
         ↓
    Keep useful components
         ↓
    Reduce noise

This can improve the signal-to-noise ratio in some applications.

---

# 16. PCA for Anomaly Detection

Unusual patterns can sometimes appear in low-variance components.

Therefore, PCA can also be useful for:

**Anomaly Detection**

An observation that behaves unusually along one of these components may be worth investigating.

The key idea is:

    Normal variation
          ↓
    Main components

    Unusual variation
          ↓
    Potentially visible in unusual components

---

# 17. PCA and Standardization

PCA is sensitive to feature scale.

This is extremely important.

Suppose we have:

    Age:
    0–100

and:

    Income:
    0–1,000,000

Income has a much larger numerical scale.

Without scaling, it can dominate the PCA calculation.

Therefore:

> It is good practice to standardize features before applying PCA.

The course states that its PCA examples use standardized data.

---

# 18. Standardization

A common standardization method is:

    Standardized Value
        =
    (Value - Mean) / Standard Deviation

After standardization, features are generally centered around:

    Mean ≈ 0

with:

    Standard Deviation ≈ 1

This puts the features on a more comparable scale before PCA.

---

# 19. PCA Works With Numerical Features

PCA works with numerical features.

It is designed for features representing quantities such as:

- Measurements
- Counts
- Continuous numerical variables

Categorical variables should not simply be passed directly into PCA.

Categorical data needs appropriate preprocessing first, depending on the modeling workflow.

---

# 20. Outliers and PCA

PCA can be strongly influenced by outliers.

An extreme observation can affect the directions of variation that PCA discovers.

Therefore:

> Consider removing, handling, or constraining extreme outliers before applying PCA when appropriate.

This does not mean every outlier should automatically be removed.

It means that you should understand how extreme observations affect the PCA result.

---

# 21. Example — Automobile Dataset

The course returns to the:

**1985 Automobile dataset**

The target is:

    price

Four features are selected:

    highway_mpg
    engine_size
    horsepower
    curb_weight

These features have useful relationships with the target.

---

# 22. Selecting Features

The course starts with:

    features = [
        "highway_mpg",
        "engine_size",
        "horsepower",
        "curb_weight"
    ]

Then:

    X = df.copy()
    y = X.pop("price")
    X = X.loc[:, features]

So:

    X → selected numerical features

    y → price

---

# 23. Standardizing the Features

The selected features do not naturally have the same scale.

Therefore, the course standardizes them:

    X_scaled = (
        X - X.mean(axis=0)
    ) / X.std(axis=0)

This produces a standardized representation before PCA.

---

# 24. Creating Principal Components with Scikit-Learn

The course uses:

    from sklearn.decomposition import PCA

Then:

    pca = PCA()

    X_pca = pca.fit_transform(X_scaled)

This performs two important operations:

### `fit`

Learns the principal directions from the data.

### `transform`

Converts the original standardized features into principal-component coordinates.

`fit_transform()` performs both operations together.

---

# 25. Naming the Components

The transformed data can be placed into a DataFrame.

The course creates names:

    PC1
    PC2
    PC3
    PC4

For four original features, PCA creates four components.

This gives:

    Original:
    highway_mpg
    engine_size
    horsepower
    curb_weight

    PCA:
    PC1
    PC2
    PC3
    PC4

---

# 26. PCA Loadings

After fitting PCA, the loading information is available through:

    pca.components_

The course converts it into a DataFrame:

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns,
    )

The transpose is used so that:

    Rows    → Original features
    Columns → Principal components

---

# 27. Example Loading Table

The course obtains loadings similar to:

| Feature     |    PC1 |   PC2 |    PC3 |    PC4 |
| ----------- | -----: | ----: | -----: | -----: |
| highway_mpg | -0.492 | 0.771 |  0.070 | -0.398 |
| engine_size |  0.504 | 0.627 |  0.020 |  0.594 |
| horsepower  |  0.500 | 0.014 |  0.731 | -0.464 |
| curb_weight |  0.503 | 0.113 | -0.678 | -0.523 |

The exact values are less important than understanding how to interpret them.

---

# 28. Interpreting PC1

PC1 has approximately:

    highway_mpg → negative
    engine_size → positive
    horsepower  → positive
    curb_weight → positive

This represents a contrast between:

    Large + powerful vehicles
            vs
    Smaller + economical vehicles

The course describes this as a:

**Luxury / Economy axis**

This is a good example of how PCA can reveal a hidden concept that is not represented by a single original feature.

---

# 29. PCA Creates Meaningful Directions

The original features might be:

    engine_size
    horsepower
    curb_weight
    highway_mpg

But PCA can reveal a broader concept:

    Luxury / Economy

This is one of the reasons PCA is useful for feature discovery.

It can reveal patterns that are distributed across several original features.

---

# 30. Looking at Explained Variance

The course examines the amount of variation captured by each component.

This allows us to determine:

    How much of the dataset's variation
    is represented by each component?

A component with high explained variance captures a major direction of variation.

But remember:

> High explained variance does not automatically mean high predictive power.

---

# 31. Mutual Information of PCA Components

The course also calculates MI scores for the principal components.

For example:

    PC1    1.013264
    PC2    0.379156
    PC3    0.306703
    PC4    0.203329

PC1 has the strongest individual relationship with:

    price

But the other components also show meaningful relationships.

This is important because:

> A component can explain relatively little overall variance while still being useful for predicting the target.

---

# 32. PC3 and Vehicle Type

The third component is particularly interesting.

Its loadings show a contrast between:

    horsepower
          and
    curb_weight

The course interprets this as roughly distinguishing:

    Sports cars
          vs
    Wagons

This demonstrates how PCA can uncover meaningful feature relationships that are not immediately obvious.

---

# 33. Creating a New Feature from PCA Insight

After examining PC3, the course creates a ratio:

    sports_or_wagon =
        curb_weight / horsepower

This is an example of using PCA as a **feature discovery tool**.

The process is:

    PCA
      ↓
    Discover important contrast
      ↓
    Understand the underlying relationship
      ↓
    Create a domain-inspired feature
      ↓
    Test its relationship with price

This is different from blindly using every principal component as a model feature.

---

# 34. PCA as a Descriptive Tool

One of the most valuable ideas in this chapter is:

> PCA does not have to be used only for dimensionality reduction.

You can use PCA to understand the structure of your data.

For example:

    PCA
      ↓
    Examine loadings
      ↓
    Understand each component
      ↓
    Identify important contrasts
      ↓
    Invent meaningful features

This makes PCA useful for feature discovery.

---

# 35. Two Main PCA Strategies

## Strategy 1 — Use PCA for Feature Discovery

Use components to understand the dataset.

Example:

    PC3
      ↓
    Represents horsepower vs curb_weight
      ↓
    Understand the domain meaning
      ↓
    Create sports_or_wagon

## Strategy 2 — Use PCA Components Directly

Use:

    PC1
    PC2
    PC3
    ...

as model features.

This can be useful for:

- Dimensionality reduction
- Removing redundancy
- Decorrelation
- Noise reduction

---

# 36. PCA vs Clustering

Both are unsupervised techniques used for feature engineering, but they work differently.

| Clustering                         | PCA                                                    |
| ---------------------------------- | ------------------------------------------------------ |
| Groups similar observations        | Finds major directions of variation                    |
| Produces cluster labels            | Produces continuous components                         |
| Uses centroids                     | Uses principal axes                                    |
| Useful for segmentation            | Useful for representation and dimensionality reduction |
| Creates categorical-style features | Creates numerical features                             |

Mental model:

    Clustering
        ↓
    "Which group does this observation belong to?"

    PCA
        ↓
    "Along which important directions does this observation vary?"

---

# 37. PCA Best Practices

Before applying PCA, remember:

### 1. Use numerical features

PCA requires numerical input.

### 2. Standardize features

PCA is sensitive to scale.

### 3. Consider outliers

Extreme observations can strongly influence PCA.

### 4. Interpret loadings

Do not blindly use components without understanding what they represent.

### 5. Check explained variance

Use explained variance to understand how much variation each component captures.

### 6. Check predictive usefulness

A high-variance component is not automatically a strong predictor.

---

# 38. Common Mistakes

### Mistake 1 — Applying PCA without scaling

If features have very different scales, the largest-scale features can dominate.

### Mistake 2 — Assuming PC1 is always the best predictor

PC1 captures the most variance, not necessarily the most target information.

### Mistake 3 — Ignoring low-variance components

A low-variance component can still contain useful information about the target.

### Mistake 4 — Treating PCA as magic

PCA reorganizes existing information.

It does not create new external information.

### Mistake 5 — Ignoring interpretability

PCA components are combinations of features. Understanding their loadings helps explain what each component represents.

---

# 39. PCA Workflow

A practical PCA workflow is:

    Select numerical features
            ↓
    Handle important outliers
            ↓
    Standardize features
            ↓
    Fit PCA
            ↓
    Generate components
            ↓
    Examine explained variance
            ↓
    Examine loadings
            ↓
    Calculate feature utility if useful
            ↓
    Interpret components
            ↓
    Use components or create new features
            ↓
    Validate model performance

---

# 40. One-Shot Revision

Remember PCA using this sequence:

    Original Numerical Features
              ↓
        Standardization
              ↓
             PCA
              ↓
      Principal Components
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓

Understand Data Use Features
↓ ↓
Find Relationships Reduce Dimensions
↓ ↓
Create Features Remove Redundancy

# Part 5 — Chapter 5: Target Encoding

## 1. Introduction

Most of the feature-engineering techniques covered earlier focused on numerical features.

This chapter introduces **target encoding**, a technique designed primarily for **categorical features**.

Target encoding is another way to convert categorical values into numerical values.

Unlike ordinary encoding techniques such as:

- Label encoding
- One-hot encoding

target encoding also uses the **target variable** when creating the encoded value.

Therefore, target encoding is a:

**Supervised feature-engineering technique**

---

# 2. What Is Target Encoding?

A target encoding replaces each category with a numerical value derived from the target.

Suppose we have:

    make

with categories:

    Toyota
    BMW
    Audi

and the target is:

    price

We can calculate the average price for each manufacturer.

For example:

    Toyota → 15000
    BMW    → 26000
    Audi   → 18000

The categorical feature:

    make

becomes:

    make_encoded

This converts categorical information into numerical information while preserving its relationship with the target.

---

# 3. Simple Mean Encoding

One simple form of target encoding is:

**Mean Encoding**

The encoded value is the average target value for each category.

For example:

    Toyota:
        Prices = 12000, 15000, 18000

    Toyota Mean:
        (12000 + 15000 + 18000) / 3
        = 15000

Therefore:

    Toyota → 15000

---

# 4. Example — Automobile Dataset

The course uses the Automobile dataset.

The target is:

    price

The categorical feature:

    make

can be encoded using the average price for each make.

Example:

    autos["make_encoded"] =
        autos.groupby("make")["price"].transform("mean")

The resulting data contains:

    make
    price
    make_encoded

---

# 5. Example Output

The course gives values similar to:

| make        | price | make_encoded |
| ----------- | ----: | -----------: |
| alfa-romero | 13495 |     15498.33 |
| alfa-romero | 16500 |     15498.33 |
| alfa-romero | 16500 |     15498.33 |
| audi        | 13950 |     17859.17 |
| audi        | 17450 |     17859.17 |
| audi        | 15250 |     17859.17 |
| bmw         | 16430 |     26118.75 |

Every observation belonging to the same category receives the corresponding category-level target statistic.

---

# 6. Why Target Encoding Is Useful

Target encoding is particularly useful when a categorical feature has many unique categories.

For example:

    zipcode

might contain:

    3439 unique categories

Using one-hot encoding would potentially create thousands of additional columns.

Target encoding can instead represent each category with a single numerical value.

Therefore:

    High-cardinality category
            ↓
       Target Encoding
            ↓
    One numerical feature

This can be much more compact than one-hot encoding.

---

# 7. High-Cardinality Features

A categorical feature is called **high-cardinality** when it contains many unique categories.

Examples:

    Zipcode
    Product ID
    City
    Merchant ID
    User ID
    Postal Code

One-hot encoding can become expensive when the number of categories becomes very large.

For example:

    3,000 zipcodes

could result in thousands of one-hot columns.

Target encoding instead creates something like:

    Zipcode → Expected target value

so the feature remains one column.

---

# 8. Target Encoding Is Supervised

This is an important distinction.

One-hot encoding:

    Category
       ↓
    Encoded representation

does not need the target.

Target encoding:

    Category
       +
    Target
       ↓
    Encoded representation

uses information from the target.

Therefore:

> Target encoding is supervised feature engineering.

---

# 9. The Main Problem — Overfitting

Target encoding has an important danger:

**Overfitting**

Suppose a category occurs only once.

Example:

    Category = Mercury

and the only observed price is:

    50000

A simple mean encoding would produce:

    Mercury → 50000

But that value is based on only one observation.

The actual average price of future Mercury vehicles may be very different.

Therefore:

> Rare categories can produce unreliable target encodings.

---

# 10. Rare Categories

Suppose:

    Toyota
        1000 observations

    BMW
        200 observations

    RareBrand
        1 observation

The average target for Toyota is likely to be relatively stable.

The average target for RareBrand is extremely uncertain.

Therefore:

    More observations
        ↓
    More reliable category statistic

    Fewer observations
        ↓
    Less reliable category statistic

This is why target encoding needs protection against rare categories.

---

# 11. Unknown Categories

Another problem occurs when a category appears in future data but was not present in the data used to create the encoding.

For example:

Training data:

    Toyota
    BMW
    Audi

Future data:

    Toyota
    BMW
    Mercedes

If Mercedes was not present when the encoder was trained, there is no category-specific target statistic for Mercedes.

The encoding therefore needs a fallback value.

---

# 12. Smoothing

A solution to both rare-category and unknown-category problems is:

**Smoothing**

Smoothing combines:

    Category-specific average

with:

    Overall target average

The basic idea is:

    Smoothed Encoding
        =
    weight × category average
        +
    (1 - weight) × overall average

The weight depends on how much data is available for that category.

---

# 13. Why Smoothing Works

Consider a category with many observations.

Example:

    Category A
    n = 1000

Its category average is based on lots of data.

Therefore:

    High weight
    on category average

Now consider:

    Category B
    n = 2

Its category average is unreliable.

Therefore:

    Lower weight
    on category average

and:

    Higher weight
    on overall average

So smoothing moves uncertain category estimates toward the global average.

---

# 14. M-Estimate

The course uses an:

**m-estimate**

to determine the smoothing weight.

The formula is:

    weight = n / (n + m)

where:

    n = number of observations in the category

    m = smoothing factor

---

# 15. Understanding the M-Estimate

Suppose:

    n = 3
    m = 2

Then:

    weight = 3 / (3 + 2)

    weight = 0.6

Therefore:

    60% category average
    40% overall average

is used.

---

# 16. Chevrolet Example

The course gives an example involving:

    Chevrolet

Suppose:

    Chevrolet average = 6000
    Overall average = 13285.03

and:

    m = 2

There are:

    n = 3

Chevrolet observations.

Therefore:

    weight = 3 / (3 + 2)

    weight = 0.6

The smoothed encoding becomes:

    Chevrolet
        =
    0.6 × 6000
        +
    0.4 × 13285.03

So the encoding is pulled toward the overall average rather than relying entirely on the small Chevrolet sample.

---

# 17. Effect of Increasing m

The parameter:

    m

controls the amount of smoothing.

A larger:

    m

means stronger smoothing.

This means:

    Larger m
        ↓
    More weight toward global average
        ↓
    More conservative encoding

A smaller:

    m

means:

    Less smoothing
        ↓
    More weight toward category average

---

# 18. Choosing the Smoothing Factor

The appropriate value of:

    m

depends on how noisy the categories are.

If the target varies significantly within categories:

    More data is needed
        ↓
    Larger m may be appropriate

If category averages are relatively stable:

    Less data may be sufficient
        ↓
    Smaller m may work

In practice, the smoothing factor can be treated as a parameter that should be evaluated.

---

# 19. Target Encoding and Data Splits

This is one of the most important parts of target encoding.

Because target encoding uses:

    Target values

we must be careful about how the encoding is created.

If we calculate the encoding using the same target values that we later use to train the model, information can leak from the target into the features.

This can cause:

    Artificially strong training performance
            ↓
    Overfitting
            ↓
    Poor generalization

Therefore, target encoding should be trained using an appropriate independent encoding split or equivalent leakage-safe procedure.

---

# 20. Encoding Split

The course demonstrates the concept by dividing the dataset into two parts.

One part is used to learn the target encoding:

    Encoding Split

The other part becomes the model-training data:

    Pretraining / Model Training Split

Conceptually:

    Original Dataset
          ↓
      Split Data
       ↙       ↘
    Encoding    Model Training
      Data           Data
        ↓              ↓
    Learn target     Transform using
    encoding         learned encoding

This prevents the model-training observations from directly determining their own encoded target value.

---

# 21. MovieLens Example

The course uses the:

**MovieLens1M dataset**

The dataset contains approximately one million movie ratings.

The target is:

    Rating

One feature is:

    Zipcode

There are more than:

    3000 unique zipcodes

This makes Zipcode a good candidate for target encoding.

---

# 22. Creating the Encoding Split

The course uses 25% of the data for creating the target encoding.

Conceptually:

    Full Dataset
         ↓
    ┌────┴────┐
    ↓         ↓
    25%       75%
    ↓         ↓
    Encoding  Model Training
    Split     Split

The target encoder is fitted only using the encoding split.

---

# 23. Why Use a Separate Encoding Split?

Suppose we encode a row using its own target value.

For example:

    Row:
        Zipcode = 12345
        Rating = 5

If the target encoding for zipcode 12345 is calculated using that same rating, the feature contains information directly derived from the target of that observation.

That makes the model's task artificially easy.

Using a separate encoding split avoids this direct contamination.

---

# 24. MEstimateEncoder

The course uses the:

    category_encoders

package.

The encoder is:

    MEstimateEncoder

Example:

    from category_encoders import MEstimateEncoder

Then:

    encoder = MEstimateEncoder(
        cols=["Zipcode"],
        m=5.0
    )

Here:

    Zipcode
        ↓
    Target Encoding

and:

    m = 5.0

controls the smoothing.

---

# 25. Fitting the Encoder

The encoder is fitted using the encoding split:

    encoder.fit(X_encode, y_encode)

This means the encoder learns the relationship:

    Zipcode
        ↔
    Rating

from the designated encoding data.

It should not be fitted using the same data that the final model will use in a way that causes target leakage.

---

# 26. Transforming the Training Data

After fitting the encoder:

    X_train = encoder.transform(X_pretrain)

The model-training data is transformed using the already learned encoding.

The important sequence is:

    Encoding Data
          ↓
    Fit Encoder
          ↓
    Learned Mapping
          ↓
    Transform Model Data

Not:

    Model Data
          ↓
    Calculate target statistics using itself
          ↓
    Train model

---

# 27. Distribution of Encoded Values

The course compares:

    Actual Rating

with:

    Encoded Zipcode

The encoded Zipcode distribution roughly follows the distribution of actual ratings.

This suggests that different zipcodes contain useful information about movie ratings.

Therefore:

    Zipcode
        ↓
    Target Encoding
        ↓
    Useful numerical feature

---

# 28. What Target Encoding Captures

Target encoding captures the relationship:

    Category
        ↓
    Typical target value for that category

For example:

    Zipcode A → average rating around 3.5

    Zipcode B → average rating around 4.2

    Zipcode C → average rating around 2.9

The model can now use this numerical representation.

---

# 29. Target Encoding vs One-Hot Encoding

| One-Hot Encoding                               | Target Encoding                       |
| ---------------------------------------------- | ------------------------------------- |
| Does not use target                            | Uses target                           |
| Creates multiple columns                       | Usually creates one encoded column    |
| Good for low/moderate cardinality              | Useful for high cardinality           |
| Can create very wide datasets                  | More compact                          |
| No target leakage concern from encoding itself | Must carefully prevent target leakage |
| Does not directly encode target relationship   | Encodes category-target relationship  |

---

# 30. Target Encoding vs Ordinal Encoding

Ordinal encoding might represent:

    Toyota → 0
    BMW → 1
    Audi → 2

The numbers do not necessarily represent any meaningful relationship.

Target encoding instead represents something related to the target:

    Toyota → 15000
    BMW → 26000
    Audi → 18000

Therefore, target encoding can provide information about the category's relationship with the target.

---

# 31. When to Use Target Encoding

Target encoding is especially useful for:

### High-cardinality categorical features

Examples:

    Zipcode
    City
    Product ID
    Merchant ID

when one-hot encoding would create too many columns.

### Domain-motivated features

Sometimes domain knowledge suggests that a categorical variable should be useful even though simpler feature-utility analysis does not show a strong relationship.

Target encoding can expose the category's relationship with the target.

---

# 32. When Target Encoding Can Be Dangerous

Be particularly careful when:

- Categories are very rare.
- The dataset is small.
- The target encoding is calculated using the same rows used to train the model.
- Unknown categories appear at inference time.
- Smoothing is not applied.
- The encoding is performed before train/validation splitting.

These situations can result in misleading model performance.

---

# 33. Target Leakage Mental Model

The dangerous workflow is:

    Full Dataset
          ↓
    Calculate target averages
          ↓
    Split into train/validation
          ↓
    Train model

The target information from the validation set has already influenced the feature transformation.

A safer conceptual workflow is:

    Split Data
       ↓
    ┌───────────────┐
    ↓               ↓

Encoding Data Validation / Model Data
↓
Learn Encoding
↓
Transform Other Data
↓
Train / Validate

The exact leakage-safe implementation depends on the validation strategy.

---

# 34. Smoothing Mental Model

Think of smoothing as:

    Category has lots of data
            ↓
    Trust category average more

    Category has little data
            ↓
    Trust global average more

Therefore:

    Reliable category
          ↓
    Category-specific value dominates

    Unreliable category
          ↓
    Global average dominates

---

# 35. Complete Target Encoding Workflow

    Identify categorical feature
             ↓
    Check cardinality
             ↓
    Decide whether target encoding is appropriate
             ↓
    Split data appropriately
             ↓
    Create encoding data
             ↓
    Calculate category-target statistics
             ↓
    Apply smoothing
             ↓
    Handle unknown categories
             ↓
    Transform model data
             ↓
    Train model
             ↓
    Validate without leakage
             ↓
    Compare against baseline

---

# 36. Important Terminology

| Term                | Meaning                                                   |
| ------------------- | --------------------------------------------------------- |
| Target Encoding     | Replacing categories with target-derived numerical values |
| Mean Encoding       | Encoding categories using their average target            |
| Supervised Encoding | Encoding that uses the target                             |
| Cardinality         | Number of unique categories                               |
| High Cardinality    | A categorical feature with many unique values             |
| Smoothing           | Blending category statistics with the global statistic    |
| M-Estimate          | Formula used to determine smoothing weight                |
| Encoding Split      | Data used to learn target encodings                       |
| Unknown Category    | Category not present when the encoder was fitted          |
| Rare Category       | Category with very few observations                       |
| Target Leakage      | Target information improperly entering model features     |

---

# 37. Interview Questions

### What is target encoding?

Target encoding converts categorical values into numerical values derived from the target variable, commonly using the category's mean target value.

### Is target encoding supervised or unsupervised?

Supervised, because it uses the target variable.

### Why is target encoding useful?

It is especially useful for high-cardinality categorical variables where one-hot encoding would create too many features.

### What is the main danger of target encoding?

Target leakage and overfitting.

### Why are rare categories dangerous?

Their target statistics are based on very few observations and may therefore be unreliable.

### What is smoothing?

Smoothing combines the category-specific target statistic with the overall target statistic to make estimates more reliable.

### What is an m-estimate?

A method for calculating the weight given to the category-specific statistic:

    weight = n / (n + m)

### What happens when m increases?

The encoding becomes more strongly pulled toward the overall target average.

### Why use an encoding split?

To prevent the model's training observations from directly determining their own target-derived encoding.

### When is target encoding particularly useful?

When a categorical feature has high cardinality or domain knowledge suggests that the category has useful target information.

---

# 38. Final Mental Model

Remember target encoding as:

    Categorical Feature
            ↓
    Group observations by category
            ↓
    Examine target within each category
            ↓
    Calculate category statistic
            ↓
    Smooth unreliable estimates
            ↓
    Convert category → number
            ↓
    Use numerical feature in model

The most important warning is:

> Target encoding uses the target, so leakage prevention is essential.

---

# 39. Chapter 5 Quick Revision

    Target Encoding
          ↓
    Categorical → Numerical
          ↓
    Uses Target
          ↓
    Supervised Technique
          ↓
    Excellent for High Cardinality
          ↓
    But vulnerable to Overfitting
          ↓
    Use Smoothing
          ↓
    Use Leakage-Safe Encoding
