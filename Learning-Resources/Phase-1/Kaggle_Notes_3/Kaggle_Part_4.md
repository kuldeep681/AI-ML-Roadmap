# Kaggle — Feature Engineering

## Part 4 — Chapter 4: Principal Component Analysis (PCA)

# 1. Introduction

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
