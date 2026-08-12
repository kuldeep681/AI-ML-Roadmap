# Kaggle — Feature Engineering

## Part 3 — Chapter 3: Creating Features with Clustering

# 1. Introduction

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
