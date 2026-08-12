# Kaggle Feature Engineering — Exercise Notes
## Exercise 4: Clustering With K-Means

This note is based on the Kaggle exercise notebook `exercise-clustering-with-k-means.ipynb`.

---

## 1. What This Exercise Is About

This exercise introduces **K-Means clustering as a feature-engineering technique**.

The central idea is:

> Use an unsupervised model to discover groups in the feature space, then use those discovered relationships as new model features.

The exercise has two main feature types:

1. Cluster labels
2. Cluster-distance features

It also emphasizes an important practical issue:

**Feature scaling matters for K-Means.**

---

# Part 1 — Why Scaling Matters

## 2. K-Means Is Distance-Based

K-Means groups observations according to their distance from cluster centroids.

Therefore, the numerical scale of a feature directly affects the distance calculation.

Suppose one feature ranges approximately from:

`0 to 1`

and another ranges from:

`0 to 100000`

The second feature can dominate the distance simply because its numerical values are much larger.

This means:

> A feature can receive too much influence merely because of its scale.

---

## 3. When Should Features Be Rescaled?

The exercise gives three cases.

### Case 1 — Latitude and Longitude

`Latitude`

`Longitude`

These are directly comparable geographic coordinates.

The exercise considers them a case where rescaling is not necessarily required in the same way as unrelated physical quantities.

The important principle is to consider whether the features are already on comparable scales and have comparable meaning.

---

### Case 2 — Lot Area and Living Area

`LotArea`

`GrLivArea`

Both are area measurements, but their ranges and distributions can differ.

Rescaling is generally reasonable because K-Means is sensitive to magnitude.

---

### Case 3 — Number of Doors and Horsepower

`Number of Doors`

`Horsepower`

These are fundamentally different measurements and have different scales.

Without scaling, horsepower can dominate the distance calculation.

This is a strong case for rescaling.

---

## 4. The General Rule

Ask:

> Are these features directly comparable in scale and meaning?

If yes, rescaling may not be necessary.

If no, scaling is usually safer for K-Means.

The notebook's key warning is:

> Features with larger numerical values will be weighted more heavily by a distance-based algorithm.

---

# Part 2 — Standardizing the Features

## 5. Standardization Formula

The exercise standardizes the clustering features using:

`(X - mean) / standard_deviation`

This converts each feature approximately to:

`mean ≈ 0`

`standard deviation ≈ 1`

This is commonly called standardization or z-score scaling.

---

## 6. Why Standardization Helps

After standardization:

`LotArea`

and:

`GrLivArea`

are placed on comparable scales.

Therefore, K-Means is less likely to let one feature dominate simply because its raw values are larger.

The workflow becomes:

`Original features`

↓

`Standardize`

↓

`K-Means`

---

# Part 3 — Creating Cluster Labels

## 7. Features Used for Clustering

The exercise uses:

- `LotArea`
- `TotalBsmtSF`
- `FirstFlrSF`
- `SecondFlrSF`
- `GrLivArea`

These describe different aspects of the size and area of a house.

The idea is to allow K-Means to discover groups of houses with similar spatial/size characteristics.

---

## 8. Choosing the Number of Clusters

The exercise specifies:

`n_clusters = 10`

This means K-Means should divide the observations into:

`10 groups`

The cluster count is represented by `K`.

Therefore:

`K = 10`

---

## 9. Fitting K-Means

The notebook uses:

`KMeans(n_clusters=10, n_init=10, random_state=0)`

Important parameters:

### `n_clusters=10`

Creates ten clusters.

### `n_init=10`

Runs the algorithm from multiple centroid initializations and selects a good result.

### `random_state=0`

Makes the random initialization reproducible.

---

## 10. Creating the Cluster Feature

The exercise uses:

`kmeans.fit_predict(X_scaled)`

This does two things:

1. Fits the K-Means model.
2. Returns the cluster assigned to every observation.

The result is stored as:

`X["Cluster"]`

So the original dataset gains a new feature:

`Cluster`

---

## 11. What Does a Cluster Label Mean?

A value such as:

`Cluster = 3`

does not mean:

`3 is better than 2`

or:

`Cluster 3 is twice Cluster 1`

The number is simply an identifier.

Conceptually:

`Cluster 0`

`Cluster 1`

`Cluster 2`

...

are categories discovered by the algorithm.

---

## 12. Why Can a Cluster Label Help a Model?

Suppose houses naturally form groups based on:

- Lot size
- Basement size
- First-floor area
- Second-floor area
- Living area

A prediction model might find it easier to learn:

`House → Cluster`

and then:

`Cluster + original features → SalePrice`

rather than learning the entire complicated structure from scratch.

This is a form of:

**Divide and conquer**

The model can learn different patterns for different groups.

---

# Part 4 — Visualizing the Clusters

## 13. Why Plot the Clusters?

The notebook creates plots showing:

- Feature values
- `SalePrice`
- Cluster labels

The cluster label is converted to a categorical type for visualization.

The purpose is to ask:

> Do the discovered groups correspond to meaningful patterns in the data?

A useful cluster feature should ideally capture some structure rather than merely produce arbitrary groups.

---

## 14. Cluster Labels as Feature Engineering

The complete process is:

`House features`

↓

`Standardization`

↓

`K-Means`

↓

`Cluster assignment`

↓

`Cluster column`

↓

`Prediction model`

This is different from using K-Means as the final prediction model.

K-Means is being used to **create a feature**.

---

# Part 5 — Evaluating Cluster Labels

## 15. Scoring the Dataset

The notebook uses:

`score_dataset(X, y)`

The scoring function:

- Factorizes categorical features.
- Uses XGBoost.
- Performs 5-fold cross-validation.
- Uses RMSLE.

The purpose is to compare model performance with and without the cluster feature.

The important rule is:

> The cluster feature is valuable only if it provides useful information to the final predictive model.

---

# Part 6 — Cluster-Distance Features

## 16. A Different Way to Use K-Means

K-Means does not have to produce only one categorical label.

It can also tell us:

> How far is this observation from each cluster centroid?

For ten clusters, every observation can therefore receive ten distances.

Conceptually:

`House`

↓

Distance to Cluster 0 centroid

Distance to Cluster 1 centroid

...

Distance to Cluster 9 centroid

These become numerical features.

---

## 17. `fit_predict()` vs `fit_transform()`

This distinction is one of the most important coding ideas in the exercise.

### `fit_predict()`

Returns:

`Cluster label`

Example:

`3`

Meaning:

> This observation belongs to cluster 3.

### `fit_transform()`

Returns:

`Distance to each centroid`

Example conceptually:

`[distance_0, distance_1, ..., distance_9]`

This provides much richer information.

---

## 18. Creating the Distance Features

The notebook uses:

`X_cd = kmeans.fit_transform(X_scaled)`

This produces a matrix with one column per centroid.

Because there are ten clusters:

`10 clusters`

↓

`10 distance features`

The notebook then creates column names:

`Centroid_0`

`Centroid_1`

...

`Centroid_9`

and joins these columns to the original feature set.

---

## 19. What Does a Distance Feature Mean?

Suppose:

`Centroid_3 = 0.2`

and:

`Centroid_7 = 4.5`

The observation is much closer to centroid 3 than centroid 7.

The model receives this information continuously instead of only receiving:

`Cluster = 3`

Therefore, distance features preserve more information about the observation's position in the feature space.

---

# Part 7 — Cluster Labels vs Cluster Distances

| Cluster Label | Cluster Distance |
|---|---|
| Categorical-style feature | Numerical features |
| One value per observation | One value per centroid |
| Tells you nearest group | Tells you proximity to every group |
| Compact | More expressive |
| Created with `fit_predict()` | Created with `fit_transform()` |

Mental model:

`Cluster label`

→

"Which group am I in?"

`Cluster distances`

→

"How close am I to every group?"

---

# Part 8 — Why Distance Features Can Be Useful

A hard cluster assignment throws away information.

Imagine two houses:

House A:

`Very close to Cluster 2`

House B:

`Barely closer to Cluster 2 than Cluster 3`

Both could receive:

`Cluster = 2`

But their positions are not equally similar to the cluster center.

Distance features preserve this difference.

This can make them useful to a downstream model.

---

# Part 9 — Important K-Means Concepts

## Centroid

The center representing a cluster.

## K

The number of clusters.

## Cluster Label

The identifier assigned to an observation.

## Distance

How far an observation is from a centroid.

## `n_clusters`

Number of clusters requested.

## `n_init`

Number of initialization attempts.

## `random_state`

Controls reproducibility.

## `fit_predict()`

Fits K-Means and returns labels.

## `fit_transform()`

Fits K-Means and returns distances to centroids.

---

# Part 10 — Important Pandas and Scikit-Learn Operations

### Standardization

`(X - X.mean(axis=0)) / X.std(axis=0)`

Makes numerical features comparable in scale.

### `KMeans(...)`

Creates the clustering model.

### `fit_predict()`

Creates cluster labels.

### `fit_transform()`

Creates centroid-distance features.

### `pd.DataFrame(...)`

Converts the distance matrix into a dataframe.

### `join()`

Adds the generated features to the original dataset.

### `astype("category")`

Marks the cluster labels as categorical for analysis/visualization.

---

# Part 11 — Common Mistakes

## Mistake 1 — Forgetting scaling

If features have very different scales, large-valued features can dominate distance calculations.

## Mistake 2 — Treating cluster IDs as ordered numbers

`Cluster 9` is not necessarily greater or better than `Cluster 2`.

## Mistake 3 — Assuming clustering automatically improves the model

Clusters must be validated.

## Mistake 4 — Confusing labels with distances

`fit_predict()` gives group membership.

`fit_transform()` gives distances.

## Mistake 5 — Ignoring domain meaning

A mathematically valid cluster is not automatically a useful feature.

---

# Part 12 — Exercise Mental Model

Remember the exercise as:

`Choose related numerical features`

↓

`Standardize`

↓

`K-Means`

↓

`Option 1: Cluster labels`

or:

`Option 2: Centroid distances`

↓

`Add features`

↓

`Train predictive model`

↓

`Cross-validate`

↓

`Keep useful representation`

---

# 13. What You Should Be Able to Do

After this exercise, you should be able to:

- Explain why K-Means is useful for feature engineering.
- Explain why scaling matters.
- Standardize numerical features.
- Choose clustering features.
- Configure K-Means.
- Create cluster labels.
- Understand cluster IDs.
- Visualize clusters against the target.
- Score a model with cluster labels.
- Generate centroid-distance features.
- Explain `fit_predict()` vs `fit_transform()`.
- Use cluster distances as numerical features.

---

# 14. Final Takeaway

K-Means can be used as a **feature generator**.

The two main outputs are:

`Cluster label`

and:

`Distance to cluster centroids`

The deeper idea is:

> Unsupervised learning can discover structure that a supervised model can then use as additional information.

The complete workflow is:

`Raw features`

↓

`Scale`

↓

`Discover groups`

↓

`Represent group structure`

↓

`Add representation to dataset`

↓

`Validate`

That is the feature-engineering role of K-Means.
