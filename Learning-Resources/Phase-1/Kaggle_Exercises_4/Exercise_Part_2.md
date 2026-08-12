# Kaggle --- Machine Learning Explainability

## Exercise 2 --- Partial Dependence Plots

## 1. Exercise Goal

This exercise uses the **New York City Taxi Fare Prediction** dataset to
practice Partial Dependence Plots, or **PDPs**.

The exercise teaches you how to:

-   create one-dimensional PDPs
-   interpret the shape of a PDP
-   create two-dimensional PDPs
-   understand interactions between features
-   compare PDPs before and after feature engineering
-   understand why PDP magnitude and permutation importance answer
    different questions
-   construct synthetic examples that demonstrate unusual relationships

------------------------------------------------------------------------

# 2. What Is a Partial Dependence Plot?

A Partial Dependence Plot answers:

> How does the model's prediction change as one feature changes, while
> averaging over the other features?

The model is already trained.

We are not changing the model.

Instead, we repeatedly modify one feature and ask:

``` text
What would the model predict if this feature had this value?
```

Then we average the predictions over many observations.

Conceptually:

``` text
Choose feature
      ↓
Set it to a particular value
      ↓
Predict using the trained model
      ↓
Repeat for many values
      ↓
Average predictions
      ↓
Plot the relationship
```

------------------------------------------------------------------------

# 3. Basic PDP Code

The exercise uses:

``` python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    first_model,
    val_X,
    ["pickup_longitude"]
)

plt.show()
```

The x-axis represents the feature value.

The y-axis represents the model's partial dependence, meaning the
model's average prediction as that feature is changed.

------------------------------------------------------------------------

# 4. Question 1 --- Why Does Pickup Longitude Have a U-Shape?

The PDP for `pickup_longitude` has a U-like shape.

The important interpretation is:

> Location affects taxi fare partly because location is acting as a
> proxy for distance and geographic position.

Values around the central part of the city can correspond to shorter or
more common trips, while locations farther away can correspond to longer
trips or different fare patterns.

The exact shape is a property of the trained model and the data.

It should not automatically be interpreted as:

> Longitude itself causes the fare to increase.

Instead, longitude contains information about where the ride occurs.

------------------------------------------------------------------------

# 5. Creating PDPs for All Features

The exercise asks you to create PDPs for all base features.

A convenient pattern is:

``` python
for feat_name in base_features:
    PartialDependenceDisplay.from_estimator(
        first_model,
        val_X,
        [feat_name]
    )
    plt.show()
```

This allows you to compare the shapes of the different relationships.

------------------------------------------------------------------------

# 6. Why Location Features Can Have Similar Shapes

The four location variables are:

``` text
pickup_longitude
pickup_latitude
dropoff_longitude
dropoff_latitude
```

Each describes where the trip starts or ends.

Because taxi fare depends strongly on trip distance, each location
coordinate can indirectly contain information about:

``` text
Trip length
```

Therefore, location PDPs can show non-linear patterns even though
longitude and latitude themselves are not direct distance measurements.

------------------------------------------------------------------------

# 7. Question 2 --- Two-Dimensional PDP

A 2D PDP examines two features simultaneously.

The exercise uses:

``` python
fig, ax = plt.subplots(figsize=(8, 6))

fnames = [
    ("pickup_longitude", "dropoff_longitude")
]

PartialDependenceDisplay.from_estimator(
    first_model,
    val_X,
    fnames,
    ax=ax
)

plt.show()
```

A 2D PDP answers:

> What does the model predict for combinations of these two feature
> values?

------------------------------------------------------------------------

# 8. Interpreting the 2D Location Plot

For pickup and dropoff longitude, the key idea is the relationship
between:

``` text
pickup longitude
```

and:

``` text
dropoff longitude
```

If the two values are relatively close, the longitudinal component of
the trip is smaller.

If they are far apart, the trip is more likely to be longer.

Therefore, the model can predict higher fares for combinations
representing larger movement.

This is an example of using PDPs to understand a relationship involving
two variables.

------------------------------------------------------------------------

# 9. Question 3 --- Estimating Fare Savings

The exercise gives a ride:

``` text
Starting longitude = -73.955
Ending longitude = -74.000
```

and asks how much could be saved by changing the starting longitude to:

``` text
-73.980
```

The expected exercise answer is approximately:

``` text
$6
```

The important lesson is not the exact number.

The exercise is teaching you how to use a PDP as a visual prediction
tool:

``` text
Read prediction at original feature value
        ↓
Read prediction at alternative value
        ↓
Compare them
        ↓
Estimate prediction difference
```

------------------------------------------------------------------------

# 10. Question 4 --- Add Absolute Distance Features

The exercise returns to the engineered features:

``` python
data["abs_lon_change"] = abs(
    data.dropoff_longitude - data.pickup_longitude
)

data["abs_lat_change"] = abs(
    data.dropoff_latitude - data.pickup_latitude
)
```

These features provide a more direct representation of travel distance.

The model is then retrained with them.

------------------------------------------------------------------------

# 11. What Changes in the PDP?

Before adding absolute-distance features, location coordinates were
partly acting as proxies for distance.

After adding:

``` text
abs_lon_change
abs_lat_change
```

the model has a more direct way to represent movement.

Therefore, the PDP for a raw location coordinate becomes much less
dominated by distance-related behavior.

The key comparison is:

``` text
Before:
Location → indirect distance information

After:
Distance features → direct distance information
Location → more remaining geographic effects
```

This is a very important feature-engineering insight.

------------------------------------------------------------------------

# 12. PDP as a Diagnostic Tool

This comparison demonstrates that explainability can tell us:

> What information is the model extracting from a feature?

If a feature's PDP changes substantially after adding another engineered
feature, the original feature may have been acting as a proxy.

------------------------------------------------------------------------

# 13. Question 5 --- Does a Steeper PDP Guarantee Higher Permutation Importance?

Suppose:

``` text
feat_A
```

has a steeper PDP than:

``` text
feat_B
```

Does that guarantee:

``` text
Permutation importance(A)
>
Permutation importance(B)
```

No.

The reason is that the two techniques measure different things.

### PDP

Measures:

> How does the prediction change as this feature changes?

### Permutation importance

Measures:

> How much does model performance deteriorate when this feature's
> information is destroyed?

A feature can have a steep PDP but still have lower permutation
importance.

------------------------------------------------------------------------

# 14. Why Can This Happen?

Permutation importance depends on things such as:

-   feature distribution
-   range of values
-   how often the feature varies
-   redundancy with other features
-   interactions
-   how much the model relies on the feature

Therefore:

``` text
Steep PDP
    ≠
Guaranteed high permutation importance
```

------------------------------------------------------------------------

# 15. Question 6 --- Construct a Non-Linear PDP

The exercise creates:

``` python
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2
```

so both features are uniformly distributed approximately over:

``` text
[-2, 2]
```

The target is constructed as:

``` python
y = (
    -2 * X1 * (X1 < -1)
    + X1
    - 2 * X1 * (X1 > 1)
    - X2
)
```

The resulting relationship is approximately:

``` text
x < -1      → negative slope
-1 to 1     → positive slope
x > 1       → negative slope
```

Therefore the PDP for `X1` has:

``` text
negative slope
      ↓
positive slope
      ↓
negative slope
```

This exercise demonstrates that PDPs can represent complex non-linear
relationships.

------------------------------------------------------------------------

# 16. Understanding the Boolean Conditions

Consider:

``` python
X1 < -1
```

This produces a Boolean mask.

Similarly:

``` python
X1 > 1
```

produces another mask.

The formula selectively applies additional terms to different parts of
the feature range.

This is a useful pattern for constructing synthetic data with controlled
relationships.

------------------------------------------------------------------------

# 17. Question 7 --- Flat PDP but High Permutation Importance

This is one of the most important conceptual questions in the exercise.

The exercise creates:

``` python
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

y = X1 * X2
```

So:

``` text
y = X1 × X2
```

There is a strong interaction between `X1` and `X2`.

------------------------------------------------------------------------

# 18. Why Is the PDP for X1 Flat?

Because `X2` is approximately symmetric around zero.

For a fixed `X1`:

``` text
E[y | X1]
=
E[X1 × X2 | X1]
=
X1 × E[X2]
```

and:

``` text
E[X2] ≈ 0
```

so:

``` text
E[y | X1] ≈ 0
```

for almost every value of `X1`.

Therefore:

``` text
PDP(X1) ≈ flat
```

even though `X1` is genuinely important to the model.

------------------------------------------------------------------------

# 19. Why Can Permutation Importance Still Be High?

The model needs both features to calculate:

``` text
X1 × X2
```

If we randomly shuffle `X1`, the relationship between:

``` text
X1
```

and:

``` text
X2
```

is destroyed.

The model loses important information.

Therefore performance drops significantly.

So:

``` text
PDP(X1) = flat
```

while:

``` text
Permutation importance(X1) = high
```

This is a powerful demonstration of **feature interaction**.

------------------------------------------------------------------------

# 20. Major Lesson From Exercise 2

PDPs and permutation importance should not be treated as competing
methods.

They answer different questions.

``` text
Permutation Importance
→ How much does the model rely on this feature?

PDP
→ How does the model's prediction change as this feature changes?

2D PDP
→ How do two features jointly affect predictions?
```

A feature can be:

``` text
Highly important
+
Flat PDP
```

if its effect is mainly expressed through interactions.

------------------------------------------------------------------------

# 21. Common Mistakes

### Mistake 1 --- Treating a PDP as causality

A PDP describes model behavior, not a causal relationship in the real
world.

### Mistake 2 --- Assuming a steep PDP means high importance

PDP slope and permutation importance measure different properties.

### Mistake 3 --- Ignoring interactions

A flat PDP does not prove that a feature is useless.

### Mistake 4 --- Reading a location feature literally

Longitude and latitude may be proxies for distance, geography, or other
correlated information.

------------------------------------------------------------------------

# 22. Exercise Mental Model

``` text
Trained model
      ↓
Create PDP
      ↓
Understand feature → prediction relationship
      ↓
Create 2D PDP
      ↓
Investigate interactions
      ↓
Engineer better features
      ↓
Compare PDPs
      ↓
Understand what the model is actually learning
```

------------------------------------------------------------------------

# 23. Final Takeaway

Partial Dependence Plots answer one of the most useful explainability
questions:

> **How does the model's prediction change when this feature changes?**

But the exercise also teaches the limitation:

> A feature can be important even when its individual PDP is flat.

That happens when the feature's effect depends strongly on other
features.

This is why PDPs, permutation importance, and later SHAP techniques
should be used together.
