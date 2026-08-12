# Kaggle --- Machine Learning Explainability

## Exercise 1 --- Permutation Importance

## 1. Exercise Goal

This exercise uses the **New York City Taxi Fare Prediction** dataset to
practice permutation importance.

The main goals are:

-   calculate permutation importance for a trained model
-   interpret the importance values
-   investigate why some location features matter more than others
-   create better distance-related features
-   understand why permutation importance is not the same as a
    coefficient, correlation, or causal effect

The model predicts taxi fare using features such as:

-   `pickup_longitude`
-   `pickup_latitude`
-   `dropoff_longitude`
-   `dropoff_latitude`
-   `passenger_count`

The exercise uses a Random Forest model and evaluates the importance of
features on validation data.

------------------------------------------------------------------------

# 2. Permutation Importance --- Core Idea

Permutation importance asks:

> What happens to model performance if I randomly shuffle one feature
> while keeping everything else unchanged?

The model itself is **not retrained**.

The process is:

1.  Train the model normally.
2.  Choose one feature.
3.  Shuffle that feature in the validation data.
4.  Make predictions with the shuffled data.
5.  Measure how much model performance decreases.
6.  Restore the original feature.
7.  Repeat for every feature.

If shuffling a feature causes a large performance drop, the model relied
heavily on that feature.

If shuffling causes almost no change, the model did not rely much on
that feature.

------------------------------------------------------------------------

# 3. Why Use Validation Data?

Permutation importance should normally be calculated on data that was
not used to fit the model.

Why?

Because the goal is to measure:

> How much does this feature matter for the model's predictive behavior
> on unseen data?

Using validation data gives a more realistic picture of model reliance.

------------------------------------------------------------------------

# 4. Exercise Setup

The notebook creates a train/validation split and fits the first model
using the original location features.

Conceptually:

``` python
base_features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]
```

The model is then fitted on the training set.

------------------------------------------------------------------------

# 5. Question 1 --- Which Features Are Used?

The first model uses:

-   pickup longitude
-   pickup latitude
-   dropoff longitude
-   dropoff latitude
-   passenger count

Before calculating permutation importance, understand what each feature
represents.

The four location features describe:

``` text
Pickup location
        +
Dropoff location
```

while passenger count describes the number of passengers.

The important point is that these raw location coordinates can
indirectly contain information about **trip distance**.

------------------------------------------------------------------------

# 6. Question 2 --- Calculate Permutation Importance

The exercise asks you to create a `PermutationImportance` object.

The core pattern is:

``` python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(
    first_model,
    random_state=1
).fit(val_X, val_y)

eli5.show_weights(
    perm,
    feature_names=val_X.columns.tolist()
)
```

The important part is:

``` python
.fit(val_X, val_y)
```

because the importance is being measured using validation data.

------------------------------------------------------------------------

# 7. How to Read the Importance Table

The first number represents approximately:

> How much the model's performance decreased when this feature was
> shuffled.

A larger positive value means greater model reliance.

The `±` value represents variation caused by repeating the permutation
process.

For example:

``` text
0.15 ± 0.03
```

means the feature caused an average performance decrease of about
`0.15`, with some variation across repeated shuffles.

------------------------------------------------------------------------

# 8. Negative Permutation Importance

Permutation importance can sometimes be negative.

A negative value does **not** mean:

> The feature is actively harmful in the real world.

It means that after shuffling, the model happened to perform slightly
better.

This can happen because of randomness, especially on small validation
datasets.

Therefore:

``` text
Negative importance
        ↓
Usually interpreted as
        ↓
Very weak / uncertain importance
```

rather than a causal statement that the feature is bad.

------------------------------------------------------------------------

# 9. Question 3 --- Why Do Latitude Features Matter More?

The exercise observes that the latitude features have greater importance
than the longitude features.

At first this may seem strange.

You might expect:

``` text
pickup longitude ≈ pickup latitude
dropoff longitude ≈ dropoff latitude
```

because both describe location.

Possible explanations include:

-   the city's geography is not symmetric
-   roads and neighborhoods are distributed differently in the two
    directions
-   taxi trips may have different distance patterns along latitude and
    longitude
-   Manhattan and surrounding areas have geographic structure that makes
    one coordinate more informative
-   longitude and latitude may act as indirect proxies for distance

The important lesson is:

> A feature importance result can reveal something worth investigating,
> but it does not automatically tell you why the relationship exists.

------------------------------------------------------------------------

# 10. Question 4 --- Engineer Distance Features

The exercise then creates:

``` python
data["abs_lon_change"] = abs(
    data.dropoff_longitude - data.pickup_longitude
)

data["abs_lat_change"] = abs(
    data.dropoff_latitude - data.pickup_latitude
)
```

These represent the absolute change in longitude and latitude.

Conceptually:

``` text
Pickup location
      ↓
Dropoff location
      ↓
Difference
      ↓
Absolute difference
      ↓
Approximate movement in each direction
```

The model is then trained with these additional features.

------------------------------------------------------------------------

# 11. Why Absolute Values?

Suppose:

``` text
pickup = -73.95
dropoff = -74.00
```

The raw difference is:

``` text
-0.05
```

But if the trip goes in the opposite direction:

``` text
pickup = -74.00
dropoff = -73.95
```

the difference becomes:

``` text
+0.05
```

For distance, direction usually should not matter.

Therefore:

``` text
abs(dropoff - pickup)
```

gives:

``` text
0.05
```

in both cases.

------------------------------------------------------------------------

# 12. What Happens After Adding Distance Features?

The importance results now show that **distance traveled becomes much
more important than the individual location coordinates**.

This makes intuitive sense.

The original longitude/latitude features were partly acting as proxies
for:

``` text
How far did the taxi travel?
```

Once we explicitly give the model:

``` text
abs_lon_change
abs_lat_change
```

the model can use those direct distance-related signals.

This is an important feature-engineering lesson:

> If a model is using a feature as an indirect proxy for something
> meaningful, explicitly creating that meaningful feature can make the
> model easier to understand.

------------------------------------------------------------------------

# 13. Location Still Matters

Even after adding distance features, pickup and dropoff location still
retain some importance.

This means:

``` text
Fare ≠ distance only
```

Location itself can matter because different areas can have different:

-   demand
-   traffic
-   trip patterns
-   road structures
-   airport effects
-   neighborhood effects
-   pricing-related patterns

The exercise also notes that dropoff location becomes slightly more
important than pickup location.

This is a useful example of using explainability to generate hypotheses
for further investigation.

------------------------------------------------------------------------

# 14. Question 5 --- Does Feature Scale Affect Permutation Importance?

The exercise asks whether the small numerical values of:

``` text
abs_lon_change
abs_lat_change
```

could explain their permutation importance.

The answer is:

> No.

Permutation importance is based on **how much model performance changes
when the feature's values are shuffled**.

It is not simply based on the numerical magnitude of the feature.

If you multiply a feature by 100 and retrain an appropriate model, the
permutation importance concept does not automatically become 100 times
larger.

For example:

``` text
original:
abs_lon_change = 0.05

scaled:
abs_lon_change = 5
```

The numerical representation changed, but the underlying information did
not.

------------------------------------------------------------------------

# 15. Why This Is Different From Coefficients

For a linear regression model, coefficient magnitude can be affected by
feature scale.

For example:

``` text
x
```

and:

``` text
100x
```

can produce very different coefficient values.

Permutation importance works differently.

It measures:

``` text
Performance before shuffling
        -
Performance after shuffling
```

Therefore, it is not simply a measure of numerical magnitude.

------------------------------------------------------------------------

# 16. Question 6 --- Can We Say Latitude Travel Is More Expensive?

The exercise asks:

> If latitudinal distance has higher permutation importance than
> longitudinal distance, does that prove traveling the same latitude
> distance costs more?

No.

Permutation importance tells us:

> How much the model relies on the feature for prediction.

It does **not** directly tell us:

> How much the target changes per unit of that feature.

Therefore:

``` text
Higher permutation importance
        ≠
Higher price per unit
```

The higher importance could be caused by:

-   different feature distributions
-   different ranges
-   interactions with other variables
-   geographic structure
-   model behavior

To understand the actual effect of a feature, techniques such as
**Partial Dependence Plots** are more appropriate.

------------------------------------------------------------------------

# 17. Key Distinction

Remember:

``` text
Permutation Importance
        ↓
How much does the model rely on this feature?
```

Whereas:

``` text
Partial Dependence
        ↓
How does changing this feature affect predictions?
```

And:

``` text
SHAP
        ↓
How did a feature contribute to an individual prediction?
```

These answer different questions.

------------------------------------------------------------------------

# 18. Exercise Workflow

The complete exercise can be remembered as:

``` text
Train model
    ↓
Calculate permutation importance
    ↓
Identify important features
    ↓
Question surprising results
    ↓
Engineer meaningful features
    ↓
Recalculate importance
    ↓
Interpret the new results
    ↓
Avoid confusing importance with causality
```

------------------------------------------------------------------------

# 19. Important Code Concepts

### `PermutationImportance`

Calculates permutation-based feature importance.

``` python
PermutationImportance(model, random_state=1)
```

### `.fit(val_X, val_y)`

Calculates importance using validation data.

### `eli5.show_weights()`

Displays the importance results.

### `abs()`

Converts directional differences into magnitude.

``` python
abs(dropoff - pickup)
```

------------------------------------------------------------------------

# 20. Common Mistakes

### Mistake 1 --- Treating importance as causality

A highly important feature does not prove that changing it will cause
the target to change.

### Mistake 2 --- Assuming feature magnitude determines importance

Small numerical values can still be highly important.

### Mistake 3 --- Ignoring unexpected results

Unexpected importance patterns are often clues worth investigating.

### Mistake 4 --- Confusing importance with effect size

Importance answers:

> How much does the model rely on this feature?

It does not directly answer:

> How much does the prediction increase per unit?

------------------------------------------------------------------------

# 21. Final Takeaway

Permutation importance is one of the simplest ways to look inside a
trained model.

Its most useful question is:

> **Which features does my model actually rely on for its predictions?**

In this exercise, the biggest lesson is that raw location features were
partly acting as proxies for trip distance.

By explicitly creating:

``` text
abs_lon_change
abs_lat_change
```

the model's behavior became easier to interpret.

The deeper lesson is:

> **Explainability can guide feature engineering.**
