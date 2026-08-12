# Kaggle --- Machine Learning Explainability

## Exercise 4 --- Advanced Uses of SHAP Values

## 1. Exercise Goal

This is the final exercise of the Machine Learning Explainability
course.

The exercise uses SHAP values from the hospital readmission model to
understand:

-   the distribution of feature effects
-   why SHAP effect range is not the same as feature importance
-   how binary features can have different typical effects
-   how SHAP color patterns reveal interactions
-   how SHAP dependence plots provide more detail than summary plots
-   how two features can both look important but behave very differently

The central idea is:

> Aggregating SHAP values gives powerful model-level insights while
> still preserving information about the direction and distribution of
> individual feature effects.

------------------------------------------------------------------------

# 2. Review --- What Does a SHAP Value Represent?

For an individual observation:

``` text
SHAP value > 0
```

means the feature pushes the prediction upward.

``` text
SHAP value < 0
```

means the feature pushes the prediction downward.

The magnitude tells us how strongly the feature contributes.

A SHAP summary plot displays many individual SHAP values at once.

------------------------------------------------------------------------

# 3. Summary Plot Structure

A SHAP summary plot contains many dots.

Each dot represents:

``` text
One observation
+
One feature
+
That feature's SHAP contribution
```

The plot provides three major pieces of information.

### Vertical position

Which feature the dot belongs to.

### Horizontal position

The SHAP value.

Farther right:

``` text
More positive contribution
```

Farther left:

``` text
More negative contribution
```

### Color

The actual feature value.

Typically:

``` text
Pink → high feature value
Blue → low feature value
```

This allows you to understand both:

``` text
How large is the effect?
```

and:

``` text
Does a high or low feature value cause it?
```

------------------------------------------------------------------------

# 4. Question 1 --- Which Feature Has the Larger Range?

The exercise compares:

``` text
diag_1_428
```

and:

``` text
payer_code_?
```

The question asks which feature has the larger range of effects:

``` text
maximum SHAP value
-
minimum SHAP value
```

The answer is:

``` text
diag_1_428
```

Its SHAP values extend farther between the most negative and most
positive effects.

------------------------------------------------------------------------

# 5. Why `diag_1_428` Has a Large Effect Range

Most observations may have relatively small SHAP values for this
feature.

But a small number of observations can have very large positive
contributions.

This creates a large overall range.

Therefore:

``` text
Large SHAP range
```

does not necessarily mean:

``` text
Large effect for everyone
```

It may mean:

``` text
Very large effect for a small group
```

------------------------------------------------------------------------

# 6. Question 2 --- Is SHAP Effect Range a Good Importance Measure?

No.

The range:

``` text
max(SHAP)
-
min(SHAP)
```

is very sensitive to extreme observations.

A few outliers can make the range huge.

Therefore, effect range is not a reliable substitute for permutation
importance.

------------------------------------------------------------------------

# 7. Range vs Permutation Importance

These measures answer different questions.

### SHAP Effect Range

Asks:

> How far apart are the most extreme positive and negative effects?

### Permutation Importance

Asks:

> How much does overall model performance deteriorate when the feature's
> information is destroyed?

Permutation importance is therefore more appropriate for:

> Which feature is generally important to the model across the
> population?

SHAP range is better understood as:

> How widely can this feature's contribution vary?

------------------------------------------------------------------------

# 8. Important Insight About Outliers

Imagine a feature has:

``` text
999 observations:
SHAP ≈ 0
```

and:

``` text
1 observation:
SHAP = +10
```

The range becomes very large.

But the feature may not be important for most predictions.

Therefore:

``` text
Large range
≠
Large overall importance
```

This is a major lesson from the exercise.

------------------------------------------------------------------------

# 9. Question 3 --- Binary Feature Effects

Both:

``` text
diag_1_428
```

and:

``` text
payer_code_?
```

are binary.

Their values are:

``` text
0 or 1
```

The question asks which change has the bigger typical effect:

``` text
diag_1_428: 0 → 1
```

or:

``` text
payer_code_?: 0 → 1
```

The exercise answer is:

``` text
diag_1_428
```

------------------------------------------------------------------------

# 10. Why `diag_1_428` Can Have a Larger Effect

Most `diag_1_428` observations have relatively small SHAP values.

However, the pink points representing:

``` text
diag_1_428 = 1
```

can extend far away from zero.

That means having this diagnosis can substantially increase the model's
predicted readmission risk for some patients.

In contrast, `payer_code_?` has many blue and pink observations, but
changing it from:

``` text
0 → 1
```

typically has a smaller effect.

This demonstrates an important distinction:

``` text
Rare feature with strong effect
```

can behave very differently from:

``` text
Common feature with moderate effect
```

------------------------------------------------------------------------

# 11. Question 4 --- What Does a Blue/Pink Jumble Mean?

The exercise compares:

``` text
number_inpatient
```

and:

``` text
num_lab_procedures
```

For `number_inpatient`, high and low feature values tend to separate
more clearly.

For `num_lab_procedures`, blue and pink dots are mixed together.

What does that mean?

It means:

> The effect of `num_lab_procedures` is not determined simply by whether
> the feature is high or low.

A high value can sometimes:

``` text
increase prediction
```

and sometimes:

``` text
decrease prediction
```

The same is true for low values.

------------------------------------------------------------------------

# 12. Why Does This Jumble Happen?

The most likely explanation is an **interaction effect**.

Conceptually:

``` text
num_lab_procedures
        +
another feature
        ↓
Different prediction effect
```

For example, the meaning of a large number of laboratory procedures may
depend on:

-   diagnosis
-   patient condition
-   other treatment variables
-   other medical features

Therefore, the model may interpret the same value differently for
different patients.

------------------------------------------------------------------------

# 13. Important Lesson

If the SHAP colors are cleanly separated:

``` text
High values → positive SHAP
Low values → negative SHAP
```

there is evidence of a relatively direct relationship.

If the colors are heavily mixed:

``` text
High values → positive and negative SHAP
Low values → positive and negative SHAP
```

then interactions or complex non-linear relationships may be present.

------------------------------------------------------------------------

# 14. Question 5 --- Reading a SHAP Dependence Plot

A SHAP dependence plot has:

``` text
x-axis → actual feature value
y-axis → SHAP contribution
color → another feature
```

The question asks whether there is an interaction between:

``` text
feature_of_interest
```

and:

``` text
other_feature
```

and whether the effect is more positive when the other feature is high
or low.

------------------------------------------------------------------------

# 15. How to Detect Interaction Visually

Look at the colored points.

If the relationship between:

``` text
feature_of_interest
```

and:

``` text
SHAP contribution
```

changes depending on point color, that is evidence of interaction.

For the exercise's example:

-   the pink points represent high values of `other_feature`
-   the blue points represent low values
-   the pink points show a downward trend
-   the blue points are relatively flat or may rise slightly

Therefore, the effect of `feature_of_interest` depends on
`other_feature`.

The exercise's conclusion is:

> `feature_of_interest` has a more positive impact when `other_feature`
> is high.

------------------------------------------------------------------------

# 16. Why Color Is So Useful

Without color, you might see:

``` text
x → SHAP
```

and notice a noisy trend.

Color adds:

``` text
another feature
```

to the same graph.

This helps answer:

> Why do points with similar values of the main feature have different
> SHAP contributions?

The answer may be:

> Because another feature changes the effect.

This is one of the strongest uses of SHAP dependence plots.

------------------------------------------------------------------------

# 17. Question 6 --- Compare `num_lab_procedures` and `num_medications`

The summary plot shows that both features have mixed blue and pink dots.

But their detailed behavior is different.

The exercise asks you to create:

``` python
shap.dependence_plot(
    "num_lab_procedures",
    shap_values[1],
    small_val_X
)

shap.dependence_plot(
    "num_medications",
    shap_values[1],
    small_val_X
)
```

------------------------------------------------------------------------

# 18. `num_lab_procedures` --- Interpretation

The dependence plot for:

``` text
num_lab_procedures
```

looks more like a cloud.

There is no strong, obvious upward or downward slope.

This means:

> It is difficult to identify a simple direct relationship between the
> number of lab procedures and the model's predicted readmission risk.

However, the SHAP values are not all close to zero.

Therefore:

``` text
The feature matters
```

but:

``` text
Its effect is complicated
```

A reasonable next step is to color the dependence plot using different
candidate features to search for an interaction.

------------------------------------------------------------------------

# 19. `num_medications` --- Interpretation

The dependence plot for:

``` text
num_medications
```

shows a much clearer pattern.

The SHAP contribution generally:

``` text
increases
```

until approximately:

``` text
20 medications
```

and then:

``` text
decreases
```

This creates a non-linear relationship.

Conceptually:

``` text
Few medications
      ↓
Increasing positive effect
      ↓
Around 20
      ↓
Effect turns downward
      ↓
More medications
```

This is a much more interpretable pattern than the cloud-like structure
of `num_lab_procedures`.

------------------------------------------------------------------------

# 20. Why the `num_medications` Pattern Is Interesting

The pattern is not necessarily medically obvious.

A machine learning model finding:

``` text
risk contribution rises
```

and then:

``` text
risk contribution falls
```

should lead to investigation.

Possible next steps include:

-   inspect patient groups
-   investigate interactions
-   look for unusual values
-   consult medical experts
-   check whether the model has learned a real pattern
-   check for data quality problems

Explainability therefore helps generate hypotheses rather than
automatically proving them.

------------------------------------------------------------------------

# 21. Summary Plot vs Dependence Plot

A SHAP summary plot gives a broad overview.

It helps answer:

``` text
Which features have important effects?
```

and:

``` text
Do high or low values generally push predictions up or down?
```

A dependence plot focuses on one feature.

It helps answer:

``` text
How does the feature value relate to its SHAP contribution?
```

and:

``` text
Does another feature appear to modify this relationship?
```

------------------------------------------------------------------------

# 22. SHAP vs Permutation Importance

Permutation importance:

``` text
Feature
   ↓
Shuffle it
   ↓
Measure performance loss
```

It gives a compact global importance score.

SHAP:

``` text
Feature value
   ↓
Contribution to prediction
```

It preserves much more detail.

For example, permutation importance might say:

``` text
Feature A = important
```

while SHAP can reveal:

``` text
Feature A
    ↓
Usually small effect
    ↓
Occasionally huge positive effect
```

That difference is extremely valuable.

------------------------------------------------------------------------

# 23. SHAP Summary Plot --- Mental Model

Remember the plot as:

``` text
Vertical position
→ Which feature?

Horizontal position
→ Positive or negative contribution?

Color
→ High or low feature value?

Spread
→ How variable is the feature's effect?
```

This gives you four dimensions of information from one visualization.

------------------------------------------------------------------------

# 24. SHAP Dependence Plot --- Mental Model

Remember:

``` text
X-axis
→ Actual feature value

Y-axis
→ SHAP contribution

Color
→ Another feature

Color-dependent pattern
→ Possible interaction
```

------------------------------------------------------------------------

# 25. Important Code Concepts

### `shap.TreeExplainer`

Used to calculate SHAP values for tree-based models.

``` python
explainer = shap.TreeExplainer(model)
```

### `explainer.shap_values()`

Calculates feature contributions.

### `shap.summary_plot()`

Shows SHAP values across many observations.

``` python
shap.summary_plot(
    shap_values[1],
    small_val_X
)
```

### `shap.dependence_plot()`

Shows the relationship between a feature's value and its SHAP
contribution.

``` python
shap.dependence_plot(
    "num_lab_procedures",
    shap_values[1],
    small_val_X
)
```

------------------------------------------------------------------------

# 26. Common Mistakes

### Mistake 1 --- Treating SHAP range as importance

A few extreme points can create a huge range.

### Mistake 2 --- Looking only at horizontal position

You must also inspect color and spread.

### Mistake 3 --- Assuming high feature value always means positive effect

Interactions can make the same feature value behave differently for
different observations.

### Mistake 4 --- Ignoring feature interactions

A jumble of colors often indicates that another variable changes the
feature's effect.

### Mistake 5 --- Treating SHAP as causal evidence

SHAP explains what the model learned.

It does not prove a real-world causal relationship.

------------------------------------------------------------------------

# 27. Full Course Explainability Workflow

The four exercises build toward a complete workflow:

``` text
Permutation Importance
        ↓
What features does the model rely on?
        ↓
Partial Dependence
        ↓
How do those features affect predictions?
        ↓
SHAP for an individual
        ↓
Why did this individual receive this prediction?
        ↓
Aggregated SHAP
        ↓
How do feature effects vary across the population?
        ↓
SHAP dependence plots
        ↓
What interactions may explain the variation?
```

------------------------------------------------------------------------

# 28. Final Takeaway

The biggest lesson from this final exercise is:

> **A single feature-importance number is often not enough to understand
> a machine learning model.**

SHAP lets you see:

-   direction of effects
-   magnitude of effects
-   variation across observations
-   high vs low feature values
-   unusual cases
-   possible interactions
-   non-linear behavior

That turns explainability from:

``` text
"Which feature matters?"
```

into:

``` text
"How does this feature matter,
for whom,
in which direction,
by how much,
and under what conditions?"
```

That is the core idea behind advanced model explainability.
