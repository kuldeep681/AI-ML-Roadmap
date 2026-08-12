# Kaggle Feature Engineering — Exercise Notes
## Exercise 2: Mutual Information

This note is based on the Kaggle exercise notebook `exercise-mutual-information.ipynb`.

---

## 1. What This Exercise Is About

The goal of this exercise is to use **Mutual Information (MI)** to identify promising features in the Ames housing dataset and then investigate whether apparently weak features become useful through **feature interactions**.

The exercise follows this workflow:

1. Inspect a few feature/target relationships.
2. Use Mutual Information to rank all features.
3. Inspect the highest and lowest MI scores.
4. Look for patterns in the highly ranked features.
5. Investigate a categorical feature that has low MI.
6. Test whether that feature interacts with other variables.
7. Build a first shortlist of features for later feature engineering.

The important lesson is that feature engineering does not begin by blindly creating columns. First, investigate what information appears useful.

---

## 2. Dataset and Prediction Target

The exercise uses the **Ames housing dataset**.

The target is:

`SalePrice`

The model-development problem is therefore:

`House features → SalePrice`

The notebook contains 78 features, so examining every feature manually would be inefficient. Mutual Information provides an initial screening method.

---

## 3. The Three Features Used for the Warm-Up

The exercise first looks at:

- `YearBuilt`
- `MoSold`
- `ScreenPorch`

These are plotted against `SalePrice`.

The purpose is not to calculate MI manually. The purpose is to build intuition:

> A feature that has a more informative relationship with the target should generally receive a higher MI score.

A visual relationship can help you form a prediction before calculating the actual scores.

---

## 4. Mutual Information in Practice

The notebook uses:

`mutual_info_regression`

because `SalePrice` is a numerical regression target.

The helper function performs several important steps:

- Copies the feature dataframe.
- Converts categorical columns to integer codes using `factorize()`.
- Identifies integer/discrete features.
- Calculates MI scores.
- Stores the results in a pandas Series.
- Sorts the scores from highest to lowest.

The conceptual operation is:

`Feature → Mutual Information → Score`

A higher score means the feature has a stronger statistical dependency with the target.

It does **not** mean the feature will automatically improve a trained model by the same amount.

---

## 5. Why the Notebook Converts Categorical Features

Many Ames features are categorical.

Scikit-learn's MI function needs the feature values represented numerically. The notebook therefore factorizes categorical columns.

Conceptually:

`Category A → 0`

`Category B → 1`

`Category C → 2`

These numbers are identifiers, not meaningful rankings.

The notebook also creates a `discrete_features` list so that the MI calculation knows which features should be treated as discrete.

This distinction matters because discrete and continuous variables are handled differently by the MI estimator.

---

## 6. Computing MI for the Entire Dataset

The exercise separates the target:

`X = df.copy()`

`y = X.pop("SalePrice")`

Then calculates:

`mi_scores = make_mi_scores(X, y)`

The resulting Series contains one MI score per feature.

The notebook then examines:

- The highest-ranked features.
- The lowest-ranked features.
- A horizontal bar plot of the top 20.

This is a feature-screening step.

---

## 7. How to Read the MI Ranking

The ranking helps answer:

> Which individual features appear to contain useful information about `SalePrice`?

The most useful way to interpret the ranking is as a starting point for investigation.

A high score suggests:

- The feature is associated with the target.
- It deserves attention.
- It may be a good candidate for further feature engineering.

A low score suggests:

- The feature has weak individual association with the target.
- It may be less promising by itself.

But a low score does **not** prove that the feature is useless.

A feature can become useful through an interaction with another feature.

---

## 8. Looking for Themes in High-MI Features

The exercise asks whether the highest-scoring features make sense from a real-world perspective.

The notebook points toward three broad themes:

- **Location**
- **Size**
- **Quality**

This is an important feature-engineering habit.

Do not look only at individual column names.

Look for groups of related variables.

For example:

`Size`

may be represented by several area-related measurements.

`Quality`

may be represented by several condition or quality measurements.

`Location`

may be represented by neighborhood or related geographic information.

This gives you ideas for the next stage: combining related features.

---

## 9. Why MI Is Only a Starting Point

MI is a **univariate** feature-utility measure in this exercise.

That means it examines features individually.

It does not directly answer:

> What happens when Feature A and Feature B are used together?

This is why the exercise immediately moves from MI scores to interaction analysis.

A feature can have:

`Low individual MI`

but:

`Strong interaction with another feature`

and therefore still be useful.

---

# Part 2 — Discovering Feature Interactions

## 10. The `BldgType` Feature

The exercise investigates:

`BldgType`

This describes the broad structure of the dwelling.

The categories listed in the notebook are:

- `1Fam` — Single-family Detached
- `2FmCon` — Two-family Conversion
- `Duplx` — Duplex
- `TwnhsE` — Townhouse End Unit
- `TwnhsI` — Townhouse Inside Unit

---

## 11. Why Investigate a Low-MI Feature?

`BldgType` does not receive a particularly high MI score.

A boxen plot is used to compare `SalePrice` across the different building types.

The distributions are fairly similar.

That means:

`BldgType alone`

does not strongly separate the target.

But the exercise asks an important question:

> Could `BldgType` become useful when combined with another feature?

This is the idea of an **interaction effect**.

---

## 12. What Is an Interaction Effect?

An interaction occurs when the effect of one feature depends on the value of another feature.

For example:

`GrLivArea → SalePrice`

may behave differently for different:

`BldgType`

categories.

The relationship is therefore not simply:

`GrLivArea → SalePrice`

It is closer to:

`GrLivArea + BldgType → SalePrice`

where the influence of living area depends on the type of dwelling.

---

## 13. The Two Features Tested for Interaction

The exercise investigates `BldgType` against:

### `GrLivArea`

Above-ground living area.

### `MoSold`

Month in which the house was sold.

The notebook runs the same plotting code twice:

`feature = "GrLivArea"`

and:

`feature = "MoSold"`

The plots use:

- x-axis: the selected numerical feature
- y-axis: `SalePrice`
- hue: `BldgType`
- separate panels for each building type

---

## 14. How to Recognize an Interaction

The notebook gives the key interpretation:

> Trend lines that are significantly different from one category to another indicate an interaction effect.

Think of it visually.

If all categories follow approximately the same relationship:

`Feature ↑ → SalePrice ↑`

then there may be little interaction.

If the relationship changes noticeably between categories, there is evidence of interaction.

For example:

`BldgType A → steep relationship`

`BldgType B → flatter relationship`

This means the effect of the numerical feature depends on the building type.

---

## 15. Why This Matters for Feature Engineering

Suppose `BldgType` has weak individual MI.

You might initially ignore it.

But if:

`BldgType × GrLivArea`

has a meaningful interaction, then `BldgType` can still be valuable.

This demonstrates an important rule:

> Do not discard a feature solely because its individual MI score is low.

Always consider possible interactions.

---

# Part 3 — Building a First Development Feature Set

## 16. Inspecting the Top Features

The notebook displays:

`mi_scores.head(10)`

The goal is to identify the ten most promising individual features.

The exercise emphasizes that these top features tend to reflect:

- Location
- Size
- Quality

These become a starting point for the next exercise.

---

## 17. Combining MI With Domain Knowledge

The exercise does not say:

> Only use the top ten features.

Instead, it recommends using the top features as a starting point and combining them with related variables.

A good workflow is:

`MI ranking`

↓

`Understand what the feature represents`

↓

`Look for related variables`

↓

`Look for interactions`

↓

`Create new features`

↓

`Validate with a model`

This is much better than simply selecting the top numerical scores.

---

## 18. What This Exercise Teaches

The central lesson is:

> Mutual Information helps you find promising individual features, while interaction analysis helps you discover relationships that MI alone cannot reveal.

The full process is:

`New dataset`

↓

`Calculate MI`

↓

`Rank features`

↓

`Inspect meaningful themes`

↓

`Investigate weak-but-plausible features`

↓

`Find interactions`

↓

`Develop new features`

↓

`Validate`

---

## 19. Important Code Concepts

### `X.pop("SalePrice")`

Separates the target from the feature dataframe while removing it from `X`.

### `factorize()`

Converts categorical values into integer codes.

### `mutual_info_regression()`

Calculates MI for a regression target.

### `corrwith()`

Not used as the main method here, but conceptually useful for comparing feature-target relationships.

### `sns.catplot()`

Used to inspect the distribution of `SalePrice` across categories.

### `sns.lmplot()`

Used to visualize regression trends and investigate interactions.

### `df.melt()`

Reshapes several columns into a long format suitable for plotting multiple variables together.

---

## 20. What You Should Be Able to Do After This Exercise

You should be able to:

- Explain what MI is used for.
- Calculate MI scores for a regression dataset.
- Rank features by MI.
- Interpret high and low MI scores correctly.
- Understand why MI is not a model-performance score.
- Recognize that MI is univariate.
- Look for themes among highly ranked features.
- Investigate a low-MI feature for interactions.
- Use category-specific trend lines to identify interactions.
- Create a shortlist of features for later engineering.

---

## 21. Common Mistakes

### Mistake 1 — Treating MI as model accuracy

An MI score is not:

`80% accuracy`

or:

`80% feature importance`.

It measures statistical dependence.

### Mistake 2 — Throwing away every low-MI feature

A low-MI feature may participate in a useful interaction.

### Mistake 3 — Selecting features only by ranking

Domain knowledge and visual investigation still matter.

### Mistake 4 — Ignoring interactions

A feature may become useful when its effect depends on another feature.

---

## 22. Exercise Mental Model

Remember the exercise as:

`Many features`

↓

`Mutual Information`

↓

`Rank`

↓

`Find promising themes`

↓

`Investigate suspiciously weak features`

↓

`Check interactions`

↓

`Create features`

↓

`Validate`

---

## 23. Final Takeaway

Mutual Information is a **screening tool**, not the final decision-maker.

Use it to answer:

> Which individual features appear promising?

Then go further:

> What do these features represent?

> Which features interact?

> What new representation could expose these relationships?

That is the bridge from feature selection to feature engineering.
