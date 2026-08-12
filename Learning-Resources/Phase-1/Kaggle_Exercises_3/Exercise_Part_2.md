# Kaggle Feature Engineering — Exercise Notes
## Exercise 3: Creating Features

This note is based on the Kaggle exercise notebook `exercise-creating-features.ipynb`.

---

## 1. What This Exercise Is About

This exercise takes the promising features discovered in the Mutual Information and interaction analysis exercise and turns them into **actual engineered features**.

The notebook demonstrates five practical techniques:

1. Mathematical combinations
2. Numerical × categorical interaction features
3. Count features
4. Breaking down a categorical feature
5. Grouped transformations

The important theme is:

> Feature engineering should be guided by what the data means in the real world.

The notebook uses the Ames housing dataset and predicts:

`SalePrice`

The model used for scoring is:

`XGBRegressor`

The exercise uses 5-fold cross-validation and RMSLE as the evaluation metric.

---

# Part 1 — Mathematical Feature Transformations

## 2. Why Combine Features Mathematically?

Many useful relationships are not represented directly by one column.

For example:

`GrLivArea`

tells us living area.

`LotArea`

tells us lot area.

But:

`GrLivArea / LotArea`

tells us how much of the lot is occupied by above-ground living area.

That ratio can represent something more meaningful than either raw feature alone.

---

## 3. Why the Exercise Focuses on Ratios and Sums

The notebook specifically notes that the features being combined describe areas and therefore have compatible units.

This makes mathematical combinations easier to interpret.

Because the model is XGBoost, the exercise focuses on useful transformations such as:

- Ratios
- Sums

rather than relying heavily on polynomial transformations.

---

## 4. Feature 1 — `LivLotRatio`

The exercise creates:

`LivLotRatio = GrLivArea / LotArea`

Conceptually:

`LotArea`

↓

How much land is available?

`GrLivArea`

↓

How much above-ground living space exists?

Then:

`GrLivArea / LotArea`

↓

Living-area density relative to the lot.

The ratio may contain information that is harder for the model to obtain directly from the two raw values.

---

## 5. Feature 2 — `Spaciousness`

The exercise creates:

`Spaciousness = (FirstFlrSF + SecondFlrSF) / TotRmsAbvGrd`

First:

`FirstFlrSF + SecondFlrSF`

gives total first- and second-floor area.

Then this is divided by:

`TotRmsAbvGrd`

which represents the total number of rooms above ground.

Conceptually:

`Total above-ground floor area / Number of above-ground rooms`

This gives a rough measure of average space per room.

The important idea is not the exact name. It is the reasoning:

> Combine related measurements to construct a more meaningful quantity.

---

## 6. Feature 3 — `TotalOutsideSF`

The notebook creates:

`TotalOutsideSF`

by adding:

- `WoodDeckSF`
- `OpenPorchSF`
- `EnclosedPorch`
- `Threeseasonporch`
- `ScreenPorch`

Conceptually:

`Outdoor area 1`

+

`Outdoor area 2`

+

`Outdoor area 3`

+

...

↓

`TotalOutsideSF`

Instead of asking the model to separately learn the contribution of every outdoor-area variable, we explicitly provide a combined measure.

---

## 7. The Code Pattern

A new dataframe is created:

`X_1 = pd.DataFrame()`

Then each engineered feature is added as a new column.

The important pandas operations are:

- `/` for ratios
- `+` for sums
- Column selection through `df.column_name`

The resulting feature dataframe can later be joined to the original feature set.

---

# Part 2 — Interaction With a Categorical Feature

## 8. The Interaction Discovered Earlier

The previous exercise found an interaction between:

`BldgType`

and:

`GrLivArea`

The effect of living area on price differs according to the building type.

Now the exercise explicitly represents that interaction.

---

## 9. Why One-Hot Encoding Is Needed

`BldgType` is categorical.

A model cannot directly multiply a category such as:

`1Fam`

by a numerical value such as:

`2000`

in a meaningful way.

So the categorical feature is first one-hot encoded.

For example, conceptually:

`BldgType`

↓

`Bldg_1Fam`

`Bldg_2FmCon`

`Bldg_Duplx`

`Bldg_TwnhsE`

`Bldg_TwnhsI`

Each row has a 1 for its category and 0 for the others.

---

## 10. Creating the Interaction Features

The notebook uses:

`pd.get_dummies(df.BldgType, prefix="Bldg")`

This creates one binary column per building type.

Then:

`X_2 = X_2.mul(df.GrLivArea, axis=0)`

multiplies every category indicator by the corresponding row's living area.

The result is effectively:

`Bldg_1Fam × GrLivArea`

`Bldg_2FmCon × GrLivArea`

`Bldg_Duplx × GrLivArea`

etc.

---

## 11. Why This Works

Suppose a house is:

`1Fam`

with:

`GrLivArea = 2000`

The interaction features become conceptually:

`Bldg_1Fam × GrLivArea = 2000`

while the other building-type interactions become:

`0`

For a duplex with 2000 square feet:

`Bldg_Duplx × GrLivArea = 2000`

and the other interaction columns are 0.

This lets the model learn different relationships between living area and price for different building types.

---

## 12. General Pattern: Numerical × Categorical Interaction

This technique can be generalized.

Suppose:

`Category`

interacts with:

`NumericFeature`

The workflow is:

`Categorical feature`

↓

`One-hot encode`

↓

`Multiply each dummy column by numerical feature`

↓

`Interaction features`

This is especially useful when visualization suggests different slopes or relationships across categories.

---

# Part 3 — Count Features

## 13. What Is a Count Feature?

A count feature summarizes how many conditions are present.

The exercise asks for:

`PorchTypes`

It counts how many different outdoor porch/deck areas are greater than zero.

The source columns are:

- `WoodDeckSF`
- `OpenPorchSF`
- `EnclosedPorch`
- `Threeseasonporch`
- `ScreenPorch`

---

## 14. Why Count Presence Instead of Total Area?

`TotalOutsideSF` answers:

> How much outdoor area does the house have?

`PorchTypes` answers:

> How many different kinds of outdoor areas does the house have?

These are different pieces of information.

For example:

House A:

`WoodDeckSF > 0`

`OpenPorchSF > 0`

Others = 0

Then:

`PorchTypes = 2`

House B:

Only a large deck exists.

Then:

`PorchTypes = 1`

The two houses could have similar total area but different numbers of outdoor-area types.

---

## 15. The Pandas Technique

The notebook uses:

`df[[columns]].gt(0.0).sum(axis=1)`

Break this into steps.

### Step 1 — Select the columns

Choose the five porch/deck columns.

### Step 2 — `.gt(0.0)`

Check whether each value is greater than zero.

This converts the values conceptually into:

`True / False`

### Step 3 — `.sum(axis=1)`

Because boolean values behave like:

`True = 1`

`False = 0`

the row-wise sum counts how many conditions are true.

Therefore:

`Number of positive porch/deck areas`

becomes:

`PorchTypes`

---

# Part 4 — Break Down a Categorical Feature

## 16. The `MSSubClass` Feature

The exercise examines:

`MSSubClass`

The feature describes the type of dwelling.

The categories contain a more general category followed by a more specific description.

The exercise asks us to keep only the first part.

---

## 17. Why Break a Categorical Feature Down?

A detailed category may contain information at one level.

But the broader category may have a different predictive relationship.

For example, a value could conceptually look like:

`TypeA_specific_description`

The exercise wants:

`TypeA`

This creates a simpler categorical feature that captures the broader grouping.

---

## 18. The String Operation

The notebook uses:

`df.MSSubClass.str.split("_", n=1, expand=True)[0]`

Important pieces:

### `.str`

Allows pandas string operations on the column.

### `.split("_")`

Splits the text whenever `_` occurs.

### `n=1`

Only split once.

This is important because the exercise wants the first component, not every component.

### `expand=True`

Returns the split results in separate columns.

### `[0]`

Selects the first part.

The resulting feature is:

`MSClass`

---

## 19. Why `n=1` Matters

Without limiting the split, a string containing multiple underscores could be separated into many pieces.

The exercise specifically asks for the first word/category.

Therefore:

`split("_", n=1)`

is the appropriate operation.

---

# Part 5 — Grouped Transform

## 20. The Neighborhood Idea

The exercise gives a useful domain insight:

> A home's value depends partly on how it compares with typical homes in its neighborhood.

A raw feature such as:

`GrLivArea`

tells us the house's own living area.

But it does not directly tell us:

> Is this house unusually large or small compared with other houses in the same neighborhood?

That is where the grouped feature comes in.

---

## 21. Creating `MedNhbdArea`

The notebook creates:

`MedNhbdArea`

using:

`df.groupby("Neighborhood")["GrLivArea"].transform("median")`

This calculates the median `GrLivArea` for each neighborhood and assigns that neighborhood's median back to every row belonging to that neighborhood.

---

## 22. Why `transform()` Instead of `agg()`?

This distinction is important.

A normal group aggregation produces one value per group.

For example:

`Neighborhood A → median = 1500`

`Neighborhood B → median = 1800`

But we need a feature with one value for **every house**.

`transform()` returns the grouped statistic aligned with the original rows.

So the dataset effectively becomes:

`House 1 → Neighborhood A → 1500`

`House 2 → Neighborhood A → 1500`

`House 3 → Neighborhood B → 1800`

and so on.

---

## 23. Why Use the Median?

The median represents a typical value while being less sensitive to extreme observations than the mean.

For housing data, this can be useful because neighborhoods may contain unusually large or unusually small homes.

The feature therefore represents:

`Typical living area in this home's neighborhood`

---

# Part 6 — Combining the Feature Sets

## 24. Joining the New Features

The notebook creates five feature groups:

`X_1`

Mathematical transforms.

`X_2`

Categorical interaction features.

`X_3`

Count feature.

`X_4`

Categorical breakdown.

`X_5`

Grouped transformation.

They are combined with:

`X_new = X.join([X_1, X_2, X_3, X_4, X_5])`

This creates one feature set containing:

`Original features + engineered features`

---

## 25. Evaluating the Feature Set

The notebook then calls:

`score_dataset(X_new, y)`

The scoring helper:

- Encodes remaining categorical columns using factorization.
- Uses `XGBRegressor`.
- Performs 5-fold cross-validation.
- Uses negative mean squared log error internally.
- Converts the result into RMSLE.

The important idea is:

> Feature engineering should ultimately be judged by validation performance, not by whether a feature sounds clever.

---

# Part 7 — Five Feature-Engineering Patterns to Remember

## Pattern 1 — Mathematical Combination

`Feature A + Feature B`

or:

`Feature A / Feature B`

Use when the combination has a meaningful interpretation.

Example:

`GrLivArea / LotArea`

---

## Pattern 2 — Categorical Interaction

`One-hot(Category) × NumericFeature`

Use when the numerical relationship differs across categories.

Example:

`BldgType × GrLivArea`

---

## Pattern 3 — Count Feature

`Count(condition_1, condition_2, ...)`

Use when the number of present components contains useful information.

Example:

`PorchTypes`

---

## Pattern 4 — Categorical Decomposition

`Detailed category → broader category`

Use when the original category contains hierarchical information.

Example:

`MSSubClass → MSClass`

---

## Pattern 5 — Grouped Transform

`Group → calculate statistic → map statistic back to every row`

Use when an observation is meaningful relative to its group.

Example:

`Neighborhood → median GrLivArea`

---

# 26. Important Pandas Operations

### `pd.DataFrame()`

Creates a new dataframe for engineered features.

### `pd.get_dummies()`

One-hot encodes categorical variables.

### `.mul(..., axis=0)`

Performs row-aligned multiplication.

### `.gt(0.0)`

Checks whether values are greater than zero.

### `.sum(axis=1)`

Calculates row-wise sums.

### `.str.split()`

Splits string values.

### `.groupby()`

Creates groups according to a categorical variable.

### `.transform()`

Calculates a grouped statistic while returning a result aligned to the original rows.

### `.join()`

Combines engineered features with the original feature set.

---

# 27. Common Mistakes

### Mistake 1 — Combining unrelated units

Mathematical combinations should have a meaningful interpretation.

### Mistake 2 — Forgetting alignment

When multiplying a dummy dataframe by a Series, row alignment matters.

The notebook therefore uses:

`axis=0`

### Mistake 3 — Counting values instead of presence

`PorchTypes` counts how many porch/deck categories are present, not their total area.

### Mistake 4 — Using `aggregate()` when a row-aligned feature is needed

For this task, `transform()` is appropriate because every original row needs the group's median.

### Mistake 5 — Creating features without validation

A feature can make intuitive sense and still fail to improve the model.

---

# 28. Exercise Mental Model

Remember the entire exercise as:

`Understand the data`

↓

`Find useful relationships`

↓

`Combine related variables`

↓

`Represent interactions`

↓

`Count useful conditions`

↓

`Extract broader categories`

↓

`Compare observations with their groups`

↓

`Join features`

↓

`Validate`

---

# 29. What You Should Be Able to Do

After completing this exercise, you should be able to:

- Create ratios.
- Create sums.
- Create count features.
- One-hot encode a categorical variable.
- Build numerical × categorical interactions.
- Split categorical strings.
- Create grouped statistics.
- Use `groupby().transform()`.
- Combine multiple feature sets.
- Evaluate engineered features with cross-validation.

---

# 30. Final Takeaway

The key lesson is not the five particular features.

The bigger idea is:

> Good feature engineering changes raw columns into representations that express useful relationships more directly.

The five patterns from this exercise give you a practical toolkit:

`Combine`

`Interact`

`Count`

`Decompose`

`Group`

Then:

`Validate`

That final step is what tells you whether the engineering actually helped.
