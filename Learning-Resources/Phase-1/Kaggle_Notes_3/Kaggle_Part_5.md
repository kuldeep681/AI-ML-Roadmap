# Kaggle — Feature Engineering

## Part 5 — Chapter 5: Target Encoding

# 1. Introduction

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
