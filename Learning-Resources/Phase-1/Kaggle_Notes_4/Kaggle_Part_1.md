# Kaggle — Explainable Machine Learning

# Part 1 — Understanding Model Insights

## Chapter 1 — What Types of Insights Are Possible?

Machine learning models are often described as **black boxes** because they can make accurate predictions without making their reasoning obvious to humans.

However, a trained model can provide useful insights about what it has learned.

This course focuses on three major types of insights:

1. **Global feature importance**
   - Which features have the biggest impact on model predictions?

2. **Individual prediction explanations**
   - For one specific prediction, how did each feature affect the result?

3. **Global feature effects**
   - Across many predictions, how does changing a feature generally affect the model's predictions?

These insights help us understand the behavior of sophisticated machine learning models rather than treating them as completely opaque systems.

---

# Why Model Insights Are Valuable

Understanding a model is useful for several reasons:

- Debugging
- Feature engineering
- Directing future data collection
- Informing human decision-making
- Building trust

---

## 1. Debugging

Real-world datasets are often:

- unreliable
- disorganized
- incomplete
- noisy
- incorrectly recorded

There can also be mistakes in preprocessing code.

Another major danger is **target leakage**.

A model may appear to perform extremely well while actually learning patterns that do not make sense in the real world.

Model insights can help identify these problems.

### Example

Suppose domain knowledge tells us:

> House size should have some relationship with house price.

But after examining the model, we discover that an unrelated feature is dominating predictions.

That should make us investigate:

- incorrect data
- preprocessing errors
- leakage
- incorrect feature definitions
- accidental correlations

Therefore:

> Understanding what a model has learned is often the first step toward finding bugs.

---

# 2. Informing Feature Engineering

Feature engineering is often one of the most effective ways to improve machine learning models.

The process usually looks like:

    Raw features
        ↓
    Create new features
        ↓
    Train model
        ↓
    Evaluate
        ↓
    Create better features
        ↓
    Repeat

Feature engineering can sometimes be driven by domain knowledge.

But this becomes difficult when a dataset contains:

- hundreds of features
- thousands of features
- poorly documented variables
- anonymous variables

---

## Example: Anonymous Features

Imagine a dataset containing:

    f1
    f2
    f3
    ...
    f527
    f528
    ...

Suppose there is no meaningful description of what these variables represent.

It would be difficult to know what transformations to try.

Model-inspection techniques can reveal that:

- `f527` is important
- `f528` is important
- the two features appear to work together

That could suggest trying transformations such as:

    f527 - f528
    f527 + f528
    f527 / f528

One such transformation could become a very powerful feature.

Therefore:

> Model insights can guide feature engineering instead of relying entirely on intuition.

---

# 3. Directing Future Data Collection

In many businesses and organizations, collecting data is expensive.

For example, an organization may have to decide whether it is worth collecting:

- customer demographic information
- additional transaction information
- behavioral information
- geographic information
- sensor measurements

Model insights can help determine which existing variables are valuable.

This can help answer:

> What additional information would be worth collecting?

If an existing feature is highly influential, a more detailed version of that information might be valuable.

---

# 4. Informing Human Decision-Making

Not every machine learning system makes decisions completely automatically.

Some systems provide predictions that humans use to make decisions.

Examples include:

- loan decisions
- healthcare decisions
- risk assessment
- fraud investigation
- business decisions

In these situations, the explanation behind a prediction can sometimes be more valuable than the prediction itself.

For example:

    Model prediction:
    High risk

    Explanation:
    High debt
    Low income
    Short credit history

A human decision-maker can use this information to understand why the model reached its conclusion.

---

# 5. Building Trust

People should not automatically trust a machine learning model simply because it has a high validation score.

Data errors and modeling mistakes are common.

Before relying on a model for important decisions, people may want to verify that the model is learning sensible patterns.

Model explanations can help.

If the model's behavior agrees with established knowledge, users are more likely to trust it.

For example:

    Expected:
    More goals → higher chance of Man of the Match

    Model insight:
    More goals → higher prediction

This behavior makes intuitive sense.

Therefore:

> Explainability can help people understand and trust machine learning systems.

---

# Chapter 2 — Permutation Importance

## What Is Feature Importance?

One of the most basic questions we can ask about a machine learning model is:

> Which features have the biggest impact on predictions?

This concept is called **feature importance**.

There are several ways to measure feature importance.

Different methods answer slightly different questions and have different limitations.

This course focuses on:

# Permutation Importance

Permutation importance is a method for measuring how much a trained model depends on each feature.

It is useful because it is:

- relatively fast
- widely used
- easy to understand
- model-independent in many practical situations
- based directly on the model's predictive performance

---

# Core Idea

Permutation importance is calculated **after the model has already been trained**.

Suppose we have a model predicting a person's height at age 20.

Available features at age 10 might include:

- height at age 10
- number of socks owned
- other measurements

Clearly, height at age 10 should be useful.

Number of socks owned probably isn't.

Permutation importance tests this by deliberately destroying the relationship between one feature and the target.

---

# How Permutation Importance Works

Suppose the validation data contains:

    Height at Age 10
    Socks Owned
    Other Features
    Target

First, evaluate the trained model normally.

Then:

1. Randomly shuffle one feature.
2. Keep every other feature unchanged.
3. Make predictions using the shuffled data.
4. Compare the new performance with the original performance.
5. Measure how much performance decreased.
6. Restore the original feature.
7. Repeat for the next feature.

The larger the performance decrease, the more important the feature is.

---

# Step-by-Step Mental Model

    Train model
        ↓
    Evaluate normal validation data
        ↓
    Pick one feature
        ↓
    Shuffle that feature
        ↓
    Make predictions again
        ↓
    Measure performance drop
        ↓
    Larger drop = more important feature

The model itself is not retrained during this process.

The model stays fixed.

Only the validation feature values are shuffled.

---

# Why Does Shuffling Work?

Suppose the model relies heavily on:

    Height at Age 10

If we randomly shuffle that column, the height values are no longer matched with the correct observations.

The model loses useful information.

Therefore:

    Large performance drop
            ↓
    Feature was important

Now suppose we shuffle:

    Socks Owned

If the model barely used this information:

    Very small performance drop
            ↓
    Feature was probably not important

---

# Permutation Importance Algorithm

For every feature:

1. Start with the trained model.
2. Calculate the original validation performance.
3. Randomly permute the selected feature.
4. Calculate validation performance again.
5. Calculate the performance deterioration.
6. Repeat the shuffle several times.
7. Average the performance changes.
8. Move to the next feature.

The resulting score represents the feature's permutation importance.

---

# Example Dataset — FIFA 2018

The example uses FIFA 2018 match statistics.

The goal is to predict whether a player from a team received the:

    Man of the Match

award.

The target is converted from:

    Yes
    No

into:

    True
    False

---

# Building the Example Model

The example loads the data and selects numeric features.

The target is created with:

    y = (data['Man of the Match'] == "Yes")

The numeric columns are selected as model features.

The data is split into:

    Training data
    Validation data

A Random Forest classifier is then trained.

The important point is that **permutation importance is calculated after this model has been fitted**.

---

# Calculating Permutation Importance

The example uses the `eli5` library.

The important workflow is:

    trained model
        ↓
    validation data
        ↓
    PermutationImportance
        ↓
    feature importance scores

The permutation importance object is fitted using the validation features and validation target.

The resulting weights can then be displayed with the feature names.

---

# Understanding the Results

Example results include:

    Goal Scored              0.1750 ± 0.0848
    Distance Covered         0.0500 ± 0.0637
    Yellow Card              0.0437 ± 0.0637
    Off-Target               0.0187 ± 0.0500
    Free Kicks               0.0187 ± 0.0637
    ...
    Passes                  -0.0500 ± 0.0637

The features near the top have greater importance.

The features near the bottom have less importance.

---

# What Does the Importance Number Mean?

The first number represents the average reduction in model performance after shuffling that feature.

For example:

    Goal Scored = 0.1750

means that randomly shuffling `Goal Scored` caused the model's performance to decrease substantially.

Therefore, the model relied heavily on that feature.

---

# What Does ± Mean?

Permutation importance contains randomness because the feature is shuffled randomly.

Therefore, the exact performance drop can change between different shuffles.

The:

    ± value

represents variation in the measured importance across repeated shuffles.

For example:

    0.1750 ± 0.0848

means:

    Average importance = 0.1750
    Variation = 0.0848

A larger variation means the estimated importance is less stable.

---

# Negative Permutation Importance

Sometimes permutation importance can be negative.

For example:

    Passes = -0.0500

This means the model performed slightly better after the feature was shuffled.

That does not necessarily mean:

> The feature is harmful.

It can happen because of random chance.

This is especially common when:

- the dataset is small
- the feature is weak
- the measured performance has high variance

A feature with importance close to zero may randomly produce a small negative value.

Therefore:

> Negative permutation importance should be interpreted carefully.

---

# Example Interpretation

The most important feature in the FIFA example was:

    Goal Scored

This makes intuitive sense.

Scoring goals is strongly related to the likelihood that a player receives the Man of the Match award.

This is exactly the kind of insight we want from model interpretation.

---

# Important Mental Model

Permutation importance asks:

> "If I destroy the information contained in this feature, how much worse does my trained model become?"

Therefore:

    Large performance drop
        =
    Model relied heavily on feature

    Small performance drop
        =
    Model relied little on feature

---

# Important Limitations

Permutation importance tells us **how important a feature is**, but it does not directly tell us:

- whether high values increase predictions
- whether low values increase predictions
- whether the relationship is linear
- whether the feature has different effects for different observations

For those questions, other techniques are useful.

In particular:

    Permutation Importance
        ↓
    Which features matter?

    Partial Dependence
        ↓
    How does a feature affect predictions?

    SHAP
        ↓
    How did features affect an individual prediction?

---

# Chapter 1 + 2 Quick Revision

## Model Insights

Three major questions:

    1. Which features matter?
    2. How did features affect one prediction?
    3. How does a feature generally affect predictions?

Uses:

    Debugging
    Feature Engineering
    Data Collection
    Human Decision-Making
    Trust

---

## Permutation Importance

Definition:

> A feature-importance technique that measures how much model performance decreases when the values of a feature are randomly shuffled.

Process:

    Train model
        ↓
    Evaluate model
        ↓
    Shuffle one feature
        ↓
    Evaluate again
        ↓
    Measure performance drop
        ↓
    Repeat for every feature

Remember:

> **Permutation importance measures dependency, not direction.**

It tells us how much a feature matters, but not whether increasing that feature increases or decreases predictions.

---

# Key Terms

| Term                   | Meaning                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Model Interpretability | Understanding how a model behaves and makes predictions                             |
| Feature Importance     | Measuring how much features influence model performance                             |
| Permutation Importance | Measuring importance by shuffling a feature and observing performance deterioration |
| Validation Data        | Data used to evaluate the trained model                                             |
| Performance Drop       | Reduction in model performance after feature permutation                            |
| Baseline Performance   | Model performance before shuffling a feature                                        |
| Negative Importance    | A small performance improvement after shuffling, often caused by randomness         |
| Feature Engineering    | Creating or transforming features to improve model performance                      |
| Target Leakage         | Using information that would not be available when making the prediction            |

---

# Chapter 3 — Partial Dependence Plots

## What Are Partial Dependence Plots?

Permutation importance tells us:

> Which variables have the biggest impact on predictions?

Partial Dependence Plots, or **PDPs**, answer a different question:

> How does a particular feature affect the model's predictions?

This distinction is important.

For example:

    Permutation Importance:
    "Goal Scored is important."

    Partial Dependence:
    "Increasing Goal Scored substantially increases the predicted probability of winning Man of the Match."

---

# Why Use Partial Dependence?

PDPs can answer questions such as:

> Controlling for other house characteristics, how does location affect predicted house prices?

Or:

> How does a particular health-related feature affect predicted health risk while accounting for other features?

PDPs are especially useful for understanding sophisticated models such as:

- Decision Trees
- Random Forests
- Gradient Boosting models

---

# How Partial Dependence Works

The model is first trained normally.

The model is **not retrained** for every feature value.

Instead, we manipulate a feature after the model has been fitted.

Suppose one row represents a football team with:

    Ball Possession = 50%
    Passes = 100
    Attempts = 10
    Goals = 1

We can repeatedly change:

    Ball Possession

while keeping the other features fixed.

For example:

    Ball Possession = 40%
    Prediction

    Ball Possession = 50%
    Prediction

    Ball Possession = 60%
    Prediction

and so on.

This gives us a relationship between:

    Feature Value
        ↓
    Model Prediction

---

# Why Use Multiple Rows?

A single row may have unusual interactions between features.

For example, the effect of possession might depend on:

    Goals Scored
    Attempts
    Passes
    Distance Covered

Therefore, using only one row could give a misleading result.

Instead, the same process is repeated over many rows.

The predictions are then averaged.

This produces the **partial dependence** of the selected feature.

---

# Mental Model

    Select feature
        ↓
    Change its value
        ↓
    Make predictions
        ↓
    Repeat for many observations
        ↓
    Average predictions
        ↓
    Plot feature value vs prediction

The resulting graph shows the model's average relationship between the feature and the prediction.

---

# Example — Decision Tree

The FIFA example uses a Decision Tree Classifier.

The model predicts:

    Probability of winning Man of the Match

The decision tree uses features such as:

    Goal Scored
    On-Target
    Attempts
    Passes
    Corners
    Distance Covered
    Off-Target
    Free Kicks

---

# Reading a Decision Tree

When reading the tree:

- Internal nodes show splitting conditions.
- Leaves represent final prediction regions.
- The values at leaves show the counts of the target classes.

For example:

    Goal Scored <= 0.5

means the tree separates observations according to whether the team scored zero goals or at least one goal.

---

# PDP Example — Goal Scored

The PDP for:

    Goal Scored

shows that scoring a goal substantially increases the predicted probability of winning Man of the Match.

However, additional goals beyond the first may have much less impact according to this particular model.

This is an important point:

> A feature can have a nonlinear effect.

The relationship does not necessarily have to be:

    More feature
        ↓
    Constantly more prediction

---

# PDP Example — Distance Covered

A PDP for:

    Distance Covered (Kms)

using the simple Decision Tree can look very step-like.

This happens because a decision tree creates predictions using discrete splitting rules.

The PDP is therefore reflecting the structure of the tree.

---

# Decision Tree vs Random Forest PDP

A Random Forest model can produce a smoother partial dependence curve.

For example, the Random Forest in the lesson suggests that the predicted probability of winning Man of the Match is higher around:

    100 km

of total distance covered.

Running substantially more than that can reduce the prediction.

The Random Forest curve appears more realistic than the simple step-like Decision Tree curve.

However, the dataset is small, so the result should not be over-interpreted.

---

# 2D Partial Dependence Plots

PDPs can also examine the interaction between **two features**.

Instead of:

    One feature → Prediction

we examine:

    Feature A + Feature B → Prediction

The example uses:

    Goal Scored
    Distance Covered (Kms)

---

# Interpreting the 2D PDP

The plot shows predicted outcomes for combinations of:

    Goals Scored
    Distance Covered

For example:

    Goals >= 1
    +
    Distance around 100 km
    =
    High predicted probability

But when:

    Goals = 0

distance covered has little effect.

This means the effect of distance depends on the number of goals scored.

That is a **feature interaction**.

---

# Important Mental Model

A 1D PDP asks:

> "How does this feature generally affect predictions?"

A 2D PDP asks:

> "How do these two features jointly affect predictions?"

---

# Permutation Importance vs PDP

| Technique              | Main Question                                   |
| ---------------------- | ----------------------------------------------- |
| Permutation Importance | Which features matter most?                     |
| 1D PDP                 | How does one feature affect predictions?        |
| 2D PDP                 | How do two features jointly affect predictions? |

Think of them as complementary techniques.

---

# Chapter 4 — SHAP Values

## What Are SHAP Values?

Permutation importance and partial dependence plots provide **general model insights**.

But sometimes we want to understand:

> Why did the model make this particular prediction?

This is where **SHAP values** are useful.

SHAP stands for:

**SHapley Additive exPlanations**

SHAP values break an individual prediction into contributions from the model's features.

---

# Why Individual Explanations Matter

Individual explanations are useful when a specific prediction needs to be understood.

Examples include:

### Banking

A model rejects a loan application.

The bank may need to explain:

- why the application was rejected
- which factors contributed
- which factors increased risk

### Healthcare

A model predicts high disease risk.

A healthcare provider may want to know:

- which factors increased the risk
- which factors reduced the risk
- which factors could potentially be addressed

---

# SHAP Baseline Idea

SHAP values compare a prediction against a **baseline prediction**.

The basic idea is:

> How much did each feature move the prediction away from the baseline?

Suppose:

    Baseline prediction = 0.50
    Final prediction = 0.70

The features collectively caused:

    +0.20

of movement from the baseline.

SHAP decomposes that change among the features.

---

# SHAP Equation

The key relationship is:

    Sum of SHAP values
    =
    Prediction for the observation
    -
    Baseline prediction

Therefore:

    Baseline
       +
    Feature contributions
       =
    Final prediction

This is one of the most important ideas in SHAP.

---

# Example

Suppose:

    Base Value = 0.4979
    Prediction = 0.70

The difference is approximately:

    0.70 - 0.4979
    = 0.2021

The SHAP values of all features collectively explain this difference.

Some features may push the prediction upward.

Others may push it downward.

---

# Reading a SHAP Force Plot

In the lesson:

- Pink/red contributions increase the prediction.
- Blue contributions decrease the prediction.
- The size of a contribution represents its magnitude.
- The baseline is the starting point.
- The final prediction is the result after adding all contributions.

For example:

    Baseline
       ↓
    Goal Scored → increases prediction
       ↓
    Ball Possession → decreases prediction
       ↓
    Other features
       ↓
    Final prediction = 0.70

---

# SHAP Code

The example uses the `shap` library.

A tree-based model can be explained using:

    shap.TreeExplainer(my_model)

The explainer then calculates SHAP values for the observation.

For classification problems, SHAP can provide separate explanations for different outcomes.

In the example:

    shap_values[0]

represents the negative outcome.

    shap_values[1]

represents the positive outcome.

The lesson focuses on:

    shap_values[1]

because the goal is to explain the probability of winning Man of the Match.

---

# TreeExplainer

For tree-based models, the lesson uses:

    shap.TreeExplainer

This is appropriate for models such as tree ensembles.

The basic process is:

    Trained model
        ↓
    TreeExplainer
        ↓
    SHAP values
        ↓
    Individual prediction explanation

---

# Other SHAP Explainers

The lesson also introduces other explainers.

### TreeExplainer

Designed for tree-based models.

### DeepExplainer

Designed for deep learning models.

### KernelExplainer

Can work with many different model types.

However, KernelExplainer is slower and provides an approximation rather than exact SHAP values.

---

# TreeExplainer vs KernelExplainer

| Explainer       | Typical Use    | Speed / Accuracy                        |
| --------------- | -------------- | --------------------------------------- |
| TreeExplainer   | Tree models    | Efficient for supported tree models     |
| DeepExplainer   | Deep learning  | Designed for neural networks            |
| KernelExplainer | General models | More general but slower and approximate |

The important principle is:

> Choose an explainer appropriate for the model type whenever possible.

---

# Chapter 5 — Aggregating SHAP Values

## Why Aggregate SHAP Values?

Individual SHAP values explain:

> Why did this particular prediction happen?

But we can also calculate SHAP values across many observations.

This lets us answer:

> What has the model learned across the dataset?

This provides powerful model-level insights.

---

# SHAP Summary Plots

Permutation importance provides a simple ranking:

    Feature A
    Feature B
    Feature C
    Feature D

But permutation importance does not tell us **how** each feature affects predictions.

For example, a feature could:

- have a large effect on only a few observations
- have a moderate effect on nearly every observation

Both situations could produce similar overall importance.

SHAP summary plots provide much more information.

---

# How to Read a SHAP Summary Plot

Each dot represents one observation.

Each dot has three important properties.

### 1. Vertical Position

The vertical position tells us which feature the dot represents.

Features near the top are generally more influential.

---

### 2. Color

Color indicates whether the feature value is:

    High
    or
    Low

This helps us understand the direction of the relationship.

---

### 3. Horizontal Position

Horizontal position represents the SHAP value.

A point far to the right means:

    Feature increased prediction

A point far to the left means:

    Feature decreased prediction

---

# Example — Goal Scored

Suppose the summary plot shows:

    High Goal Scored values
        → right side

and:

    Low Goal Scored values
        → left side

This indicates:

    More goals
        ↓
    Higher prediction

while:

    Fewer goals
        ↓
    Lower prediction

This gives us information that permutation importance alone cannot provide.

---

# Example — Yellow Card

The lesson observes that:

- Yellow Card usually has little impact.
- There are some extreme observations where a high value strongly decreases the prediction.

This is important because a simple feature-importance score might hide this behavior.

SHAP allows us to see the distribution of effects.

---

# Example — Ignored Features

The summary plot can reveal features such as:

    Red
    Yellow & Red

having almost no impact.

This suggests the model effectively ignored those variables.

---

# Generating a Summary Plot

The process is:

    Train model
        ↓
    Create TreeExplainer
        ↓
    Calculate SHAP values for validation data
        ↓
    Pass SHAP values to summary_plot

For classification problems, select the appropriate class's SHAP values.

In the example:

    shap_values[1]

is used to explain the positive outcome.

---

# Why SHAP Can Be Computationally Expensive

Calculating SHAP values can take considerable time.

This is especially important with:

- large datasets
- complex models
- many observations

The lesson notes that XGBoost has SHAP optimizations that can make SHAP calculations substantially faster.

Therefore:

> SHAP is powerful, but you should consider computational cost when applying it to large datasets.

---

# SHAP Dependence Contribution Plots

Another SHAP visualization is the:

**SHAP dependence contribution plot**

It is conceptually related to a Partial Dependence Plot, but provides additional information.

---

# PDP vs SHAP Dependence Plot

A PDP mainly shows:

    Feature value
        ↓
    Average prediction effect

A SHAP dependence plot shows:

    Feature value
        ↓
    Actual contribution to prediction

for individual observations.

This lets us see the **distribution of effects**, not just the average effect.

---

# Reading a SHAP Dependence Plot

Each dot represents one observation.

### Horizontal Axis

The actual feature value.

### Vertical Axis

The SHAP value for that observation.

Therefore:

    X-axis
    =
    Feature value

    Y-axis
    =
    Feature contribution

---

# Example — Ball Possession %

Suppose the plot slopes upward.

This suggests:

    More Ball Possession
        ↓
    Higher model prediction

But the points may not all lie exactly on one line.

There may be substantial vertical spread.

That spread tells us:

> The effect of Ball Possession depends on other features.

This indicates feature interactions.

---

# Interaction Example

Consider two observations with approximately the same:

    Ball Possession %

but different SHAP values.

One may have:

    Positive contribution

while another has:

    Negative contribution

This means Ball Possession does not have a fixed effect.

The effect depends on the context of the other features.

---

# Interaction Coloring

SHAP dependence plots can color points according to another feature.

For example:

    X-axis:
    Ball Possession %

    Y-axis:
    SHAP value

    Color:
    Goal Scored

This can reveal interactions between:

    Ball Possession
    and
    Goal Scored

---

# Example Interpretation

The lesson gives an example where points with:

    Goal Scored = 1

behave differently from the general trend.

The general relationship may be:

    More possession
        ↓
    Higher prediction

But for teams scoring only one goal, the relationship can reverse.

This suggests:

> The effect of possession depends on how many goals the team scores.

This is an example of a feature interaction.

---

# Dependence Plot Code Concept

The SHAP dependence plot uses:

    shap.dependence_plot(
        'Ball Possession %',
        shap_values[1],
        X,
        interaction_index="Goal Scored"
    )

The important parameters are:

    Feature being investigated
    ↓
    SHAP values
    ↓
    Dataset
    ↓
    Feature used to reveal interaction

If `interaction_index` is not supplied, SHAP can automatically choose a potentially interesting interaction feature.

---

# Complete Course 4 Mental Model

The techniques build on each other.

    ┌──────────────────────────────┐
    │ Permutation Importance       │
    │ Which features matter?       │
    └──────────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │ Partial Dependence           │
    │ How does a feature affect    │
    │ predictions on average?      │
    └──────────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │ SHAP Values                   │
    │ Why did this particular      │
    │ prediction happen?           │
    └──────────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │ SHAP Summary / Dependence    │
    │ What patterns and           │
    │ interactions exist across   │
    │ many predictions?           │
    └──────────────────────────────┘

---

# The Four Main Questions

## Question 1

> Which features does the model rely on?

Use:

**Permutation Importance**

---

## Question 2

> How does a feature generally affect predictions?

Use:

**Partial Dependence Plot**

---

## Question 3

> Why did the model make this specific prediction?

Use:

**SHAP Values**

---

## Question 4

> How do feature values and interactions affect predictions across many observations?

Use:

**SHAP Summary and Dependence Plots**

---

# Permutation Importance vs PDP vs SHAP

| Technique               | Main Purpose                                                 |
| ----------------------- | ------------------------------------------------------------ |
| Permutation Importance  | Rank feature importance                                      |
| Partial Dependence Plot | Show average effect of a feature                             |
| 2D PDP                  | Show average interaction between two features                |
| SHAP Values             | Explain an individual prediction                             |
| SHAP Summary Plot       | Show feature importance + direction across observations      |
| SHAP Dependence Plot    | Show feature value vs individual contribution + interactions |

---

# Important Distinctions

## Permutation Importance

Tells us:

> **How much does the model depend on this feature?**

It does not directly tell us:

> Whether high values increase or decrease predictions.

---

## Partial Dependence

Tells us:

> **What is the average effect of changing a feature?**

It is useful for understanding the overall relationship.

---

## SHAP

Tells us:

> **How much did each feature contribute to this prediction compared with the baseline?**

It provides both:

- direction
- magnitude

for individual predictions.

Aggregating SHAP values provides broader model-level insights.

---

# Important Practical Warnings

## 1. Model explanations are not automatically causal explanations

If a model shows:

    Feature X
        ↓
    Higher prediction

that does not automatically mean:

    Changing X
        ↓
    Causes the outcome to increase

These techniques explain the model's learned behavior.

They do not automatically establish causality.

---

## 2. Small datasets require caution

The FIFA dataset used in these examples is relatively small.

Therefore:

- importance estimates can be unstable
- PDPs may be sensitive to the fitted model
- SHAP patterns should not automatically be generalized
- random variation can affect results

Model interpretation should always consider the quality and size of the underlying dataset.

---

## 3. Feature interactions matter

A feature may behave differently depending on other features.

For example:

    Ball Possession
        +
    Goals Scored

may produce a different prediction than Ball Possession considered alone.

This is why:

- 2D PDPs
- SHAP summary plots
- SHAP dependence plots

are useful.

---

# Final Revision

## Model Explainability

The purpose is not merely to know:

    Prediction = 0.70

We also want to know:

    Why 0.70?
    Which features mattered?
    Which features increased it?
    Which features decreased it?
    How does the relationship change across observations?

---

## Core Workflow

    Train a model
        ↓
    Validate the model
        ↓
    Inspect feature importance
        ↓
    Study feature effects
        ↓
    Explain individual predictions
        ↓
    Investigate interactions
        ↓
    Use insights for:
        - debugging
        - feature engineering
        - data collection
        - human decisions
        - building trust

---

# Course 4 Key Terms

| Term                    | Meaning                                                                      |
| ----------------------- | ---------------------------------------------------------------------------- |
| Explainability          | Understanding why a model behaves the way it does                            |
| Feature Importance      | Measuring how much features matter to a model                                |
| Permutation Importance  | Measuring importance by shuffling feature values                             |
| Partial Dependence Plot | Showing the average effect of a feature on predictions                       |
| 2D PDP                  | Showing the combined effect of two features                                  |
| SHAP                    | SHapley Additive exPlanations                                                |
| SHAP Value              | Contribution of a feature to an individual prediction relative to a baseline |
| Base Value              | Baseline prediction before individual feature contributions                  |
| SHAP Summary Plot       | Global overview of feature importance and contribution direction             |
| SHAP Dependence Plot    | Relationship between a feature's value and its SHAP contribution             |
| Feature Interaction     | Situation where the effect of one feature depends on another feature         |
| TreeExplainer           | SHAP explainer designed for tree-based models                                |
| DeepExplainer           | SHAP explainer designed for deep learning models                             |
| KernelExplainer         | General-purpose SHAP explainer that is slower and approximate                |

---

# Final Mental Model

Remember the progression:

    "What matters?"
          ↓
    Permutation Importance

    "How does it affect predictions?"
          ↓
    Partial Dependence

    "Why this particular prediction?"
          ↓
    SHAP Values

    "How does the effect vary across observations?"
          ↓
    SHAP Summary / Dependence Plots

The central idea of Course 4 is:

> **Don't treat a machine learning model as a black box. Use model-inspection techniques to understand what it learned, how features influence predictions, and why individual predictions were made.**
