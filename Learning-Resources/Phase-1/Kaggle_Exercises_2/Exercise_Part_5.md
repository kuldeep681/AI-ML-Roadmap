# Kaggle Intermediate Machine Learning

# Exercise 5 — Data Leakage

---

## 🎯 Exercise Overview

This exercise is about one of the most important practical problems in machine learning:

> **Data leakage**

The exercise focuses mainly on **target leakage** and **train-test contamination**.

Instead of only memorizing definitions, the exercise gives several real-world situations where you have to decide whether information is actually available at prediction time.

The five scenarios are:

1. Nike shoelaces and leather usage
2. Nike shoelaces and leather orders
3. Cryptocurrency prediction
4. Surgeon infection rates
5. Housing prices

The most important lesson is:

> **A feature is safe only if that information would genuinely be available when the prediction is made.**

---

# 1. What Is Data Leakage?

Data leakage happens when information that should not be available to the model gets into the training process.

This can make the model look extremely accurate during development while performing poorly after deployment.

The dangerous part is:

```
Leakage
   ↓
Unrealistically good validation score
   ↓
Confidence in model
   ↓
Deployment
   ↓
Poor real-world performance
```

---

# 2. The Two Main Types

The exercise focuses on:

### 1. Target Leakage

Information related to the target becomes available to the model even though it would not be available at prediction time.

### 2. Train-Test Contamination

Information from validation/test data influences the training or preprocessing process.

---

# 3. The Most Important Question

Whenever you are unsure whether a feature causes leakage, ask:

> **"Would I have this exact information at the moment I need to make the prediction?"**

If the answer is:

```
No
```

then the feature should not be used in that form.

---

# Step 1 — The Data Science of Shoelaces

## 🧩 Scenario

Nike wants to predict:

> How many shoelaces they will need each month.

Available features include:

- Current month
- Previous month's advertising expenditure
- Macroeconomic information
- Amount of leather used during the current month

The model becomes almost perfectly accurate when:

```
leather_used
```

is included.

Why?

Because the amount of leather used tells you how many shoes were produced, which strongly indicates how many shoelaces were required.

---

## ❓ Is `leather_used` Leakage?

### Answer:

**It depends.**

The important question is:

> When is the amount of leather determined?

---

### Case 1 — Determined Before Prediction

Suppose at the beginning of the month Nike already decides:

```
Leather needed = 1000 kg
```

and you make your prediction afterward.

Then:

```
leather_used / planned leather
```

could potentially be available when the prediction is made.

In this situation, it may be acceptable.

---

### Case 2 — Determined During the Month

Suppose Nike only knows the actual amount of leather used after production has happened.

Then you cannot use that value to predict how many shoelaces are needed at the beginning of the month.

You are using future information.

That is:

> **Target leakage**

---

## 🧠 Key Lesson

The feature itself isn't automatically leakage.

The **timing of when the information becomes available** determines whether it is leakage.

---

# Step 2 — Return of the Shoelaces

Now Nike suggests using:

```
Amount of leather ordered
```

instead of:

```
Amount of leather actually used
```

This sounds safer.

But again:

> **It depends on when the leather is ordered.**

---

## Case 1 — Shoelaces Ordered First

Suppose the process is:

```
Order shoelaces
      ↓
Order leather
```

When you predict shoelace requirements, the leather order does not exist yet.

Therefore:

```
leather_ordered
```

would not be available.

Using it would cause leakage.

---

## Case 2 — Leather Ordered First

Suppose:

```
Order leather
      ↓
Predict/order shoelaces
```

Now the amount of leather ordered is already known when the shoelace prediction is made.

Therefore, it can be used legitimately.

---

## 🧠 Key Lesson

Even when a feature seems reasonable, always ask:

```
What happened first?
```

This is especially important for:

- financial data
- medical data
- production systems
- sales forecasting
- time-series problems
- recommendation systems

---

# Step 3 — Getting Rich With Cryptocurrencies?

## 🧩 Scenario

A friend creates a model to predict the cryptocurrency price one day ahead.

The model uses:

- Current price
- Amount sold in the last 24 hours
- Price change in the last 24 hours
- Price change in the last hour
- Number of tweets mentioning the currency in the last 24 hours

The model has an average error of less than $1.

The friend claims this proves the model is highly accurate.

---

# ❓ Is There Data Leakage?

According to the exercise:

> **No leakage is present in these features.**

Why?

Because these features should be available at the time the prediction is made.

For example:

```
Current price
Previous 24-hour activity
Previous 1-hour change
Previous 24-hour tweets
```

are historical/current information.

They are not future information.

---

# ⚠️ But There Is Another Problem

A very low prediction error does not automatically mean the model is useful for investment.

Suppose:

```
Today's price = $100
```

and the model predicts:

```
Tomorrow's price = $100
```

The prediction error could be:

```
$0
```

That sounds perfect.

But the model has not necessarily told you whether the price will:

```
Increase
    or
Decrease
```

---

# 🎯 Better Prediction Target

Instead of predicting:

```
Tomorrow's price
```

it may be more useful to predict:

```
Change in price
```

For example:

```
Tomorrow's change = +$10
```

or:

```
Tomorrow's change = -$5
```

The important question becomes:

> Can the model consistently predict whether the price will rise or fall?

---

## 🧠 Key Lesson

A model can have a low numerical error while still being poor for the actual business decision.

Always ask:

> **Does the prediction target match the decision I actually need to make?**

---

# Step 4 — Preventing Infections

## 🧩 Scenario

A healthcare organization wants to predict whether a patient will develop an infection after surgery.

Each row represents one patient.

The target is:

```
infection = Yes / No
```

Some surgeons may have higher or lower infection rates.

So you create a feature:

```
surgeon_infection_rate
```

calculated from all surgeries performed by that surgeon.

---

# ❓ Is This Safe?

This can create **both**:

1. Target leakage
2. Train-test contamination

---

# 4.1 Target Leakage

Suppose you calculate a surgeon's infection rate using all surgeries, including the current patient.

Imagine:

```
Surgeon A
```

has performed:

```
100 surgeries
```

and:

```
10 infections
```

So:

```
Infection rate = 10%
```

Now you are predicting patient #101.

If patient #101's outcome is already included in the calculated rate, the target has indirectly influenced the feature.

That is leakage.

---

# 4.2 How to Avoid Target Leakage

For each patient, calculate the surgeon's infection rate using only information that existed **before that patient's surgery**.

Conceptually:

```
Previous surgeries
      ↓
Calculate surgeon infection rate
      ↓
Predict current patient
      ↓
Current patient's outcome
      ↓
Do NOT use it to calculate its own feature
```

This respects chronological order.

---

# 4.3 Train-Test Contamination

There is another problem.

Suppose you calculate the surgeon's infection rate using:

```
Training surgeries
+
Test surgeries
```

Now information from the test set has influenced a feature used by the model.

That means the test set is no longer completely independent.

This is:

> **Train-test contamination**

---

# 🧠 Why Is This Dangerous?

The test set exists to answer:

> "How will my model perform on unseen data?"

But if information from the test set is used during feature creation, the test data is no longer truly unseen.

The evaluation becomes overly optimistic.

---

# 4.4 Correct Mental Model

For a prediction:

```
Past information
      ↓
Feature creation
      ↓
Model
      ↓
Prediction
      ↓
Future/current target
```

Never allow:

```
Target
  ↓
Feature
```

or:

```
Test data
  ↓
Training feature calculation
```

---

# Step 5 — Housing Prices

Now the exercise returns to the housing-price problem.

You want to predict the price of a new house.

Potential features:

1. House size
2. Average sales price of homes in the same neighborhood
3. Latitude and longitude
4. Whether the house has a basement

---

# ❓ Which Feature Is Most Likely Leakage?

Answer:

```
Feature 2
```

That is:

> **Average sales price of homes in the same neighborhood**

---

# 5.1 Why Can Neighborhood Average Price Leak?

The key question is:

> How and when is the neighborhood average calculated?

Suppose the average is calculated using historical sales.

That could be perfectly valid.

But suppose the average is updated using the sale price of the very house you are trying to predict.

Then the target has influenced the feature.

That is:

> **Target leakage**

---

# 5.2 Extreme Example

Imagine a neighborhood has only one recent sale.

The house you're trying to predict is that sale.

If the neighborhood average is calculated using that house's actual selling price:

```
Neighborhood average
      =
Actual target price
```

The model has effectively been given the answer.

The validation score could become extremely good.

But when predicting a brand-new house:

```
The house hasn't sold yet.
```

Therefore its price cannot legitimately contribute to the neighborhood average.

---

# 5.3 Why the Other Features Are Safer

### Feature 1 — House Size

The size is normally known when the prediction is made.

So it is generally safe.

---

### Feature 3 — Latitude/Longitude

These are available when the prediction is made.

They don't depend on the future sale price.

So there is no target leakage risk in the described scenario.

---

### Feature 4 — Basement

Whether the house has a basement is known from the house description.

It does not depend on the future sale price.

So it is also safe.

---

# 🧠 Complete Exercise Summary

The five exercises teach the same fundamental principle from different angles.

```
Is the feature available
when prediction is made?
         ↓
      YES
         ↓
   Potentially safe

         OR

      NO
         ↓
      Leakage
```

---

# 6. Target Leakage — Deep Understanding

Target leakage occurs when information that is related to the target becomes available to the model in a way that would not be possible at prediction time.

The key issue is **timing**.

---

## Example

Suppose:

```
Target = Did patient develop infection?
```

Feature:

```
Treatment given after infection
```

The feature may be highly predictive.

But it occurs after the target is determined.

Therefore:

```
Target
   ↓
Treatment
```

Using treatment to predict infection is backwards.

---

# 7. Train-Test Contamination — Deep Understanding

Train-test contamination happens when information from the validation/test data influences:

- preprocessing
- feature engineering
- statistics used to construct features
- model selection

The result is an evaluation that is no longer truly independent.

---

# 8. Target Leakage vs Train-Test Contamination

| Concept                  | Main Problem                                                     |
| ------------------------ | ---------------------------------------------------------------- |
| Target Leakage           | Future/target-related information enters a feature               |
| Train-Test Contamination | Validation/test information influences training or preprocessing |

---

## Easy Way to Remember

### Target Leakage

Ask:

> "Did the target or future information sneak into my feature?"

### Train-Test Contamination

Ask:

> "Did information from my validation/test data sneak into training?"

---

# 9. The Timeline Test

One of the best ways to detect leakage is to draw a timeline.

For example:

```
Past
  ↓
Feature available
  ↓
Prediction time
  ↓
Target happens
  ↓
Future
```

A valid feature should come from:

```
Past → Prediction time
```

A dangerous feature comes from:

```
Prediction time → Future
```

---

# 10. Feature Availability

When designing features, ask:

### Question 1

When is this feature created?

### Question 2

When is this feature updated?

### Question 3

When is the prediction made?

### Question 4

Does the feature use the target?

### Question 5

Does the feature use information from the future?

### Question 6

Does the feature calculation include validation/test observations?

These questions can reveal many leakage problems.

---

# 11. Why Leakage Produces Amazing Scores

Suppose your model normally achieves:

```
Accuracy = 82%
```

Then you accidentally introduce leakage.

Suddenly:

```
Accuracy = 98%
```

It may look like a breakthrough.

But the model may actually be learning information that will not exist in production.

Therefore:

> **Suspiciously good performance should make you investigate your data.**

---

# 12. Real-World Examples of Leakage

## Healthcare

Predict:

```
Disease
```

Using:

```
Treatment given after diagnosis
```

Problem:

```
Treatment occurs after target.
```

---

## Finance

Predict:

```
Loan default
```

Using:

```
Information recorded after the loan default
```

Problem:

```
Future information.
```

---

## E-commerce

Predict:

```
Customer churn
```

Using:

```
Cancellation date
```

Problem:

```
The feature directly reveals the target.
```

---

## Housing

Predict:

```
Sale price
```

Using:

```
Average price calculated using the future sale itself
```

Problem:

```
Target leakage.
```

---

## Manufacturing

Predict:

```
Production requirement
```

Using:

```
Actual production quantity already completed
```

Problem:

```
Information wasn't available when the prediction was needed.
```

---

# ⚠️ Common Mistakes

## Mistake 1 — Looking only at correlation

A feature can have extremely high correlation with the target and still be invalid.

The question isn't:

```
"Does this feature predict the target well?"
```

The question is:

```
"Would this feature exist at prediction time?"
```

---

## Mistake 2 — Assuming historical data is always safe

Historical-looking data can still contain information that was recorded after the target event.

Always understand how the data was collected.

---

## Mistake 3 — Ignoring timestamps

For many real-world problems, timestamps are essential.

You need to know:

```
When feature became available
```

versus:

```
When target became known
```

---

## Mistake 4 — Using the entire dataset for feature engineering

For example:

```
Calculate average using entire dataset
      ↓
Then train/test split
```

This can leak information from the validation/test set.

---

## Mistake 5 — Believing a very high score immediately

An unexpectedly strong score can be a warning sign.

Investigate:

- suspicious features
- timestamps
- preprocessing
- feature engineering
- duplicate information
- target-derived variables

---

# 🔁 When NOT to Use a Feature

Do not use a feature if:

- it is created after the prediction time
- it directly or indirectly contains the target
- it uses future information
- it was calculated using test-set observations
- it was calculated using the target of the same observation
- it depends on information unavailable in production

---

# 🧪 Practical Leakage Checklist

Before training a model, ask:

```
┌─────────────────────────────────┐
│ Is this feature available       │
│ at prediction time?             │
└─────────────────────────────────┘
                ↓
               YES
                ↓
┌─────────────────────────────────┐
│ Does it depend on the target?   │
└─────────────────────────────────┘
                ↓
               NO
                ↓
┌─────────────────────────────────┐
│ Was validation/test information │
│ used to create it?              │
└─────────────────────────────────┘
                ↓
               NO
                ↓
          Safer Feature
```

---

# 🎤 Interview Revision

### What is data leakage?

Data leakage occurs when information that would not legitimately be available when making a prediction enters the model-training process.

### What are the two main types?

```
Target leakage
Train-test contamination
```

### What is target leakage?

When a feature contains information about the target that would not be available at prediction time.

### What is train-test contamination?

When information from validation/test data influences training, preprocessing, or feature engineering.

### How do you detect target leakage?

Think about the timeline and ask whether the feature would actually be available when the prediction is made.

### Why can a very high validation score be suspicious?

Because leakage can make a model appear much more accurate than it really is.

### How can pipelines help?

They help keep preprocessing inside the model-validation workflow so validation information doesn't accidentally influence preprocessing.

### What was the leakage feature in the housing example?

The:

```
Average sales price of homes
in the same neighborhood
```

because its calculation may include information from the target house's sale.

### What was the issue with the surgeon example?

The surgeon's infection rate could cause both:

```
Target leakage
```

and:

```
Train-test contamination
```

depending on how the rate is calculated.

---

# 🧠 Exercise 5 — One-Minute Revision

```
DATA LEAKAGE
      ↓
Information reaches model
that should not be available
      ↓
┌──────────────────────┐
│ Target Leakage       │
│                      │
│ Future/target info   │
│ enters features      │
└──────────────────────┘

┌──────────────────────┐
│ Train-Test            │
│ Contamination         │
│                      │
│ Test/validation info │
│ influences training  │
└──────────────────────┘
```

Remember the golden question:

> **"Would this information actually be available at the exact moment I need to make the prediction?"**

If not:

```
Don't use it.
```

And remember:

> **A suspiciously good model is sometimes a leakage problem, not a brilliant model.**

---

# ⭐ Core Lesson

> **Data leakage can make a machine learning model look extremely accurate while making it useless in production. The safest way to detect leakage is to think carefully about when every feature becomes available, whether it depends on the target, and whether validation/test information influenced its creation.**
