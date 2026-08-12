# Kaggle --- Machine Learning Explainability

## Exercise 3 --- SHAP Values

## 1. Exercise Goal

This exercise moves from **model-level explanations** to **individual
prediction explanations**.

The scenario is hospital readmission prediction.

The hospital wants to identify patients who may be at high risk of being
readmitted.

The model is not supposed to replace doctors.

Instead:

``` text
Machine Learning Model
        ↓
Highlight important risk factors
        ↓
Doctors review the patient
        ↓
Doctors make the final decision
```

The exercise combines:

-   permutation importance
-   partial dependence plots
-   SHAP values

to build an explanation workflow for medical professionals.

------------------------------------------------------------------------

# 2. The Dataset

The target is:

``` text
readmitted
```

The dataset contains patient-related features such as:

-   `time_in_hospital`
-   `num_lab_procedures`
-   `num_procedures`
-   `num_medications`
-   `number_outpatient`
-   `number_emergency`
-   `number_inpatient`
-   `number_diagnoses`
-   demographic indicators
-   payer indicators
-   medical specialty indicators
-   diagnosis indicators
-   medication indicators

Some feature names require special interpretation.

For example:

``` text
diag_1_428
```

means the first diagnosis corresponds to diagnosis code `428`.

A feature such as:

``` text
glimepiride_No
```

indicates the encoded medication status.

Features beginning with:

``` text
medical_specialty_
```

describe the specialty of the doctor.

------------------------------------------------------------------------

# 3. Choosing the Right Explainability Technique

The exercise gives three major tools.

### Permutation Importance

Use it to ask:

> Which features does the model rely on most?

### Partial Dependence Plot

Use it to ask:

> How does changing a feature affect the model's average prediction?

### SHAP Values

Use them to ask:

> Why did this particular patient receive this particular prediction?

This progression is important:

``` text
Global importance
        ↓
General feature effect
        ↓
Individual prediction explanation
```

------------------------------------------------------------------------

# 4. Step 1 --- Give Doctors a Model Overview

The first task asks you to create one or two graphics or tables that
give doctors a quick overview of what the model has learned.

A strong approach is to use:

``` text
Permutation Importance
```

and:

``` text
Partial Dependence Plot
```

or another compact combination of model-level explanations.

The goal is not to overwhelm the doctors.

The goal is to show:

-   which variables matter
-   whether the model behaves in a medically plausible way
-   whether there are obvious data or modeling problems

------------------------------------------------------------------------

# 5. Why This Step Matters

A model can produce accurate predictions while learning relationships
that are suspicious.

For example:

``` text
Model predicts accurately
        ↓
But uses an unexpected variable
        ↓
Investigate
        ↓
Possible data problem / leakage / hidden relationship
```

Explainability therefore helps with model debugging and trust.

------------------------------------------------------------------------

# 6. Step 2 --- Investigate `number_inpatient`

The exercise identifies:

``` text
number_inpatient
```

as an important feature.

The doctors want to know:

> How does this variable affect predicted readmission risk?

The appropriate tool is a PDP.

The basic code is:

``` python
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

feature_name = "number_inpatient"

PartialDependenceDisplay.from_estimator(
    my_model,
    val_X,
    [feature_name]
)

plt.show()
```

------------------------------------------------------------------------

# 7. Interpreting the PDP

The PDP shows how the model's predicted probability changes as:

``` text
number_inpatient
```

changes.

The exercise observes that increasing the number of inpatient procedures
leads to increased predicted readmission risk.

This is intuitively plausible.

A patient with more previous inpatient events may represent a more
complex or higher-risk medical history.

But remember:

> This is the model's learned relationship, not proof of causation.

------------------------------------------------------------------------

# 8. Step 3 --- Compare With `time_in_hospital`

The doctors want another feature for comparison:

``` text
time_in_hospital
```

Create another PDP.

The purpose is not merely to make another graph.

The real question is:

> Is the effect of `time_in_hospital` large or small compared with
> `number_inpatient`?

This is why PDPs are useful.

Two features can both matter, but their prediction effects can have very
different magnitudes.

------------------------------------------------------------------------

# 9. Step 4 --- Compare Model Behavior With Raw Data

The PDP for:

``` text
time_in_hospital
```

shows only a small change in predicted readmission risk, around 5
percentage points across its range.

The doctors are surprised.

The exercise therefore asks you to examine the raw data directly.

The code is:

``` python
all_train = pd.concat(
    [train_X, train_y],
    axis=1
)

all_train.groupby(
    ["time_in_hospital"]
).mean().readmitted.plot()

plt.show()
```

This plots the observed average readmission rate for each hospital-stay
duration.

------------------------------------------------------------------------

# 10. Why Compare Raw Rates With PDP?

This is an important modeling lesson.

The raw relationship:

``` text
time_in_hospital
        ↓
Observed readmission rate
```

does not necessarily equal:

``` text
time_in_hospital
        ↓
Model prediction
```

Why?

Because the model considers many features simultaneously.

For example:

``` text
Raw relationship:
time_in_hospital → readmission
```

could look strong or weak because other patient characteristics are
mixed together.

The model's PDP attempts to isolate the model's learned contribution of
that feature while averaging over the other variables.

Therefore, differences between raw data and PDP are not automatically
evidence of an error.

------------------------------------------------------------------------

# 11. Important Concept --- Correlation vs Model Effect

Suppose:

``` text
Patients with longer stays
        ↓
Often have more serious conditions
```

Then raw readmission rates may reflect both:

``` text
length of stay
+
severity of illness
```

The model may already capture severity through other variables.

Therefore, after accounting for those variables, the additional
contribution of:

``` text
time_in_hospital
```

may be relatively small.

This is one reason model explanations can differ from simple group
averages.

------------------------------------------------------------------------

# 12. Step 5 --- Build `patient_risk_factors`

This is the most important task in the exercise.

The hospital wants a function:

``` python
patient_risk_factors(model, patient)
```

that accepts one patient's data and produces an explanation of:

-   which features increased readmission risk
-   which features decreased readmission risk
-   how strongly each feature affected the prediction

This is exactly where SHAP values are useful.

------------------------------------------------------------------------

# 13. SHAP Values for an Individual

SHAP stands for:

``` text
SHapley Additive exPlanations
```

For an individual prediction, SHAP decomposes the prediction into:

``` text
Baseline prediction
+
Feature contribution 1
+
Feature contribution 2
+
Feature contribution 3
+
...
=
Final prediction
```

Conceptually:

``` text
Base value
    ↓
Feature pushes prediction up
    ↓
Another feature pushes prediction down
    ↓
Another feature pushes prediction up
    ↓
Final patient risk
```

------------------------------------------------------------------------

# 14. SHAP Code Pattern

The exercise provides the following basic approach:

``` python
import shap

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    data_for_prediction
)
```

For a tree-based model such as Random Forest:

``` python
shap.TreeExplainer()
```

is an appropriate explainer.

------------------------------------------------------------------------

# 15. Force Plot

A SHAP force plot can visualize the individual prediction.

Conceptually:

``` text
Base prediction
       |
       +---- feature increases risk
       |
       +---- feature decreases risk
       |
       +---- feature increases risk
       |
       ↓
Final prediction
```

The important information is the direction and magnitude of each
contribution.

------------------------------------------------------------------------

# 16. Positive and Negative SHAP Values

A positive SHAP value means:

``` text
Feature pushes prediction upward
```

For this exercise:

``` text
Positive SHAP
→ increases predicted readmission risk
```

A negative SHAP value means:

``` text
Feature pushes prediction downward
```

For this exercise:

``` text
Negative SHAP
→ decreases predicted readmission risk
```

The magnitude tells you how large the contribution is.

------------------------------------------------------------------------

# 17. Why This Is Better for an Individual Patient

Suppose two patients both receive:

``` text
30% readmission risk
```

Their explanations may be completely different.

Patient A:

``` text
Previous inpatient visits → strong increase
Other factors → small effects
```

Patient B:

``` text
Different diagnosis → increase
Medication-related features → decrease
```

The prediction alone does not reveal this.

SHAP allows us to see the individual reasoning behind the prediction.

------------------------------------------------------------------------

# 18. Selecting Only Important Features

The exercise says you do not need to show every tiny feature
contribution.

A useful implementation can:

1.  calculate SHAP values
2.  pair each feature with its contribution
3.  sort by absolute SHAP magnitude
4.  keep the largest contributors
5.  visualize them

The important sorting idea is:

``` python
abs(shap_value)
```

because we care about both:

``` text
large positive effect
```

and:

``` text
large negative effect
```

------------------------------------------------------------------------

# 19. A Useful Mental Model

Think of the whole exercise as:

``` text
Hospital data
      ↓
Train model
      ↓
Permutation Importance
      ↓
What matters globally?
      ↓
Partial Dependence
      ↓
How does an important feature affect predictions?
      ↓
SHAP
      ↓
Why did this particular patient get this prediction?
```

------------------------------------------------------------------------

# 20. Important Safety and Interpretation Principle

The model is assisting doctors.

It is not replacing them.

The scenario explicitly says:

``` text
Doctors make the final decision.
```

The model provides evidence and highlights factors for consideration.

This is particularly important in high-stakes domains such as
healthcare.

------------------------------------------------------------------------

# 21. Common Mistakes

### Mistake 1 --- Treating SHAP as causality

A SHAP value explains the model's prediction.

It does not prove that changing the feature will cause the real-world
outcome to change.

### Mistake 2 --- Showing every feature

Too many tiny contributions can make the explanation difficult to
understand.

### Mistake 3 --- Ignoring the baseline

SHAP contributions explain the difference between:

``` text
baseline prediction
```

and:

``` text
individual prediction
```

### Mistake 4 --- Confusing global and local explanations

Permutation importance is mainly global.

SHAP can be used locally for an individual prediction and can also be
aggregated globally.

------------------------------------------------------------------------

# 22. Final Takeaway

This exercise teaches how explainability techniques can be combined into
a practical workflow.

``` text
Permutation Importance
→ What matters?

PDP
→ How does it affect predictions?

SHAP
→ Why did this individual receive this prediction?
```

The final product is not simply a model.

It is:

``` text
Model
+
Evidence
+
Interpretation
+
Human review
```

That is the core idea behind interpretable machine learning.
