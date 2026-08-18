# Dataset Documentation

## Dataset Name

Student Performance

## Source

UCI Machine Learning Repository

## UCI Dataset ID

320

## Dataset URL

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

## Dataset Description

The Student Performance dataset contains information related to
students' academic performance, demographic characteristics,
social relationships, family background, school-related factors,
study behaviour, and lifestyle.

## Dataset Size

- Records: 649
- Predictor features: 30
- Grade variables: 3
- Total variables available: 33

## Predictor Variables

The dataset contains the following 30 predictor variables:

- school
- sex
- age
- address
- famsize
- Pstatus
- Medu
- Fedu
- Mjob
- Fjob
- reason
- guardian
- traveltime
- studytime
- failures
- schoolsup
- famsup
- paid
- activities
- nursery
- higher
- internet
- romantic
- famrel
- freetime
- goout
- Dalc
- Walc
- health
- absences

## Target Variables

The dataset provides three academic grade variables:

- G1 - first-period grade
- G2 - second-period grade
- G3 - final grade

## Primary Research Target

The primary target for this project is:

**G3 - Final Grade**

G1 and G2 will be excluded from the primary early-prediction
experiment because they represent previous-period academic grades
and are strongly related to the final grade.

## Research Design

The primary experiment will investigate whether student demographic,
social, family, behavioural, and school-related characteristics can
predict final academic performance without relying on G1 and G2.

A secondary experiment may include G1 and G2 to compare the effect of
previous academic grades on model performance.

## Data Acquisition

The dataset was obtained programmatically from the UCI Machine
Learning Repository using the `ucimlrepo` Python package.

## Files

The downloaded raw dataset is stored locally in:

- `data/raw/student_performance_features.csv`
- `data/raw/student_performance_targets.csv`

## Citation

Cortez, P. (2008). Student Performance. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5TG7T
