# Student Performance Prediction Using Machine Learning

## 1. Project Title

Student Performance Prediction and At-Risk Student Identification
Using Machine Learning

---

## 2. Project Overview

This project investigates the use of machine learning techniques to
predict student academic performance and identify students who may be
at risk of poor final academic performance.

The project uses the UCI Student Performance dataset, which contains
information about student demographics, family background, school
characteristics, social relationships, study behaviour, and lifestyle.

Two complementary machine learning tasks are investigated:

1. Regression to predict the continuous final grade (G3).
2. Classification to identify At-Risk and Not At-Risk students.

---

## 3. Problem Statement

Educational institutions may benefit from identifying students who
could require additional academic support.

Traditional analysis may make it difficult to identify complex
relationships between demographic, behavioural, social, family, and
school-related characteristics.

This project therefore investigates whether machine learning models
can identify useful patterns in student data and provide predictions
that may support early academic intervention.

---

## 4. Aim

The main aim of this project is to investigate whether machine
learning can be used to predict student final academic performance and
identify students who may be at risk of poor academic outcomes.

---

## 5. Objectives

The project objectives are:

1. Acquire and understand the UCI Student Performance dataset.

2. Explore the demographic, academic, social, family, behavioural,
   and lifestyle characteristics contained in the dataset.

3. Prepare the dataset for machine learning using appropriate
   preprocessing techniques.

4. Develop regression models for predicting the final grade G3.

5. Develop classification models for identifying At-Risk students.

6. Compare candidate machine learning algorithms using appropriate
   evaluation metrics.

7. Apply hyperparameter tuning to selected models.

8. Evaluate final models using cross-validation and held-out test data.

9. Identify important features associated with model predictions.

10. Discuss the limitations and practical implications of the results.

---

## 6. Research Questions

### Primary Research Question

Can machine learning models predict students' final academic
performance using demographic, family, social, behavioural, and
school-related characteristics?

### Secondary Research Question

Can machine learning models identify students who may be at risk of
poor final academic performance?

### Additional Question

Which student characteristics contribute most strongly to the
classification model's predictions?

---

## 7. Dataset

The project uses the UCI Student Performance dataset.

The dataset contains:

- 649 records
- 30 predictor variables
- G1, G2, and G3 grade variables

The primary target is G3, the final student grade.

G1 and G2 are excluded from the primary early-prediction experiments
because they represent previous-period academic grades.

---

## 8. Machine Learning Tasks

### 8.1 Regression

The regression task predicts G3 as a continuous numerical value.

Candidate models are evaluated using:

- RMSE
- MAE
- R²

The Tuned Random Forest achieved the strongest cross-validation
performance in the completed experiment.

### 8.2 Classification

The classification task converts G3 into two categories:

- At Risk: G3 < 10
- Not At Risk: G3 >= 10

The evaluated classification models include:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

The primary evaluation focuses on Macro F1 and At-Risk Recall in
addition to accuracy.

---

## 9. Expected Contribution

The project provides an experimental comparison of machine learning
approaches for student performance prediction and academic risk
identification.

The analysis also highlights the difference between optimizing
overall model performance and maximizing sensitivity toward students
who may require additional support.

---

## 10. Project Deliverables

The project deliverables include:

- dataset documentation
- data understanding notebook
- preprocessing notebook
- classification notebook
- final evaluation notebook
- source code
- methodology documentation
- final academic report
- presentation materials
- contribution documentation

---

## 11. Project Structure

The project follows this structure:

```text
student-performance-ml/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classification.ipynb
│   └── 04_final_evaluation.ipynb
│
├── presentation/
│
├── report/
│
├── results/
│   ├── figures/
│   └── tables/
│
└── src/

---

## 12. Ethical Considerations

Predictions about student academic performance should not be treated as
definitive judgments about students.

Machine learning predictions may contain errors and may reflect patterns
present in the training dataset.

An At-Risk prediction should therefore be interpreted as a potential
indicator for further review and support rather than as a final
classification of a student's ability or future.

Any real-world educational application should include appropriate human
oversight and consideration of fairness, privacy, and the potential
consequences of automated predictions.

---

## 13. Project Limitations

The dataset contains only 649 students and may not represent all
student populations.

The At-Risk class is relatively difficult to identify reliably.

The primary experiments exclude previous-period grades G1 and G2,
which limits the amount of academic information available to the
models.

The regression results also indicate that the available features
explain only part of the variation in final grades.

Further validation using larger and more diverse datasets would be
required before applying the models in a real educational environment.

---

## 14. Current Experimental Outcome

The completed experiments produced the following main results.

### Regression

The Tuned Random Forest achieved:

- CV RMSE: 2.6325
- Test RMSE: 2.7644
- Test MAE: 2.0116
- Test R²: 0.2164

### Classification

The Tuned Random Forest achieved:

- CV Macro F1: 0.7396
- Test Accuracy: 75.38%
- Test Macro F1: 0.5623

### At-Risk Detection

Logistic Regression achieved the strongest held-out test At-Risk
recall at 45%.

These results indicate that machine learning can identify useful
patterns in student performance data, while also demonstrating the
difficulty of reliably identifying the minority At-Risk group.
