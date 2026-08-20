# Individual Contribution Matrix

## 1. Purpose

This document defines the responsibilities and individual contributions
of the five group members for the IT41033 - NIA Mini Project.

The project follows the required workflow:

1. Identifying the Problem
2. Data Understanding
3. Data Collection
4. Data Preprocessing
5. Data Mining and Machine Learning
6. Evaluation and Interpretation
7. Documentation and Reporting

All group members are expected to contribute to the project and to the
final report and presentation.

---

## 2. Contribution Overview

| Member | Primary Responsibility | Main Project Area |
|---|---|---|
| Member 01 | Problem definition, data collection and data understanding | Problem Definition / Data Understanding |
| Member 02 | Data preprocessing and feature engineering | Data Preprocessing |
| Member 03 | Regression modelling and analysis | Data Mining and Machine Learning |
| Member 04 | Classification modelling and analysis | Data Mining and Machine Learning |
| Member 05 | Evaluation, interpretation, literature review and final documentation | Evaluation / Documentation |

---

## 3. Member 01

### Primary Responsibilities

- Define the research problem.
- Define the research objectives and questions.
- Identify and obtain the UCI Student Performance dataset.
- Document the dataset source and variables.
- Perform initial data understanding.
- Explore dataset structure and distributions.
- Investigate data quality issues.
- Prepare exploratory visualizations.

### Related Project Files

- `notebooks/01_data_understanding.ipynb`
- `src/data_loader.py`
- `docs/dataset.md`
- `docs/project-proposal.md`

### Contribution Status

The work associated with this area has been implemented in the
repository and should be reviewed and confirmed as the actual
individual contribution by Member 01.

---

## 4. Member 02

### Primary Responsibilities

- Perform data cleaning and preprocessing.
- Investigate missing values and inconsistencies.
- Prepare numerical features.
- Encode categorical features.
- Apply feature scaling where appropriate.
- Prepare the machine learning preprocessing pipeline.
- Perform feature engineering.
- Prepare the classification target.

### Related Project Files

- `notebooks/02_data_preprocessing.ipynb`
- `src/data_preprocessing.py`
- `src/feature_engineering.py`

### Contribution Status

The preprocessing and feature-engineering implementation is present in
the repository. The final individual contribution should be confirmed
with Member 02 before submission.

---

## 5. Member 03

### Primary Responsibilities

- Define the regression modelling task.
- Train candidate regression models.
- Compare alternative regression methods.
- Perform cross-validation.
- Perform hyperparameter tuning.
- Evaluate regression models using appropriate metrics.
- Select the final regression model.
- Interpret regression results.

### Related Project Files

- `notebooks/02_data_preprocessing.ipynb`
- `src/models.py`
- `src/evaluation.py`
- `notebooks/04_final_evaluation.ipynb`

### Main Regression Evaluation

The final experimental results include:

- Cross-validation RMSE: 2.6325
- Test RMSE: 2.7644
- Test MAE: 2.0116
- Test R²: 0.2164

### Contribution Status

The regression implementation and results are present in the
repository. The final individual contribution should be confirmed
with Member 03 before submission.

---

## 6. Member 04

### Primary Responsibilities

- Define the classification task.
- Define the At-Risk classification target.
- Train candidate classification models.
- Compare alternative classification methods.
- Perform cross-validation.
- Perform hyperparameter tuning.
- Evaluate classification performance.
- Analyse At-Risk detection performance.
- Interpret classification results.

### Related Project Files

- `notebooks/03_classification.ipynb`
- `src/feature_engineering.py`
- `src/models.py`
- `src/evaluation.py`
- `notebooks/04_final_evaluation.ipynb`

### Main Classification Evaluation

The final experimental results include:

- CV Macro F1: 0.7396
- Test Accuracy: 75.38%
- Test Macro F1: 0.5623

Logistic Regression achieved the strongest held-out At-Risk recall
at 45%.

### Contribution Status

The classification implementation and results are present in the
repository. The final individual contribution should be confirmed
with Member 04 before submission.

---

## 7. Member 05

### Primary Responsibilities

- Perform final model evaluation and comparison.
- Interpret the final experimental findings.
- Analyse feature importance.
- Document limitations and ethical considerations.
- Conduct the literature review.
- Integrate research findings into the final report.
- Prepare the discussion and conclusion sections.
- Coordinate final report structure.
- Assist with final presentation preparation.
- Review IEEE references.

### Related Project Files

- `notebooks/04_final_evaluation.ipynb`
- `src/evaluation.py`
- `docs/methodology.md`
- `report/`
- `presentation/`

### Contribution Status

The final evaluation implementation is already present in the
repository. The remaining literature review, final report and
presentation work should be completed and recorded as part of the
actual contribution of Member 05.

---

## 8. Shared Group Responsibilities

All members should participate in the final stages of the project.

Shared responsibilities include:

- Reviewing the final results.
- Checking the correctness of the methodology.
- Reviewing the final report.
- Reviewing tables and figures.
- Reviewing the discussion and conclusion.
- Checking references.
- Reviewing the presentation.
- Final proofreading.
- Final submission preparation.

---

## 9. Final Contribution Verification

The contribution allocation above represents the project work
structure.

Before final submission, each member's actual contribution should be
reviewed and confirmed.

The final contribution statement should reflect the work that each
member actually completed rather than assigning work solely based on
the planned project structure.

---

## 10. Repository Evidence

Git commits, notebooks, source files, documentation and final
deliverables can be used as supporting evidence of project development.

The repository is maintained centrally during development, with the
current project repository hosted under the group's Member 01 GitHub
repository.

Individual contribution records should therefore be considered
together with the contribution matrix and the group's agreed
responsibilities.
