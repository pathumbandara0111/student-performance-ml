# Methodology

## 1. Research Approach

This project uses a supervised machine learning approach to investigate
student academic performance using the UCI Student Performance dataset.

Two prediction tasks are investigated:

1. Regression — predicting the continuous final grade (G3).
2. Classification — identifying students who may be at risk of poor
   final academic performance.

The experiments are implemented using Python and scikit-learn.

---

## 2. Dataset

The project uses the UCI Student Performance dataset.

The dataset contains:

- 649 student records
- 30 predictor variables
- G1, G2, and G3 academic grade variables

The 30 predictor variables describe demographic, family, school,
social, behavioural, and lifestyle characteristics.

The primary academic outcome is G3, representing the student's final
grade.

G1 and G2 are excluded from the primary early-prediction experiments
because they represent previous academic-period grades and are strongly
related to the final grade.

---

## 3. Data Acquisition

The dataset was obtained programmatically from the UCI Machine Learning
Repository using the `ucimlrepo` Python package.

The raw data is stored locally in:

- `data/raw/student_performance_features.csv`
- `data/raw/student_performance_targets.csv`

The feature and target data are maintained separately to provide a
clear distinction between predictor variables and academic outcomes.

---

## 4. Data Understanding

The first analysis stage examines the structure and quality of the
dataset.

The data understanding process includes:

- loading the feature and target datasets
- inspecting dataset dimensions
- examining feature names and data types
- checking for missing values
- examining numerical variables
- examining categorical variables
- reviewing the distribution of the final grade
- examining the distribution of the classification target

The results of this stage are documented in:

`notebooks/01_data_understanding.ipynb`

---

## 5. Data Preprocessing

The preprocessing stage prepares the raw student data for machine
learning.

The preprocessing workflow includes:

- separating predictors from target variables
- identifying numerical and categorical variables
- handling numerical variables
- handling categorical variables
- encoding categorical variables
- scaling numerical variables where required
- preparing the data for machine learning pipelines

Preprocessing operations are implemented using scikit-learn
transformers and pipelines to reduce the risk of data leakage.

The preprocessing analysis is documented in:

`notebooks/02_data_preprocessing.ipynb`

---

## 6. Regression Task

The regression experiment predicts the continuous final grade G3.

The regression task evaluates multiple machine learning approaches and
compares their performance using cross-validation and held-out test
data.

The primary evaluation metric is Root Mean Squared Error (RMSE).

Additional evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

Lower RMSE and MAE values indicate better predictive performance,
while a higher R² indicates that the model explains a greater
proportion of the variation in the target variable.

The regression experiment identified the Tuned Random Forest as the
preferred model based on cross-validation performance.

---

## 7. Classification Task

The classification experiment converts the final grade into an
At-Risk classification target.

The target is defined as:

- G3 < 10 → At Risk
- G3 >= 10 → Not At Risk

The classification task evaluates:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

Because the classification target contains an At-Risk minority class,
overall accuracy alone is not sufficient for evaluating the models.

The primary classification evaluation measures include:

- Macro F1-score
- At-Risk Recall
- Accuracy
- Test Macro F1-score
- At-Risk F1-score

The classification experiment is documented in:

`notebooks/03_classification.ipynb`

---

## 8. Cross-Validation

Cross-validation is used to obtain a more reliable estimate of model
performance during model comparison.

The classification and regression experiments use cross-validation
rather than relying exclusively on a single training/test split.

Cross-validation results are used to compare candidate models and
support model selection.

---

## 9. Hyperparameter Tuning

Hyperparameter optimization is applied to the Random Forest models.

For classification, RandomizedSearchCV is used to investigate
different Random Forest configurations.

The selected classification configuration includes:

- n_estimators = 100
- max_depth = 20
- min_samples_split = 5
- min_samples_leaf = 1
- max_features = log2

The selected regression configuration includes:

- n_estimators = 300
- max_depth = 10
- min_samples_split = 2
- min_samples_leaf = 4
- max_features = 0.5

The tuned models are evaluated using the same evaluation methodology
as the other candidate models.

---

## 10. Final Evaluation

The final evaluation stage consolidates the regression and
classification experiments.

The final evaluation includes:

- model comparison
- cross-validation performance
- held-out test performance
- regression error analysis
- classification performance
- At-Risk detection performance
- feature importance analysis
- research findings
- project limitations
- final conclusions

The final evaluation is documented in:

`notebooks/04_final_evaluation.ipynb`

---

## 11. Feature Importance

Feature importance is examined using the selected Tuned Random Forest
classification model.

The feature importance values are extracted from the fitted Random
Forest model after preprocessing and categorical encoding.

The transformed feature names are obtained from the preprocessing
pipeline so that the importance values correspond to the actual
features used by the trained model.

The feature importance analysis is used to identify variables that
contribute strongly to the model's classification decisions.

Feature importance is interpreted as model-specific importance rather
than as evidence of causal relationships.

---

## 12. Model Selection

Model selection is based on the evaluation objectives of each task.

For regression, the Tuned Random Forest is selected because it achieved
the strongest cross-validation RMSE performance.

For overall classification performance, the Tuned Random Forest is
selected because it achieved the highest cross-validation Macro
F1-score.

However, Logistic Regression demonstrated stronger sensitivity toward
the At-Risk class on the held-out test set.

Therefore, the project distinguishes between:

- overall classification performance
- At-Risk detection performance

This distinction is important because a model with strong overall
performance may not identify the largest possible proportion of
At-Risk students.

---

## 13. Final Experimental Results

### Regression

The selected Tuned Random Forest achieved:

- Cross-validation RMSE: 2.6325
- Test RMSE: 2.7644
- Test MAE: 2.0116
- Test R²: 0.2164

### Classification

The selected Tuned Random Forest achieved:

- Cross-validation Macro F1: 0.7396
- Test Accuracy: 75.38%
- Test Macro F1: 0.5623
- Test At-Risk Recall: 30%

### At-Risk Detection

Logistic Regression achieved:

- Test At-Risk Recall: 45%

This demonstrates the trade-off between overall classification
performance and sensitivity toward the minority At-Risk class.

---

## 14. Reproducibility

The project is organized into separate directories for:

- raw data
- processed data
- notebooks
- source code
- documentation
- results
- report materials
- presentation materials

The Python environment is managed using a virtual environment and
dependencies are recorded in `requirements.txt`.

The main experimental notebooks are:

1. `01_data_understanding.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_classification.ipynb`
4. `04_final_evaluation.ipynb`

This structure allows the analysis to be reproduced and reviewed in a
logical sequence.

---

## 15. Limitations

The methodology has several limitations.

The dataset contains only 649 student records, which limits the amount
of information available for model training.

The At-Risk class is a minority class, making reliable identification
more difficult.

The primary prediction experiments exclude G1 and G2 in order to
represent an early-prediction scenario.

The regression model explains only a limited proportion of the
variation in final grades.

The dataset also does not contain every factor that may influence
student academic performance.

Therefore, the resulting models should be considered decision-support
tools rather than definitive assessments of individual students.
