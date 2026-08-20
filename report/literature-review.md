# Literature Review

## 1. Introduction

Educational institutions increasingly collect large amounts of
student-related data that can be analysed to understand and predict
academic performance. Educational Data Mining (EDM) applies data
mining and machine learning techniques to educational data to discover
patterns and support educational decision-making.

Student performance prediction is one of the important applications of
EDM. Previous research has investigated both continuous prediction of
academic grades and classification of students according to performance
levels or risk categories.

This literature review examines research related to student performance
prediction, machine learning in education, regression and
classification approaches, and the identification of students who may
be at risk of poor academic performance.

---

## 2. Educational Data Mining and Student Performance

Educational Data Mining provides methods for extracting useful
information from educational datasets. Student performance prediction
is an important EDM task because predictive models can potentially
help educators identify patterns associated with academic success or
difficulty.

Zhang et al. [6] describe student performance prediction as an
important application of Educational Data Mining. Their work presents
the prediction process as involving data collection, problem
formulation, modelling, prediction and practical application.

The use of machine learning in educational prediction therefore
provides an opportunity to analyse multiple student characteristics
simultaneously rather than relying only on individual indicators.

---

## 3. Student Performance Prediction

One of the foundational studies relevant to this project is the work
of Cortez and Silva [1], which introduced the Student Performance
dataset used in this research. The dataset contains information about
student demographics, family background, school characteristics,
social relationships, study behaviour and lifestyle.

Cortez and Silva [1] investigated student achievement prediction using
the final grade (G3) as the main outcome. Their work also considered
the previous-period grades G1 and G2 as predictive variables.

The UCI documentation associated with the dataset notes that
predicting the final grade without G1 and G2 is a more difficult
problem but can be more useful for an early-prediction scenario [1].

This distinction is particularly relevant to the present project.
The primary experiments in this study exclude G1 and G2 so that the
models attempt to predict final academic performance using demographic,
social, family, behavioural and school-related characteristics.

---

## 4. Factors Associated with Academic Performance

Previous research indicates that student performance can be associated
with a wide range of factors.

Saa, Al-Emran and Shaalan [3] reviewed research on student academic
performance prediction and identified several broad groups of
predictive factors, including previous academic performance,
e-learning activity, demographic characteristics and social
information.

The findings suggest that academic performance is influenced by more
than one type of variable. Student background, behavioural
characteristics and educational factors can provide useful information
for predictive modelling.

This supports the feature selection strategy used in the present
study. Rather than using only academic grades, the primary experiment
uses 30 predictor variables covering:

- Demographic characteristics
- Family background
- School-related factors
- Study behaviour
- Social relationships
- Lifestyle
- Absences

The objective is to investigate how much predictive information can be
obtained without relying on the previous academic grades G1 and G2.

---

## 5. Regression-Based Academic Performance Prediction

Regression methods are appropriate when the target variable represents
a continuous academic outcome.

In this project, the final grade G3 is treated as a continuous
variable for the regression task.

Shahiri and Husain [2] reviewed machine learning techniques used for
student performance prediction and demonstrated the broad use of
predictive modelling approaches in educational data.

Other systematic research has also reported the use of regression and
machine-learning approaches for predicting learning outcomes. A
systematic review by Al-Shabandar et al. [5] examined a large body of
research and reported the use of different predictive approaches for
learning-outcome prediction.

The present study therefore evaluates multiple regression approaches
rather than relying on a single algorithm. The investigated models
include:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

The models are evaluated using cross-validation and held-out test data.
The primary regression metrics are:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R²

This allows both prediction error and explained variance to be
considered when selecting the final model.

---

## 6. Classification and At-Risk Student Identification

A second objective of the project is to identify students who may be
at risk of poor final academic performance.

The classification experiment transforms the continuous G3 grade into
two categories:

- Not At Risk: G3 >= 10
- At Risk: G3 < 10

The purpose of this classification is not to make a definitive
judgement about a student's ability. Instead, the classification is
intended to identify students who may require further attention or
support.

Alyahyan and Düştegör [4] conducted a systematic review of predictive
models for student success and highlighted the potential use of
predictive systems to identify students who may be at risk. Their work
also emphasizes the importance of methodological considerations when
developing predictive student-success models.

More recent research has continued to investigate machine learning for
early identification of at-risk students. Ewais et al. [7] examined
machine-learning approaches for identifying students at risk at an
early stage, demonstrating the continued relevance of this research
area.

The present project evaluates:

- Logistic Regression
- Decision Tree Classification
- Random Forest Classification

Because the At-Risk group represents a minority class in this dataset,
accuracy alone is not sufficient for evaluating the classification
models. Macro F1 and At-Risk recall are therefore also considered.

---

## 7. Model Evaluation and Validation

Reliable evaluation is essential when applying machine learning to
educational data.

A model that performs well on training data may fail to generalize to
unseen students. Therefore, the present study uses both
cross-validation and an independent held-out test set.

The methodology also compares alternative algorithms and applies
hyperparameter tuning to improve model performance.

For regression, RMSE, MAE and R² are used.

For classification, accuracy, Macro F1 and At-Risk recall are used.

This evaluation strategy reflects the need to consider multiple
aspects of predictive performance rather than selecting a model based
on a single metric.

Research reviews in student-performance prediction have also
identified issues related to generalization, model selection and
overfitting as important considerations [4], [5].

---

## 8. Research Gap

Although previous studies demonstrate that machine learning can be
used to predict student academic performance, several challenges
remain.

First, many predictive studies make use of previous academic
performance variables. While these variables can provide strong
predictive information, they may not be available when early
intervention decisions are required.

Second, student performance is influenced by multiple demographic,
social, behavioural and educational factors. Therefore, a useful
prediction system should investigate the contribution of these
different categories of information.

Third, identifying at-risk students presents a class-imbalance and
evaluation challenge. A model may achieve reasonable overall accuracy
while still failing to identify a substantial proportion of students
who belong to the minority At-Risk group.

Finally, predictive models used in education must be interpreted
carefully. A prediction should not be treated as a definitive
judgement about a student's ability or future.

---

## 9. Relevance to the Current Study

The current project addresses these issues through two complementary
machine learning tasks.

### Regression

The regression task predicts the continuous final grade G3 using
student demographic, family, social, behavioural and school-related
features.

The primary experiment excludes G1 and G2 in order to investigate an
early-prediction setting.

### Classification

The classification task identifies students as either At Risk or Not
At Risk based on their final grade.

The classification experiment places particular importance on the
ability to identify the minority At-Risk group.

### Comparative Modelling

Both tasks compare multiple machine learning methods and use
cross-validation, hyperparameter tuning and held-out test evaluation.

This provides a structured comparison between alternative approaches
and supports selection of the most appropriate model for each task.

---

## 10. Summary of the Literature

The reviewed literature demonstrates that machine learning and
Educational Data Mining provide useful approaches for analysing and
predicting student academic performance.

Previous research has established the importance of student
characteristics, academic information, behavioural factors and
educational context in predictive modelling [1]–[6].

The literature also demonstrates the relevance of identifying students
who may be at risk of poor academic outcomes [4], [7].

However, predictive performance alone is not sufficient for a
responsible educational application. Model validation, class
imbalance, generalization, fairness, privacy and human interpretation
must also be considered.

The present project builds on this research by combining regression
and classification tasks while deliberately excluding G1 and G2 from
the primary prediction experiments. This provides an opportunity to
evaluate whether demographic, social, family, behavioural and
school-related characteristics can provide useful early indicators of
final academic performance.

---

# References

[1] P. Cortez and A. M. G. Silva, “Using data mining to predict
secondary school student performance,” in *Proc. 5th FUture Business
Technology Conf. (FUBUTEC 2008)*, Porto, Portugal, 2008, pp. 5–12.

[2] A. M. Shahiri and W. Husain, “A review on predicting student's
performance using data mining techniques,” *Procedia Computer
Science*, vol. 72, pp. 414–422, 2015.

[3] A. A. Saa, M. Al-Emran, and K. Shaalan, “Factors affecting
students' performance in higher education: A systematic review of
predictive data mining techniques,” *Technology, Knowledge and
Learning*, vol. 24, pp. 567–598, 2019.

[4] E. Alyahyan and D. Düştegör, “Predicting academic success in higher
education: Literature review and best practices,” *International
Journal of Educational Technology in Higher Education*, vol. 17,
2020.

[5] M. Al-Shabandar, A. J. Hussain, A. Liatsis, and R. Keight,
“Predicting student performance using machine learning techniques: A
systematic literature review,” *Applied Sciences*, vol. 11, no. 1,
2021, Art. no. 237.

[6] Y. Zhang, W. Yun, Y. An, X. Cui, and Y. Zhang, “A review of
student performance prediction based on machine learning,” in
*Proc. 2021 International Conference on Intelligent Computing,
Automation and Applications (ICAA)*, 2021.

[7] A. Ewais *et al.*, “Machine learning for early identification of
at-risk students,” *Scientific Reports*, 2026.
