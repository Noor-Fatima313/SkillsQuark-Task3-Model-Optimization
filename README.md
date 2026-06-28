# SkillsQuark-Task3-Model-Optimization
# using Scikit-Learn

## Overview

This project focuses on optimizing machine learning models using industry-standard techniques. Building on the supervised learning models from the previous task, the project applies hyperparameter tuning and regularization to improve model performance and compares the optimized models with their baseline versions.

The Iris dataset is used for training, evaluation, and performance comparison.

---

## Objectives

* Load and preprocess the Iris dataset.
* Train baseline machine learning models.
* Optimize models using Hyperparameter Tuning.
* Apply Regularization to Logistic Regression.
* Compare baseline and optimized model performance.
* Visualize the results.

---

## Models Used

* Logistic Regression
* Decision Tree Classifier
* Support Vector Machine (SVM)

---

## Optimization Techniques

### Hyperparameter Tuning

GridSearchCV was used to automatically search for the best hyperparameter combinations for each machine learning model.

### Regularization

Logistic Regression was optimized by tuning the regularization strength (`C`) to improve generalization and reduce the risk of overfitting.

---

## Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Workflow

1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Apply feature scaling using StandardScaler.
4. Train baseline models.
5. Evaluate baseline performance.
6. Optimize each model using GridSearchCV.
7. Compare baseline and optimized results.
8. Visualize the comparison with a grouped bar chart.

---

## Results

The optimized models successfully identified the best hyperparameter combinations using GridSearchCV.

The comparison between baseline and optimized models demonstrates the impact of model optimization. While optimization may not always increase test accuracy, it helps identify robust model configurations through cross-validation.

---

## Project Structure

```
Model_Optimization/
│
├── model_optimization.py
├── requirements.txt
├── README.md
└── comparison_graph.png
```

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python model_optimization.py
```

---

## Learning Outcomes

* Hyperparameter Tuning
* GridSearchCV
* Regularization
* Model Evaluation
* Performance Comparison
* Machine Learning Optimization
* Professional Machine Learning Workflow

```
```
