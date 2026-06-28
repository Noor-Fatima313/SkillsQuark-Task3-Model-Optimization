import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# Load Dataset
# ==========================================================

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

print("=" * 60)
print("IRIS DATASET")
print("=" * 60)
print("Dataset Shape:", X.shape)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# BASELINE MODELS
# ==========================================================

print("\n" + "=" * 60)
print("BASELINE MODEL PERFORMANCE")
print("=" * 60)

baseline_models = {

    "Logistic Regression":
        LogisticRegression(max_iter=300),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Support Vector Machine":
        SVC()

}

baseline_results = {}

# ==========================================================
# Train Baseline Models
# ==========================================================

for name, model in baseline_models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    baseline_results[name] = accuracy

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            predictions
        )
    )
# ==========================================================
# HYPERPARAMETER TUNING
# ==========================================================

print("\n" + "=" * 60)
print("MODEL OPTIMIZATION USING GRIDSEARCHCV")
print("=" * 60)

optimized_results = {}

# ==========================================================
# 1. Logistic Regression Optimization
# ==========================================================

print("\nOptimizing Logistic Regression...\n")

lr_parameters = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs"]
}

lr_grid = GridSearchCV(

    estimator=LogisticRegression(max_iter=500),

    param_grid=lr_parameters,

    cv=5,

    scoring="accuracy"

)

lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_

lr_predictions = best_lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

optimized_results["Logistic Regression"] = lr_accuracy

print("Best Parameters:")
print(lr_grid.best_params_)

print(f"\nOptimized Accuracy : {lr_accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, lr_predictions))


# ==========================================================
# 2. Decision Tree Optimization
# ==========================================================

print("\n" + "=" * 60)
print("Optimizing Decision Tree...\n")

dt_parameters = {

    "criterion": ["gini", "entropy"],

    "max_depth": [2, 3, 4, 5, None],

    "min_samples_split": [2, 4, 6]

}

dt_grid = GridSearchCV(

    estimator=DecisionTreeClassifier(random_state=42),

    param_grid=dt_parameters,

    cv=5,

    scoring="accuracy"

)

dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_

dt_predictions = best_dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

optimized_results["Decision Tree"] = dt_accuracy

print("Best Parameters:")
print(dt_grid.best_params_)

print(f"\nOptimized Accuracy : {dt_accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, dt_predictions))


# ==========================================================
# 3. Support Vector Machine Optimization
# ==========================================================

print("\n" + "=" * 60)
print("Optimizing Support Vector Machine...\n")

svm_parameters = {

    "C": [0.1, 1, 10],

    "kernel": ["linear", "rbf"],

    "gamma": ["scale"]

}

svm_grid = GridSearchCV(

    estimator=SVC(),

    param_grid=svm_parameters,

    cv=5,

    scoring="accuracy"

)

svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_

svm_predictions = best_svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

optimized_results["Support Vector Machine"] = svm_accuracy

print("Best Parameters:")
print(svm_grid.best_params_)

print(f"\nOptimized Accuracy : {svm_accuracy:.4f}")

print("\nClassification Report")

print(classification_report(y_test, svm_predictions))

# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("BASELINE VS OPTIMIZED MODEL PERFORMANCE")
print("=" * 60)

comparison_df = pd.DataFrame({

    "Baseline Accuracy": baseline_results,

    "Optimized Accuracy": optimized_results

})

comparison_df = comparison_df.round(4)

print("\n")
print(comparison_df)

# ==========================================================
# Best Model
# ==========================================================

best_model = comparison_df["Optimized Accuracy"].idxmax()

best_accuracy = comparison_df["Optimized Accuracy"].max()

print("\n" + "=" * 60)
print("BEST OPTIMIZED MODEL")
print("=" * 60)

print(f"Model    : {best_model}")
print(f"Accuracy : {best_accuracy:.4f}")

# ==========================================================
# Visualization
# ==========================================================

models = comparison_df.index

x = np.arange(len(models))

width = 0.35

plt.figure(figsize=(10,6))

plt.bar(
    x - width/2,
    comparison_df["Baseline Accuracy"],
    width,
    label="Baseline"
)

plt.bar(
    x + width/2,
    comparison_df["Optimized Accuracy"],
    width,
    label="Optimized"
)

plt.xticks(x, models)

plt.ylabel("Accuracy")

plt.xlabel("Machine Learning Models")

plt.title("Baseline vs Optimized Model Performance")

plt.ylim(0.80, 1.02)

plt.legend()

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.show()

# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("""
✓ Loaded the Iris Dataset

✓ Split data into training and testing sets

✓ Applied StandardScaler for feature scaling

✓ Trained baseline machine learning models:
    - Logistic Regression
    - Decision Tree
    - Support Vector Machine

✓ Optimized all models using GridSearchCV

✓ Applied regularization to Logistic Regression

✓ Compared baseline and optimized model performance

✓ Visualized the comparison using a grouped bar chart
""")