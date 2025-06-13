# -----------------------------------------------------------------------------
# Import: Standard libraries
# -----------------------------------------------------------------------------

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, cast

# -----------------------------------------------------------------------------
# Import: External libraries
# -----------------------------------------------------------------------------

import joblib # Saving/loading trained models to a file
import numpy as np # Manipulations with n-dimensional arrays (de-facto standard)
import pandas as pd # Tabular data – DataFrame
import seaborn as sns # Statistical visualizations (based on matplotlib)
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Import: Scikit-learn - Machine learning libraries
# -----------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris  
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
)
from sklearn.model_selection import (
    GridSearchCV,  # Exhaustive hyperparameter search using cross-validation
    StratifiedKFold,  # Cross validation that preserves class distribution
    cross_validate,  # Evaluates multiple metrics efficiently
    train_test_split,  # Splits data into training and test sets
)
from sklearn.neighbors import KNeighborsClassifier  # k-nearest neighbors classifier
from sklearn.pipeline import Pipeline  # Chain of preprocessing steps followed by a model
from sklearn.preprocessing import StandardScaler  # Standardizes features 
from sklearn.svm import SVC  # Support Vector Classifier 
from sklearn.utils import Bunch  # Simple helper class returned by load_iris

# -----------------------------------------------------------------------------
# 1. Config and constants
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    cv_folds: int = 10
    test_size: float = 0.20  # 80/20 split
    output_dir: pathlib.Path = pathlib.Path("outputs")


CONFIG = Config()
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Data loading + basic EDA
# -----------------------------------------------------------------------------

iris = cast(Bunch, load_iris(as_frame=True))
X_full: pd.DataFrame = iris.data 
y_full: pd.Series = iris.target
feature_names: List[str] = iris.feature_names
target_names: List[str] = iris.target_names

# 2.1 Train‑test split 
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=CONFIG.test_size,
    stratify=y_full,
    random_state=CONFIG.random_state,
)

# 2.2 Basic EDA (on full dataset)
def basic_eda(df: pd.DataFrame, labels: pd.Series) -> None:
    sns.set_theme(style="ticks", context="notebook")

    # Pairplot - distributions + scatter‑matrix
    pairplot_path = CONFIG.output_dir / "pairplot.png"
    if not pairplot_path.exists():
        sns.pairplot(pd.concat([df, labels.rename("target")], axis=1), hue="target", corner=True)
        plt.suptitle("Iris Pair Plot by Species", y=1.02)
        plt.savefig(pairplot_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Correlation matrix
    heat_path = CONFIG.output_dir / "correlation_heatmap.png"
    if not heat_path.exists():
        corr = df.corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature Correlation Matrix")
        plt.savefig(heat_path, dpi=300, bbox_inches="tight")
        plt.close()


basic_eda(X_full, y_full)

# -----------------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------------

numeric_features: List[str] = feature_names

# StandardScaler for numeric features
preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_features)],
    remainder="drop",
)

# -----------------------------------------------------------------------------
# 4. Model Zoo
# -----------------------------------------------------------------------------

models: Dict[str, Pipeline] = {
    "LogReg": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=500, random_state=CONFIG.random_state)),
        ]
    ),
    "kNN": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", KNeighborsClassifier()),
        ]
    ),
    "SVM": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", SVC(kernel="rbf", probability=True, random_state=CONFIG.random_state)),
        ]
    ),
    "RandomForest": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=CONFIG.random_state)),
        ]
    ),
    "GBDT": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", GradientBoostingClassifier(random_state=CONFIG.random_state)),
        ]
    ),
}

# -----------------------------------------------------------------------------
# 5. Cross‑validation of models
# -----------------------------------------------------------------------------

def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scoring = ["accuracy", "f1_macro"]
    cv = StratifiedKFold(
        n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_state
    )   
    records = []
    for name, pipe in models.items():
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        records.append(
            {
                "model": name,
                "accuracy_mean": np.mean(cv_results["test_accuracy"]),
                "f1_macro_mean": np.mean(cv_results["test_f1_macro"]),
                "fit_time": np.mean(cv_results["fit_time"]),
            }
        )
    df_results = pd.DataFrame(records).sort_values("accuracy_mean", ascending=False)
    
    df_results.to_csv(CONFIG.output_dir / "model_cv_results.csv", index=False)
    return df_results


cv_results_df = evaluate_models(X_train, y_train)
print("\n----- 10‑fold CV Summary -----\n", cv_results_df, sep="")

# -----------------------------------------------------------------------------
# 6. Hyper‑parameter tuning (top‑2)
# -----------------------------------------------------------------------------

best_two = cv_results_df.head(2)["model"].tolist()

param_grids: Dict[str, Dict[str, List]] = {
    "SVM": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    },
    "kNN": {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],  # 1=Manhattan, 2=Euclidean
    },
}

grid_results = []

for name in best_two:
    print("\n----- Hyperparameter tuning for", name, "-----")
    grid = GridSearchCV(
        models[name],
        param_grids[name],
        cv=StratifiedKFold(n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_state),
        scoring="f1_macro",
        n_jobs=-1, # run in parallel
        verbose=1, # show progress
    )

    grid.fit(X_train, y_train)

    print(f"Best params for {name}: {grid.best_params_}; best acc={grid.best_score_:.3f}")

    # Save the model and result for future use
    joblib.dump(grid.best_estimator_, CONFIG.output_dir / f"best_{name}.joblib")
    grid_results.append(
        {
            "model": name,
            "best_accuracy": grid.best_score_,
            "best_params": grid.best_params_,
        }
    )

# Save grid search results to CSV
pd.DataFrame(grid_results).to_csv(
    CONFIG.output_dir / "gridsearch_summary.csv", index=False
)

# -----------------------------------------------------------------------------
# Choose overall best model based on accuracy
# -----------------------------------------------------------------------------

best_model_name = max(grid_results, key=lambda d: d["best_accuracy"])["model"]
best_model: Pipeline = joblib.load(CONFIG.output_dir / f"best_{best_model_name}.joblib")

# -----------------------------------------------------------------------------
# 7. Final training on X_train
# -----------------------------------------------------------------------------

best_model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate on test set
# -----------------------------

y_test_pred = best_model.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)
print(f"\nTest‑set accuracy: {acc_test:.3f}\n")

# Classification report
report_path = CONFIG.output_dir / "classification_report_test.txt"
with report_path.open("w") as f:
    f.write(str(classification_report(y_test, y_test_pred, target_names=target_names)))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(4, 4))
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=target_names
).plot(ax=ax, cmap="Blues", colorbar=False)

plt.title(f"{best_model_name} Confusion Matrix (Test)")
plt.savefig(CONFIG.output_dir / "confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------
# 9. CLI entry point
# -----------------------------

def main():
    print("Pipeline run completed. Outputs saved to", CONFIG.output_dir.resolve())


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Example of loading a saved model and checking its parameters
# -----------------------------------------------------------------------------

# loaded = joblib.load("outputs/best_SVM.joblib")
# y_test_pred = best_model.predict(X_test)
# acc_test = accuracy_score(y_test, y_test_pred)
# print(f"Loaded model accuracy on test set: {acc_test:.3f}")