# -*- coding: utf-8 -*-
# Mini AKI predictor + SHAP (logistic regression, synthetic-safe)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import shap
from data_generator import make_synthetic

np.random.seed(42)

CSV_PATH = "nephro_small.csv"

# Load or create dataset
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = make_synthetic(2000)
    df.to_csv(CSV_PATH, index=False)

# Features and target
target = "aki_48h"
X = df.drop(columns=[target])
y = df[target].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

num_cols = ["age", "baseline_scr_mgdl", "vanco_trough"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_cols)
    ],
    remainder="drop"
)

clf = LogisticRegression(
    max_iter=200,
    class_weight="balanced",
    solver="liblinear"
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
pipe.fit(X_train, y_train)

# Evaluation
proba = pipe.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
auprc = average_precision_score(y_test, proba)
print(f"AUROC={auc:.3f}  AUPRC={auprc:.3f}  Prevalence={y_test.mean():.3f}")

# Pick a threshold example: maximize Youden J (for demo)
fpr, tpr, thr = roc_curve(y_test, proba)
youden_idx = np.argmax(tpr - fpr)
best_thr = thr[youden_idx]
print(f"Suggested threshold (Youden): {best_thr:.3f}")

# Plot ROC
plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, label=f"AUROC={auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.title("ROC")
plt.legend(); plt.tight_layout(); plt.show()

# SHAP explanations (linear explainer fits logistic)
# Use a small background from training set to speed up
bg = shap.sample(X_train, 200, random_state=42)
explainer = shap.Explainer(pipe.predict_proba, bg)  # model-agnostic, fast enough for small data
shap_values = explainer(X_test, max_evals=500)      # returns Explanation with .values for class probs
# Class 1 explanations (positive class)
sv1 = shap_values[...,1]

# Global summary
shap.summary_plot(sv1.values, X_test, feature_names=X_test.columns, max_display=12, show=True)

# Local waterfall for a single case (e.g., first positive)
idx = int(np.where(y_test.values==1)[0][0]) if (y_test==1).any() else 0
shap.plots.waterfall(sv1[idx], max_display=12)