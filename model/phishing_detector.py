#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py – Multi-model training & evaluation for phishing detection
-------------------------------------------------------------------------
Trains and compares multiple classifiers using lexical + structural features
extracted by fastFeatures.py.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting

Output:
- Model performance report (Precision, Recall, F1, AUC)
- Saved model files in `outputs/` directory
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------
# Load dataset (make sure you have your extracted features CSV)
# ---------------------------------------------------------------------
DATA_PATH = "features_dataset.csv"  # generated from fastFeatures
df = pd.read_csv(DATA_PATH)

# Handle NaN
df = df.fillna(0)

# Assuming 'label' column: 1 = phishing, 0 = benign
X = df.drop(columns=["label"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Define models
# ---------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=200, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        colsample_bytree=0.9, subsample=0.8, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42
    )
}

# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------
report = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    cr = classification_report(y_test, preds, output_dict=True)
    f1 = cr["weighted avg"]["f1-score"]
    prec = cr["weighted avg"]["precision"]
    rec = cr["weighted avg"]["recall"]

    print(f"✅ {name}: F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | AUC={auc:.4f}")
    print(confusion_matrix(y_test, preds))

    report[name] = {
        "F1": f1, "Precision": prec, "Recall": rec, "AUC": auc
    }

    if f1 > best_score:
        best_score = f1
        best_model = (name, model)

# ---------------------------------------------------------------------
# Save best model + scaler
# ---------------------------------------------------------------------
best_name, best_clf = best_model
joblib.dump(best_clf, f"outputs/{best_name}.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

with open("outputs/model_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n🏆 Best Model: {best_name} | F1 = {best_score:.4f}")
print(f"📁 Models and reports saved in /outputs/")
