#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model_balanced.py – Multi-model training & evaluation for phishing detection
----------------------------------------------------------------------------------
Handles class imbalance via SMOTE (and fallback oversampling).
Trains and compares multiple classifiers on lexical + content features.

Outputs:
- Balanced training
- Model performance report (Precision, Recall, F1, AUC)
- Saved model + scaler in /outputs/
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
DATA_PATH = "features_extended.csv"  # or your merged dataset
df = pd.read_csv(DATA_PATH).fillna(0)

if "label" not in df.columns:
    raise ValueError("❌ Dataset must have a 'label' column (1=phishing, 0=benign).")

X = df.drop(columns=["label"])
y = df["label"]

print(f"📊 Original class distribution:\n{y.value_counts().to_dict()}")

# ---------------------------------------------------------------------
# Split data
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# Handle class imbalance with SMOTE or fallback oversampling
# ---------------------------------------------------------------------
print("⚙️ Applying SMOTE balancing...")
try:
    smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y_train)) - 1))
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE balancing: {y_train_bal.value_counts().to_dict()}")
except Exception as e:
    print(f"⚠️ SMOTE failed ({e}), using RandomOverSampler instead.")
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
    print(f"✅ After Random Oversampling: {y_train_bal.value_counts().to_dict()}")

# ---------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# Define models with class_weight='balanced' where possible
# ---------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=300, n_jobs=-1, class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42,
        n_jobs=-1, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, scale_pos_weight=(len(y_train_bal[y_train_bal==0]) / max(1,len(y_train_bal[y_train_bal==1]))),
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=64,
        colsample_bytree=0.9, subsample=0.8, random_state=42,
        class_weight="balanced", n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42
    ),
}

# ---------------------------------------------------------------------
# Train & Evaluate
# ---------------------------------------------------------------------
report = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train_bal)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    cr = classification_report(y_test, preds, output_dict=True)
    f1 = cr["weighted avg"]["f1-score"]
    prec = cr["weighted avg"]["precision"]
    rec = cr["weighted avg"]["recall"]

    print(f"✅ {name}: F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | AUC={auc:.4f}")
    print(confusion_matrix(y_test, preds))

    report[name] = {"F1": f1, "Precision": prec, "Recall": rec, "AUC": auc}

    if f1 > best_score:
        best_score = f1
        best_model = (name, model)

# ---------------------------------------------------------------------
# Save best model & scaler
# ---------------------------------------------------------------------
best_name, best_clf = best_model
joblib.dump(best_clf, f"outputs/{best_name}_balanced.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")

with open("outputs/model_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n🏆 Best Model: {best_name} | F1 = {best_score:.4f}")
print("📁 Outputs saved in /outputs/")
