"""
Corridor Risk Intelligence — Modèles finaux retenus
1) Classification : retard_corridor (ROC-AUC 0.93)
2) Classification : dépassement du seuil de franchise douane >72h (ROC-AUC 0.79)
Ce sont les deux modèles mis en avant dans le portfolio (voir historique de
décision dans le README : le modèle de régression en heures exactes a été
écarté comme livrable principal, R²=0.42 jugé insuffisant pour une mise en avant).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                              accuracy_score, confusion_matrix)
import xgboost as xgb
import shap
import joblib

df = pd.read_csv("/mnt/user-data/outputs/corridor_risk_dataset.csv")
df["date_arrivee_port"] = pd.to_datetime(df["date_arrivee_port"])
df["mois"] = df["date_arrivee_port"].dt.month
df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)
df["douane_bloque"] = (df["statut_douane"] == "bloqué (litige/documents)").astype(int)
df["douane_en_cours"] = (df["statut_douane"] == "dédouanement en cours").astype(int)
df["congestion_x_douane_bloque"] = df["volume_journalier_terminal"] * df["douane_bloque"]
df["congestion_sq"] = df["volume_journalier_terminal"] ** 1.6
df["volume_journalier_log"] = np.log1p(df["volume_journalier_terminal"])
df["depassement_franchise"] = (df["temps_sejour_port_heures"] > 72).astype(int)

cat_cols = ["terminal", "type_marchandise", "compagnie_transport", "destination_finale",
            "jour_semaine", "saison", "statut_douane"]
num_cols = ["volume_evp", "volume_journalier_terminal", "distance_corridor_km",
            "mois_sin", "mois_cos", "douane_bloque", "douane_en_cours",
            "congestion_x_douane_bloque", "congestion_sq", "volume_journalier_log"]
feature_cols = cat_cols + num_cols

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="passthrough")


def train_and_save(target_col, model_path, label):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                     subsample=0.85, colsample_bytree=0.85,
                                     eval_metric="logloss", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"=== {label} ===")
    print(f"Accuracy  : {accuracy_score(y_test, pred):.3f}")
    print(f"Precision : {precision_score(y_test, pred):.3f}")
    print(f"Recall    : {recall_score(y_test, pred):.3f}")
    print(f"F1        : {f1_score(y_test, pred):.3f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, proba):.3f}")
    print("Matrice de confusion :\n", confusion_matrix(y_test, pred))
    print()

    joblib.dump(pipeline, model_path)
    return pipeline, X_test


clf_retard, Xtest_retard = train_and_save(
    "retard_corridor", "/mnt/user-data/outputs/modele_retard_corridor.pkl",
    "Modèle 1 — Retard corridor Dakar-Bamako")

clf_franchise, Xtest_franchise = train_and_save(
    "depassement_franchise", "/mnt/user-data/outputs/modele_depassement_franchise.pkl",
    "Modèle 2 — Dépassement du seuil de franchise douane (>72h)")

# Sauvegarde de la liste des features et des valeurs possibles (pour l'app Streamlit)
feature_meta = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "options": {c: sorted(df[c].unique().tolist()) for c in cat_cols},
}
joblib.dump(feature_meta, "/mnt/user-data/outputs/feature_meta.pkl")
print("Modèles + métadonnées sauvegardés dans /mnt/user-data/outputs/")
