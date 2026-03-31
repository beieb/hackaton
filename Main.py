from ast import Import
import os
from CrossValidation import CrossValidation
import pandas as pd
from pretraitement import Preview
from model import Model

# ─── Main ───────────────────────────────────────────────────────────────────────
# Charger les données et le ground truth
df_data = Preview.load_and_preview(Preview.file_data)
df_truth = Preview.load_and_preview(Preview.file_ground_truth_train)

# Fusionner pour avoir la cible dans le DataFrame
df = df_data.merge(df_truth, on='SEQN', how='left')

# Nettoyer (la fonction retourne 3 valeurs : df_clean, scaler, dropped_cols)
df_clean, scaler, dropped_cols = Preview.clean(
    df,
    target_col='MORTSTAT_2019',
    nan_thresh=0.61,
    corr_thresh=0.95,
    var_thresh=0.01,
    nan_row_thresh=0.7,
    scale=True
)

# Sauvegarder le dataset nettoyé
df_clean.to_csv(os.path.join('data', 'data_clean.csv'), index=False)
print(f"\nDonnées nettoyées sauvegardées")

# Sauvegarder le scaler pour une utilisation future (ex : normaliser les données de test)
import joblib
joblib.dump(scaler, os.path.join('models', 'scaler.joblib'))
print("Scaler sauvegardé")

# Afficher les colonnes supprimées pour corrélation
print(f"\nColonnes supprimées pour corrélation : {len(dropped_cols)}")
if len(dropped_cols) > 0:
    print(f"Exemples : {dropped_cols[:5]}...")  # Affiche les 5 premières

# ─── Chargement des données pour le modèle ─────────────────────────────────────
X_train, y_train, X_test, seqn_test = Model.load_data()

# --- Validation croisée ---
cv_results = CrossValidation.evaluate_with_cv(X_train, y_train, n_splits=5)

# --- Entraînement du modèle final ---
model, X_val, y_val = Model.train(X_train, y_train)

# --- Évaluation ---
f1, best_threshold = Model.evaluate(model, X_val, y_val)

# --- Importance des features ---
feat_imp = Model.feature_importance(model, X_train, top_n=30)

# --- Soumission ---
Model.predict_and_submit(
    model, X_test, seqn_test,
    group_id='4',       # Remplace par ton ID de groupe
    submission_id='2',
    threshold=best_threshold
)