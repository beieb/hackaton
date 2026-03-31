import os
import pandas as pd
from pretraitement import Preview
from model import Model

# --- Chargement ---
df_data  = Preview.load_and_preview(Preview.file_data)
df_truth = Preview.load_and_preview(Preview.file_ground_truth_train)
df       = df_data.merge(df_truth, on='SEQN', how='left')

df_clean, _, dropped_cols = Preview.clean(
    df, target_col='MORTSTAT_2019',
    nan_thresh=0.61, corr_thresh=0.95,
    var_thresh=0, scale=False
)
df_clean.to_csv(os.path.join('data', 'data_clean.csv'), index=False)
print("Données nettoyées sauvegardées")

# --- Analyse et imputation ---
report     = Preview.analyze_columns(df_clean)
df_imputed = Preview.smart_impute(df_clean, report)

# Sauvegarder la version imputée (sans scale) → utilisée par LightGBM / XGBoost
df_imputed.to_csv(os.path.join('data', 'data_imputed.csv'), index=False)
print("Données imputées sauvegardées")

# Sauvegarder la version imputée + scalée → utilisée si tu testes SVM / LogReg
df_imputed_scaled, scaler = Preview.scale_features(df_imputed, target_col='MORTSTAT_2019')
df_imputed_scaled.to_csv(os.path.join('data', 'data_imputed_scaled.csv'), index=False)
print("Données imputées et normalisées sauvegardées")

# --- Vérification ---
Preview.analyze_missing_values(df_imputed, 'data_imputed')

# --- Modèle (charge data_imputed.csv) ---
X_train, y_train, X_test, seqn_test = Model.load_data()
model, X_val, y_val                 = Model.train(X_train, y_train)
f1, best_threshold                  = Model.evaluate(model, X_val, y_val)
feat_imp                            = Model.feature_importance(model, X_train, top_n=30)

# --- Soumission ---
Model.predict_and_submit(
    model, X_test, seqn_test,
    group_id='4',
    submission_id='2',
    threshold=best_threshold
)