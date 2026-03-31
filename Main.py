import os
import pandas as pd
from pretraitement import Preview
from model import Model

# --- Chargement ---
df_data     = Preview.load_and_preview(Preview.file_data)
df_truth    = Preview.load_and_preview(Preview.file_ground_truth_train)
df_test_idx = pd.read_csv(Preview.file_test, header=None)
test_seqns  = df_test_idx[0].values  # Les SEQN du test set

df = df_data.merge(df_truth, on='SEQN', how='left')

cols_a_garder = ['URXUMA', 'INDFMPIR', 'LBDBPBSI', 'merge_1327', 'BMXARMC']

# --- Nettoyage global (structure commune train+test) ---
df_clean, _, dropped_cols = Preview.clean(
    df, target_col='MORTSTAT_2019',
    nan_thresh=0.61, corr_thresh=0.95,
    var_thresh=0.01, scale=False,
    keep_cols=cols_a_garder
)
df_clean.to_csv(os.path.join('data', 'data_clean.csv'), index=False)
print("Données nettoyées sauvegardées")

# --- Séparation AVANT l'imputation ---
df_train_clean = df_clean[~df_clean['SEQN'].isin(test_seqns)].copy()
df_test_clean  = df_clean[ df_clean['SEQN'].isin(test_seqns)].copy()

print(f"Train : {df_train_clean.shape} | Test : {df_test_clean.shape}")

# --- Analyse sur le train uniquement ---
report = Preview.analyze_columns(df_train_clean)

# --- Imputation séparée ---
df_train_imputed = Preview.smart_impute(df_train_clean, report)
df_test_imputed  = Preview.smart_impute(df_test_clean,  report)

# --- Reconstitution d'un fichier unique pour load_data() ---
df_test_imputed['MORTSTAT_2019'] = float('nan')
df_all_imputed = pd.concat([df_train_imputed, df_test_imputed], ignore_index=True)
df_all_imputed.to_csv(os.path.join('data', 'data_imputed.csv'), index=False)
print("Données imputées sauvegardées")

# --- Version scalée (optionnel, pour SVM/LogReg) ---
df_imputed_scaled, scaler = Preview.scale_features(df_train_imputed, target_col='MORTSTAT_2019')
df_imputed_scaled.to_csv(os.path.join('data', 'data_imputed_scaled.csv'), index=False)
print("Données imputées et normalisées sauvegardées")

# --- Vérification ---
Preview.analyze_missing_values(df_all_imputed, 'data_imputed')

# --- Modèle ---
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