from ast import Import
import os
# from CrossValidation import CrossValidation
import pandas as pd
from pretraitement import Preview
# from model import Model
from trainmodels import prepare_xy, evaluate_model, get_models
from sklearn.model_selection import train_test_split

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

# # --- Modèle (charge data_imputed.csv) ---
# X_train, y_train, X_test, seqn_test = Model.load_data()
# # --- Validation croisée ---
# cv_results = CrossValidation.evaluate_with_cv(X_train, y_train, n_splits=5)

# # --- Entraînement du modèle final ---
# model, X_val, y_val                 = Model.train(X_train, y_train)
# # --- Évaluation ---
# f1, best_threshold                  = Model.evaluate(model, X_val, y_val)
# # --- Importance des features ---
# feat_imp                            = Model.feature_importance(model, X_train, top_n=30)

# # --- Soumission ---
# Model.predict_and_submit(
#     model, X_test, seqn_test,
#     group_id='4',
#     submission_id='2',
#     threshold=best_threshold
# )


# --- Tester des modèles classiques ---
df_models = pd.read_csv(os.path.join('data', 'data_imputed_scaled.csv'))

df_models = df_models[df_models['MORTSTAT_2019'].notna()].copy()
df_models['MORTSTAT_2019'] = df_models['MORTSTAT_2019'].astype(int)

X, y = prepare_xy(df_models, target_col='MORTSTAT_2019', id_col='SEQN')

X_train_m, X_valid_m, y_train_m, y_valid_m = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = get_models()
all_results = []

for name, model in models.items():
    try:
        result = evaluate_model(name, model, X_train_m, X_valid_m, y_train_m, y_valid_m)
        all_results.append(result)
    except Exception as e:
        print(f"Erreur avec le modèle {name}: {e}")

results_df = pd.DataFrame(all_results).sort_values(by='F1', ascending=False)
print("\n=== Comparaison des modèles ===")
print(results_df.to_string(index=False))

os.makedirs("results", exist_ok=True)
results_df.to_csv(os.path.join("results", "model_comparison.csv"), index=False)
print("Résultats sauvegardés dans results/model_comparison.csv")
