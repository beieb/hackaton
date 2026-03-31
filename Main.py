import os
import pandas as pd
from pretraitement import Preview
from model import Model

# ─── Main ───────────────────────────────────────────────────────────────────────
# Charger les données et le ground truth
df_data = Preview.load_and_preview(Preview.file_data)
df_truth = Preview.load_and_preview(Preview.file_ground_truth_train)

# Fusionner pour avoir la cible dans le DataFrame
df = df_data.merge(df_truth, on='SEQN', how='left')

# Nettoyer
df_clean, dropped_cols = Preview.clean(df, target_col='MORTSTAT_2019', nan_thresh=0.61, corr_thresh=0.95, var_thresh=0.01)


# Sauvegarder
df_clean.to_csv(os.path.join('data', 'data_clean.csv'), index=False)
print(f"\nDonnées nettoyées sauvegardées")
df_clean = pd.read_csv(os.path.join('data', 'data_clean.csv'))

# 1. Analyser les colonnes
report = Preview.analyze_columns(df_clean)

# 2. Vérifier manuellement le CSV généré si besoin
# puis imputer intelligemment
df_imputed = Preview.smart_impute(df_clean, report)

# 3. Sauvegarder
df_imputed.to_csv(os.path.join('data', 'data_imputed.csv'), index=False)
print("Données imputées sauvegardées")

Preview.analyze_missing_values(df_imputed, 'data_imputed')

# ─── Main ────────────────────────────────────────────────────────────────────
X_train, y_train, X_test, seqn_test = Model.load_data()

model, X_val, y_val = Model.train(X_train, y_train)

f1, best_threshold = Model.evaluate(model, X_val, y_val)

feat_imp = Model.feature_importance(model, X_train, top_n=30)

# Soumission avec seuil optimisé
Model.predict_and_submit(
    model, X_test, seqn_test,
    group_id='4',       # ← Remplace par ton ID de groupe
    submission_id='2',
    threshold=best_threshold    # Utilise le seuil optimisé pour le F1
)