import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


class Preview:
    file_data = os.path.join('data', 'data.csv')
    file_feat = os.path.join('data', 'feature_metadata.csv')
    file_ground_truth_train = os.path.join('data', 'ground_truth_train.csv')
    file_test = os.path.join('data', 'test_indexes.csv')

    @staticmethod
    def load_and_preview(path):
        data = pd.read_csv(path)
        return data

    @staticmethod
    def analyze_missing_values(df, name):
        missing_values = df.isnull().sum()
        print("Missing values per column:\n", missing_values)
        total_cells = df.size
        total_missing = missing_values.sum()
        print("Total missing values:", total_missing)
        print("Percentage of missing values:", (total_missing / total_cells) * 100)
        Preview.show_as_graph(
            missing_values,
            os.path.join('fig', 'missing_value', f'missing_value_{name}.png')
        )

    @staticmethod
    def show_as_graph(missing_values, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        filtered = missing_values[missing_values > 0]

        if filtered.empty:
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, 'Aucune valeur manquante',

                    ha='center', va='center', fontsize=14,
                    transform=plt.gca().transAxes)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            print(f"Aucune valeur manquante — graphique vide sauvegardé : {path}")
            return

        plt.figure(figsize=(10, 6))
        filtered.plot(kind='bar')
        plt.title('Missing Values per Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def correlation_matrix(df, name, threshold=0.8):
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric = df_numeric.dropna(thresh=len(df_numeric) * 0.5, axis=1)

        # Supprimer les colonnes avec nom vide ou NaN
        df_numeric = df_numeric.loc[:, df_numeric.columns.notna()]
        df_numeric = df_numeric.loc[:, df_numeric.columns.str.strip() != '']

        print(f"Colonnes utilisées pour la corrélation : {df_numeric.shape[1]}")

        corr_matrix = df_numeric.corr()

        high_corr_cols = (corr_matrix.abs() > threshold).any(axis=1)
        corr_filtered = corr_matrix.loc[high_corr_cols, high_corr_cols]

        print(f"Colonnes avec corrélation > {threshold} : {corr_filtered.shape[0]}")

        output_path = os.path.join('fig', 'correlation', f'correlation_{name}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        n = corr_filtered.shape[0]
        fig_size = max(20, n * 0.35)  # Taille dynamique selon le nombre de colonnes

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            corr_filtered,
            annot=False,
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.1,
            square=True,
            ax=ax
        )
        plt.title(f'Matrice de corrélation (|r| > {threshold}) — {name}')

        # Forcer l'affichage de tous les labels
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr_filtered.columns, rotation=90, fontsize=max(6, 10 - n // 20))
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr_filtered.index, rotation=0, fontsize=max(6, 10 - n // 20))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')  # bbox_inches évite les coupures
        plt.close()
        print(f"Matrice sauvegardée : {output_path}")

        Preview.print_high_corr_pairs(corr_matrix, threshold, name)

        return corr_matrix

    @staticmethod
    def print_high_corr_pairs(corr_matrix, threshold, name):
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > threshold:
                    pairs.append((cols[i], cols[j], round(val, 4)))

        pairs_df = pd.DataFrame(pairs, columns=['Variable 1', 'Variable 2', 'Corrélation'])
        pairs_df = pairs_df.reindex(
            pairs_df['Corrélation'].abs().sort_values(ascending=False).index
        )

        print(f"\nPaires avec |corrélation| > {threshold} :\n", pairs_df.to_string(index=False))

        csv_path = os.path.join('fig', 'correlation', f'high_corr_pairs_{name}.csv')
        pairs_df.to_csv(csv_path, index=False)
        print(f"Paires sauvegardées : {csv_path}")


    @staticmethod
    def clean(df, target_col=None, nan_thresh=0.80, corr_thresh=0.95, var_thresh=0,
            nan_row_thresh=0.7, scale=True, keep_cols=None):
        keep_cols = keep_cols or []

        print("=" * 60)
        print(f"Shape initiale : {df.shape}")

        # --- Étape 1 : Séparer la cible ---
        target = None
        if target_col and target_col in df.columns:
            target = df[target_col]
            df = df.drop(columns=[target_col])
            print(f"Colonne cible '{target_col}' mise de côté")

        # --- Étape 2 : Supprimer les colonnes avec trop de NaN ---
        before = df.shape[1]
        cols_to_check = [c for c in df.columns if c not in keep_cols]  # exclure keep_cols
        cols_to_keep_nan = df[cols_to_check].dropna(
            thresh=len(df) * (1 - nan_thresh), axis=1
        ).columns.tolist()
        df = df[cols_to_keep_nan + [c for c in keep_cols if c in df.columns]]
        after = df.shape[1]
        print(f"\n[1] Suppression NaN > {(1-nan_thresh)*100:.0f}% : {before} → {after} colonnes (-{before - after})")

        # --- Étape 3 : Séparer numériques / non numériques ---
        df_numeric     = df.select_dtypes(include=[np.number])
        df_non_numeric = df.select_dtypes(exclude=[np.number])
        print(f"[2] Colonnes numériques : {df_numeric.shape[1]} | non-numériques : {df_non_numeric.shape[1]}")

        # --- Étape 4 : Supprimer les colonnes à variance quasi nulle ---
        before = df_numeric.shape[1]
        from sklearn.feature_selection import VarianceThreshold
        selector  = VarianceThreshold(threshold=var_thresh)
        df_filled = df_numeric.fillna(df_numeric.median())
        selector.fit(df_filled)
        supported = df_numeric.columns[selector.get_support()].tolist()
        # Réintégrer les keep_cols même si variance faible
        protected = [c for c in keep_cols if c in df_numeric.columns and c not in supported]
        df_numeric = df_numeric[supported + protected]
        after = df_numeric.shape[1]
        print(f"[3] Suppression variance < {var_thresh} : {before} → {after} colonnes (-{before - after})")
        if protected:
            print(f"    Colonnes protégées réintégrées : {protected}")
        df = Preview.drop_useless_rows(df, nan_row_thresh=nan_row_thresh)
        print(f"[4] Après suppression des lignes : {df.shape}")

        before = df_numeric.shape[1]
        df_filled = df_numeric.fillna(df_numeric.median())
        corr_matrix = df_filled.corr().abs()
        upper   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Ne pas supprimer les keep_cols même si corrélées
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > corr_thresh) and col not in keep_cols
        ]
        df_numeric = df_numeric.drop(columns=to_drop)
        after = df_numeric.shape[1]
        print(f"[5] Suppression corrélation > {corr_thresh} : {before} → {after} colonnes (-{before - after})")
        print(f"    Colonnes supprimées : {to_drop}")

        # --- Étape 6 : Supprimer les lignes problématiques ---

        df = Preview.drop_useless_rows(df, nan_row_thresh=nan_row_thresh)
        print(f"[4] Après suppression des lignes : {df.shape}")

        # --- Étape 7 : Normalisation ---
        if scale:
            df_scaled, scaler = Preview.scale_features(
                pd.concat([df_numeric, df_non_numeric], axis=1), target_col=target_col
            )
            print(f"[6] Normalisation des features numériques")
        else:
            df_scaled = pd.concat([df_numeric, df_non_numeric], axis=1)
            scaler    = None

        # --- Reconstruction ---
        if target is not None:
            df_scaled[target_col] = target.values

        print(f"\nShape finale : {df_scaled.shape}")
        print("=" * 60)

        return df_scaled, scaler, to_drop
    
    @staticmethod
    def drop_useless_rows(df, nan_row_thresh=0.7, verbose=True):
        """
        Supprime les lignes :
        - Complètement vides.
        - Avec un taux de NaN supérieur à `nan_row_thresh` (défaut : 70 %).
        - Dupliquées.

        Args:
            df (DataFrame) : Dataset à nettoyer.
            nan_row_thresh (float) : Seuil de NaN pour supprimer une ligne (ex: 0.7 = 70 %).
            verbose (bool) : Afficher les logs.

        Returns:
            DataFrame nettoyé.
        """
        initial_rows = df.shape[0]

        # Supprimer les lignes complètement vides
        df = df.dropna(how='all')
        empty_rows_removed = initial_rows - df.shape[0]

        # Supprimer les lignes avec trop de NaN
        nan_per_row = df.isnull().mean(axis=1)
        df = df[nan_per_row < nan_row_thresh]
        nan_rows_removed = initial_rows - empty_rows_removed - df.shape[0]

        # Supprimer les doublons
        df = df.drop_duplicates()
        dup_rows_removed = initial_rows - empty_rows_removed - nan_rows_removed - df.shape[0]

        if verbose:
            print(f"Lignes supprimées :")
            print(f"  - Vides : {empty_rows_removed}")
            print(f"  - Trop de NaN (> {nan_row_thresh*100}%) : {nan_rows_removed}")
            print(f"  - Dupliquées : {dup_rows_removed}")
            print(f"  - Total : {initial_rows - df.shape[0]} ({100*(initial_rows - df.shape[0])/initial_rows:.1f} %)")
            print(f"Lignes restantes : {df.shape[0]}")

        return df
    
    @staticmethod
    def scale_features(df, target_col=None, verbose=True):
        """
        Normalise les features numériques avec StandardScaler.
        Conserve les colonnes non numériques et la cible.

        Args:
            df (DataFrame) : Dataset à normaliser.
            target_col (str) : Nom de la colonne cible (optionnel).
            verbose (bool) : Afficher les logs.

        Returns:
            DataFrame normalisé, objet StandardScaler ajusté.
        """
        from sklearn.preprocessing import StandardScaler

        # Séparer les colonnes numériques et non numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Exclure la cible si spécifiée
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
            non_numeric_cols.append(target_col)

        if verbose:
            print(f"Colonnes numériques à normaliser : {len(numeric_cols)}")
            print(f"Colonnes non numériques/ignorées : {len(non_numeric_cols)}")

        # Initialiser et ajuster le scaler
        scaler = StandardScaler()
        df_scaled = df.copy()
        if len(numeric_cols) > 0:
            df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df_scaled, scaler

    @staticmethod                              # ← bien indenté dans la classe
    def analyze_columns(df, nan_thresh_report=0.2):
        results = []
        for col in df.columns:
            series      = df[col].dropna()
            n_total     = len(df[col])
            n_missing   = df[col].isnull().sum()
            pct_missing = n_missing / n_total
            unique_vals = sorted(series.unique())
            n_unique    = len(unique_vals)

            is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})

            if is_binary:
                col_type = 'binaire'
                rate_1   = series.mean()
                nan_likely_zero = (pct_missing > 0.3) and (rate_1 < 0.5)
                suggestion = 'NaN → 0 (probable absent)' if nan_likely_zero else 'NaN → mode'
            elif n_unique <= 10:
                col_type   = 'catégorielle'
                suggestion = 'NaN → mode'
            else:
                col_type = 'continue'
                skewness = series.skew()
                suggestion = 'NaN → médiane' if abs(skewness) > 1 else 'NaN → moyenne'

            results.append({
                'colonne'      : col,
                'type'         : col_type,
                'n_unique'     : n_unique,
                'valeurs_uniq' : str(unique_vals[:5]),
                'pct_nan'      : round(pct_missing * 100, 1),
                'suggestion'   : suggestion
            })

        report = pd.DataFrame(results)

        print("\n=== Résumé des types de colonnes ===")
        print(report['type'].value_counts().to_string())
        print(f"\nColonnes avec > {nan_thresh_report*100:.0f}% de NaN :")
        print(report[report['pct_nan'] > nan_thresh_report * 100][
            ['colonne', 'type', 'pct_nan', 'suggestion']
        ].to_string(index=False))

        os.makedirs('fig', exist_ok=True)
        report.to_csv(os.path.join('fig', 'column_analysis.csv'), index=False)
        print("\nAnalyse sauvegardée dans fig/column_analysis.csv")

        return report
    

    @staticmethod                              # ← bien indenté dans la classe
    def smart_impute(df, column_report):
        df = df.copy()

        for _, row in column_report.iterrows():
            col        = row['colonne']
            suggestion = row['suggestion']
            if col not in df.columns:
                continue

            if 'NaN → 0' in suggestion:
                df[col] = df[col].fillna(0)
            elif 'mode' in suggestion:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            elif 'médiane' in suggestion:
                df[col] = df[col].fillna(df[col].median())
            elif 'moyenne' in suggestion:
                df[col] = df[col].fillna(df[col].mean())

        remaining_nan = df.isnull().sum().sum()
        print(f"NaN restants après imputation : {remaining_nan}")

        return df

'''
 ─── Main ───────────────────────────────────────────────────────────────────────
# Charger les données et le ground truth
df_data = Preview.load_and_preview(Preview.file_data)
df_truth = Preview.load_and_preview(Preview.file_ground_truth_train)

# Fusionner pour avoir la cible dans le DataFrame
df = df_data.merge(df_truth, on='SEQN', how='left')

# Nettoyer et normaliser
df_clean, scaler, dropped_cols = Preview.clean(
    df,
    target_col='MORTSTAT_2019',  # Corrige le nom de la colonne (majuscules)
    nan_thresh=0.20,               # Seuil pour les colonnes (20 % de NaN max)
    corr_thresh=0.90,              # Seuil de corrélation (ajustable)
    var_thresh=0.01,               # Seuil de variance
    nan_row_thresh=0.8,            # Seuil pour les lignes (80 % de NaN max)
    scale=True                     # Active la normalisation
)

# Sauvegarder le dataset nettoyé et normalisé
output_path = os.path.join('data', 'data_clean_scaled.csv')
df_clean.to_csv(output_path, index=False)
print(f"\nDonnées nettoyées, normalisées et sauvegardées : {output_path}")

# Optionnel : Sauvegarder le scaler pour réutilisation
import joblib
scaler_path = os.path.join('models', 'scaler.joblib')
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)
print(f"Scaler sauvegardé : {scaler_path}")
'''