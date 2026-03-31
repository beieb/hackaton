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
        plt.figure(figsize=(10, 6))
        missing_values[missing_values > 0].plot(kind='bar')
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


# ─── Main ───────────────────────────────────────────────────────────────────────
df_data = Preview.load_and_preview(Preview.file_data)
Preview.analyze_missing_values(df_data, 'data')
corr = Preview.correlation_matrix(df_data, 'data', threshold=0.8)