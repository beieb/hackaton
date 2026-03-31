import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns


class Model:

    @staticmethod
    def load_data():
        df = pd.read_csv(os.path.join('data', 'data_imputed.csv'))

        
        df_train = df[df['MORTSTAT_2019'].notna()].copy()
        df_test  = df[df['MORTSTAT_2019'].isna()].copy()

        X_train = df_train.drop(columns=['SEQN', 'MORTSTAT_2019'])
        y_train = df_train['MORTSTAT_2019'].astype(int)
        X_test  = df_test.drop(columns=['SEQN', 'MORTSTAT_2019'])
        seqn_test = df_test['SEQN'].astype(int)

        print(f"Train : {X_train.shape} | Test : {X_test.shape}")
        print(f"\nDistribution des classes :")
        print(y_train.value_counts())
        print(f"Ratio vivants/morts : {y_train.value_counts(normalize=True).round(3).to_dict()}")

        return X_train, y_train, X_test, seqn_test

    @staticmethod
    def train(X_train, y_train):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train        # Préserve le ratio vivants/morts dans les deux splits
        )

        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced', # Compense le déséquilibre vivants/morts
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__('lightgbm').early_stopping(50, verbose=False),  # Stop si pas d'amélioration après 50 rounds
                __import__('lightgbm').log_evaluation(100)
            ]
        )

        print(f"\nMeilleur nombre d'arbres : {model.best_iteration_}")

        return model, X_val, y_val

    @staticmethod
    def evaluate(model, X_val, y_val):
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        print("\n" + "=" * 50)
        print(f"F1-score : {f1:.4f}")
        print("\nRapport de classification :")
        print(classification_report(y_val, y_pred, target_names=['Vivant (0)', 'Mort (1)']))

        # --- Matrice de confusion ---
        os.makedirs(os.path.join('fig', 'model'), exist_ok=True)
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Prédit Vivant', 'Prédit Mort'],
                    yticklabels=['Vrai Vivant', 'Vrai Mort'])
        plt.title(f'Matrice de confusion (F1={f1:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join('fig', 'model', 'confusion_matrix.png'), dpi=150)
        plt.close()
        print("Matrice de confusion sauvegardée")

        # --- Optimisation du seuil de décision ---
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.9, 0.01):
            y_pred_t = (y_proba >= thresh).astype(int)
            score = f1_score(y_val, y_pred_t)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        print(f"\nSeuil optimal : {best_thresh:.2f} → F1 = {best_f1:.4f}")
        print("=" * 50)

        return f1, best_thresh

    @staticmethod
    def feature_importance(model, X_train, top_n=30):
        feat_imp = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print(f"\nTop {top_n} features les plus importantes :")
        print(feat_imp.head(top_n).to_string())

        # --- Graphique ---
        plt.figure(figsize=(10, 8))
        feat_imp.head(top_n).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.title(f'Top {top_n} features importantes')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join('fig', 'model', 'feature_importance.png'), dpi=150)
        plt.close()
        print("Feature importance sauvegardée")

        return feat_imp

    @staticmethod
    def predict_and_submit(model, X_test, seqn_test, group_id, submission_id, threshold=0.5):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

        submission = pd.DataFrame({
            'SEQN': seqn_test,
            'prediction': y_pred
        }).sort_values('SEQN')

        # Vérifications
        assert len(submission) == 5000, f"Erreur : {len(submission)} lignes au lieu de 5000"
        assert submission['prediction'].isin([0, 1]).all(), "Erreur : prédictions non binaires"

        filename = f"{group_id}_{submission_id}.csv"
        submission.to_csv(filename, index=False, header=False)
        print(f"\nSoumission sauvegardée : {filename}")
        print(f"Distribution des prédictions : {submission['prediction'].value_counts().to_dict()}")

        return submission


