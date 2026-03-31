import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from model import Model  # On réutilise la classe Model existante

class CrossValidation:
    @staticmethod
    def run_cv(X_train, y_train, n_splits=5, random_state=42):
        """
        Effectue une validation croisée stratifiée sur les données d'entraînement.
        Retourne les scores F1 moyens et les écarts-types pour chaque fold.

        Args:
            X_train (DataFrame): Features d'entraînement.
            y_train (Series): Labels d'entraînement.
            n_splits (int): Nombre de folds (défaut: 5).
            random_state (int): Graine aléatoire pour la reproductibilité.

        Returns:
            dict: Dictionnaire avec les scores F1 moyens, écarts-types, et détails par fold.
        """
        # Initialiser StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialiser le modèle (mêmes paramètres que dans Model.train)
        model_template = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1
        )

        # Stockage des scores F1 et des meilleurs itérations pour chaque fold
        f1_scores = []
        best_iterations = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n--- Fold {fold + 1}/{n_splits} ---")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Entraînement du modèle (comme dans Model.train)
            model = model_template.__class__(**model_template.get_params())  # Copie du modèle
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    __import__('lightgbm').early_stopping(50, verbose=False),
                    __import__('lightgbm').log_evaluation(100)
                ]
            )

            # Prédiction et calcul du F1-score
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
            best_iterations.append(model.best_iteration_)

            print(f"F1-score: {f1:.4f} | Meilleur itération: {model.best_iteration_}")

        # Calcul des statistiques
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        mean_iter = np.mean(best_iterations)

        print("\n" + "=" * 50)
        print(f"Cross-validation (StratifiedKFold, {n_splits} folds):")
        print(f"F1-score moyen: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Nombre moyen d'arbres: {mean_iter:.1f}")
        print("=" * 50)

        # Retourne les résultats
        return {
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_iter": mean_iter,
            "f1_scores": f1_scores,
            "best_iterations": best_iterations
        }

    @staticmethod
    def evaluate_with_cv(X_train, y_train, n_splits=5):
        """
        Évalue le modèle avec une validation croisée et affiche les résultats.
        Utilise la méthode run_cv ci-dessus.
        """
        cv_results = CrossValidation.run_cv(X_train, y_train, n_splits=n_splits)

        # Affichage des résultats par fold
        print("\nRésultats détaillés par fold:")
        for i, (f1, iter) in enumerate(zip(cv_results["f1_scores"], cv_results["best_iterations"])):
            print(f"Fold {i + 1}: F1={f1:.4f} | Arbres={iter}")

        return cv_results