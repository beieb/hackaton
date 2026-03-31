import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_xy(df, target_col='MORTSTAT_2019', id_col='SEQN'):
    cols_to_drop = []
    if id_col in df.columns:
        cols_to_drop.append(id_col)
    if target_col in df.columns:
        cols_to_drop.append(target_col)

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    return X, y


def evaluate_model(name, model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    results = {
        'Model': name,
        'F1': f1_score(y_valid, y_pred),
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred, zero_division=0),
        'Recall': recall_score(y_valid, y_pred, zero_division=0)
    }

    print("=" * 70)
    print(f"Model: {name}")
    print("=" * 70)
    print(f"F1-score  : {results['F1']:.4f}")
    print(f"Accuracy  : {results['Accuracy']:.4f}")
    print(f"Precision : {results['Precision']:.4f}")
    print(f"Recall    : {results['Recall']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    return results


if __name__ == "__main__":
    data_path = os.path.join("data", "data_clean_scaled.csv")
    output_path = os.path.join("results", "model_comparison.csv")
    os.makedirs("results", exist_ok=True)

    df = load_data(data_path)
    X, y = prepare_xy(df, target_col='MORTSTAT_2019', id_col='SEQN')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    all_results = []

    for name, model in models.items():
        try:
            result = evaluate_model(name, model, X_train, X_valid, y_train, y_valid)
            all_results.append(result)
        except Exception as e:
            print(f"Erreur avec le modèle {name}: {e}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="F1", ascending=False)

    print("\n" + "=" * 70)
    print("Résumé des modèles")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés dans : {output_path}")

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }