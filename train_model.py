"""
Train and save the AI4I 2020 predictive-maintenance Random Forest pipeline.

This reproduces the final model from the exploration notebook (same feature
set, same preprocessing, same RandomForestClassifier hyperparameters) without
any of the Colab-only or exploratory-analysis code.

Run this once, wherever you have normal internet access (Google Colab or your
own machine), to produce:
    model_artifacts/random_forest_pipeline.joblib
    model_artifacts/model_metadata.json

Then copy (or git add) the model_artifacts/ folder into this project before
deploying the Streamlit app.

Usage:
    python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/601/data.csv"
ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
SELECTED_THRESHOLD = 0.50
RANDOM_STATE = 42

TARGET = "Machine failure"
COLUMNS_TO_EXCLUDE = [
    "UID", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
]


def load_data() -> pd.DataFrame:
    print(f"Loading dataset from {DATA_URL} ...")
    df = pd.read_csv(DATA_URL)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_features = ["Type"]
    numerical_features = [c for c in X_train.columns if c not in categorical_features]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numerical", StandardScaler(), numerical_features),
        ],
        remainder="drop",
    )


def main() -> None:
    df = load_data()

    X = df.drop(columns=COLUMNS_TO_EXCLUDE)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    final_rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("Fitting Random Forest pipeline on the training split...")
    final_rf_model.fit(X_train, y_train)

    y_test_prob = final_rf_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= SELECTED_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

    test_metrics = {
        "accuracy": float((y_test_pred == y_test.to_numpy()).mean()),
        "precision_failure": precision,
        "recall_failure": recall,
        "f1_failure": f1,
        "average_precision": float(average_precision_score(y_test, y_test_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    print("\nTest-set metrics at threshold", SELECTED_THRESHOLD)
    print(json.dumps(test_metrics, indent=2))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "random_forest_pipeline.joblib"
    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    joblib.dump(final_rf_model, model_path, compress=3)

    metadata = {
        "project": "Industrial Predictive Maintenance AI",
        "dataset": "AI4I 2020 Predictive Maintenance Dataset",
        "model": "Random Forest Classifier",
        "target": TARGET,
        "input_features": list(X.columns),
        "classification_threshold": SELECTED_THRESHOLD,
        "threshold_selection_basis": "Validation-set analysis",
        "test_metrics": test_metrics,
        "limitations": [
            "AI4I is a synthetic benchmark dataset, not live sensor data.",
            "The selected threshold has not been validated in a real factory.",
            "Dataset failure labels have documented inconsistencies with the individual failure-mode flags.",
        ],
        "scikit_learn_version": sklearn.__version__,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved model to:    {model_path.resolve()}")
    print(f"Saved metadata to: {metadata_path.resolve()}")
    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
