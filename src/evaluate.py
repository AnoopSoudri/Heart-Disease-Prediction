# src/evaluate.py

import argparse
import yaml
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_loader import load_raw
from src.features import engineer_features

def evaluate_model(config_path):
    cfg = yaml.safe_load(open(config_path))
    model_path = cfg['training']['model_path']
    data_path = cfg['data']['raw']

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Load and prepare data
    df = load_raw(data_path)
    if df['target'].nunique() > 2:
        df['target'] = (df['target'] > 0).astype(int)

    df = engineer_features(df)

    # Train/test split (same params as training)
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['target'])
    y = df['target']
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg['training']['test_size'],
        stratify=y,
        random_state=cfg['training']['random_state']
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
