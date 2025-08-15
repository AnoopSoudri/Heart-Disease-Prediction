import os
import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def main(config_path):
    # ---- Load Data ----
    df = pd.read_csv("data/raw/heart.csv", sep=None, engine="python")
    
    # Handle case where column names might be in one string
    if len(df.columns) == 1:
        df = pd.read_csv("data/raw/heart.csv", sep="\t")

    if "target" not in df.columns:
        raise ValueError(f"'target' column not found. Available columns: {df.columns.tolist()}")

    X = df.drop("target", axis=1)
    y = df["target"]

    # ---- Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Scale Data ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Train Model ----
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # ---- Save Model & Scaler ----
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model and scaler saved in 'models/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
