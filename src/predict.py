import argparse, json
from src.model import load_model
import pandas as pd

def main(model_path, record_json):
    model = load_model(model_path)
    record = json.loads(record_json)
    df = pd.DataFrame([record])
    probability = model.predict_proba(df)[:,1][0]
    print({"probability": float(probability), "prediction": int(probability > 0.5)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    main(args.model, args.record)
