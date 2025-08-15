from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

def build_pipeline(preprocessor, params=None):
    if params is None:
        params = dict(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf = XGBClassifier(**params)
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    return pipe

def save_model(pipe, path):
    joblib.dump(pipe, path)

def load_model(path):
    return joblib.load(path)
