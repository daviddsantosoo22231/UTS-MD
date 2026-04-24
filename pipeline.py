import os, pickle
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

FEATURES_PATH = "A.csv"
TARGETS_PATH  = "A_targets.csv"
CLS_MODEL_OUT = "best_cls_model.pkl"
REG_MODEL_OUT = "best_reg_model.pkl"

CAT_FEATS = ["gender","branch","part_time_job","family_income_level","city_tier","internet_access","extracurricular_involvement"]
NUM_FEATS = ["cgpa","tenth_percentage","twelfth_percentage","backlogs","study_hours_per_day","attendance_percentage",
             "projects_completed","internships_completed","coding_skill_rating","communication_skill_rating",
             "aptitude_skill_rating","hackathons_participated","certifications_count","sleep_hours","stress_level",
             "skill_composite","academic_score"]

def load_dataset(fp, tp):
    df = pd.read_csv(fp).merge(pd.read_csv(tp), on="Student_ID")
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows")
    return df

def feature_engineering(df):
    df = df.copy()
    df["skill_composite"] = (df["coding_skill_rating"] + df["communication_skill_rating"] + df["aptitude_skill_rating"]) / 3
    df["academic_score"]  = df["cgpa"]*10*0.5 + df["tenth_percentage"]*0.25 + df["twelfth_percentage"]*0.25
    return df

def prepare_data(df):
    df = feature_engineering(df).drop(columns=["Student_ID"])
    X = df[NUM_FEATS + CAT_FEATS]
    y_cls = (df["placement_status"] == "Placed").astype(int)
    y_reg = df["salary_lpa"]
    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    _,    _,    yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    return X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te

def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_FEATS), ("cat", cat, CAT_FEATS)])

def train_cls(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment("placement_classification")
    with mlflow.start_run(run_name="GBClassifier"):
        p = {"n_estimators":100,"learning_rate":0.1,"max_depth":4}
        mlflow.log_params(p)
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", GradientBoostingClassifier(**p, random_state=42))])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mlflow.log_metrics({"accuracy": accuracy_score(y_te, preds), "f1": f1_score(y_te, preds)})
        mlflow.sklearn.log_model(pipe, "cls_model")
        print(f"[CLS] Accuracy={accuracy_score(y_te, preds):.4f}")
    pickle.dump(pipe, open(CLS_MODEL_OUT, "wb"))
    print(f"Saved -> {CLS_MODEL_OUT}")
    return pipe

def train_reg(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment("salary_regression")
    with mlflow.start_run(run_name="GBRegressor"):
        p = {"n_estimators":100,"learning_rate":0.1,"max_depth":4}
        mlflow.log_params(p)
        pipe = Pipeline([("pre", build_preprocessor()), ("reg", GradientBoostingRegressor(**p, random_state=42))])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mlflow.log_metrics({"rmse": mean_squared_error(y_te, preds)**0.5, "r2": r2_score(y_te, preds)})
        mlflow.sklearn.log_model(pipe, "reg_model")
        print(f"[REG] R2={r2_score(y_te, preds):.4f}")
    pickle.dump(pipe, open(REG_MODEL_OUT, "wb"))
    print(f"Saved -> {REG_MODEL_OUT}")
    return pipe

if __name__ == "__main__":
    df = load_dataset(FEATURES_PATH, TARGETS_PATH)
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = prepare_data(df)
    train_cls(X_tr, X_te, yc_tr, yc_te)
    train_reg(X_tr, X_te, yr_tr, yr_te)
    print("[DONE] Run: mlflow ui")