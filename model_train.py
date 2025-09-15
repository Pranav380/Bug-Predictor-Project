from __future__ import annotations
import argparse, joblib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

def get_features_labels(df: pd.DataFrame):
    y = df["buggy_label"].astype(int).values
    drop = ["buggy_label","file"]
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    num_cols = list(X.columns)
    return X, y, num_cols

def train_and_eval(df: pd.DataFrame, model_out: str):
    X, y, num_cols = get_features_labels(df)

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer(
        transformers=[("num", numeric, num_cols)],
        remainder="drop"
    )

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_f1, best_model = None, -1, None

    for name, clf in candidates.items():
        print(f"\n=== {name} ===")
        pipe = ImbPipeline(steps=[("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])
        prfs = []
        aucs = []
        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xva)
            yprob = pipe.predict_proba(Xva)[:,1] if hasattr(pipe, "predict_proba") else None
            p, r, f, _ = precision_recall_fscore_support(yva, ypred, average="binary", zero_division=0)
            auc = roc_auc_score(yva, yprob) if yprob is not None else np.nan
            print(f"Fold {fold}: P={p:.3f} R={r:.3f} F1={f:.3f} AUC={auc:.3f}")
            prfs.append((p, r, f)); aucs.append(auc)
        p = np.mean([x[0] for x in prfs]); r = np.mean([x[1] for x in prfs]); f = np.mean([x[2] for x in prfs])
        a = np.nanmean(aucs)
        print(f"CV mean: P={p:.3f} R={r:.3f} F1={f:.3f} AUC={a:.3f}")

        if f > best_f1:
            best_f1, best_name, best_model = f, name, pipe

    print(f"\n[+] Best model: {best_name} (F1={best_f1:.3f})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X, y)
    joblib.dump({"model": best_model, "feature_names": num_cols}, model_out)
    print(f"[+] Saved model to: {model_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV built by build_dataset.py")
    ap.add_argument("--model_out", required=True, help="Path to save joblib model")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    train_and_eval(df, args.model_out)
