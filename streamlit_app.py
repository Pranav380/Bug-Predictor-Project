from __future__ import annotations
import argparse, sys, joblib
import pandas as pd
import streamlit as st
from pathlib import Path

def get_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo", required=True, help="Path to local git repo to analyze")
    ap.add_argument("--model", required=True, help="Trained model .joblib")
    ap.add_argument("--since_months", type=int, default=6, help="History window size")
    return ap.parse_known_args()[0]

_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT / "src"))
from miners.git_miner import mine_git_history
from utils.static_metrics import crawl_repo_static_metrics

st.set_page_config(page_title="Bug Predictor", layout="wide")
args = get_args()
st.title("🐞 Bug Predictor Dashboard")
st.caption("Predict bug-prone files by combining static metrics, Git history, and ML.")

payload = joblib.load(args.model)
pipe = payload["model"]
feature_names = payload["feature_names"]

repo_path = args.repo
since_months = args.since_months

with st.spinner(f"Mining repo: {repo_path}"):
    git_data = mine_git_history(repo_path, since_months=since_months)
    git_df = pd.DataFrame(list(git_data.values()))
    sm_df = pd.DataFrame(crawl_repo_static_metrics(repo_path))

df = pd.merge(sm_df, git_df, on="file", how="inner").fillna(0)
if df.empty:
    st.error("No analyzable files found; check repo path and file types.")
    st.stop()

X = df[[c for c in df.columns if c in feature_names]].copy()
pred_proba = pipe.predict_proba(X)[:,1] if hasattr(pipe, "predict_proba") else None
if pred_proba is None:
    st.error("Loaded model does not support probability outputs.")
    st.stop()

df["risk"] = pred_proba
df["pred_label"] = (df["risk"] >= 0.5).astype(int)

st.subheader("Predicted Risk by File")
st.write("Higher risk means more likely to be bug-prone. Sort by risk to prioritize review/tests.")
st.dataframe(df.sort_values("risk", ascending=False)[["file","risk","avg_cc","max_cc","mi","commits","churn_added","churn_deleted","distinct_authors","last_modified_days"]])

st.subheader("Top Risky Files")
topk = st.slider("How many files to show?", 5, 100, 20)
st.table(df.sort_values("risk", ascending=False).head(topk)[["file","risk"]])

try:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        import pandas as pd
        importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
        st.subheader("Feature Importance")
        st.bar_chart(importances.head(20))
except Exception:
    st.info("Feature importance not available for this model/pipeline.")

st.caption("Note: Labels used in training come from commit message heuristics; treat predictions as guidance, not absolute truth.")
