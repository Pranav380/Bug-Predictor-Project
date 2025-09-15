from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

from utils.static_metrics import crawl_repo_static_metrics
from miners.git_miner import mine_git_history
from utils.io_utils import save_csv

def build_dataset(repo: str, since_months: int, out_csv: str):
    print(f"[+] Mining Git history from: {repo} (since {since_months} months)")
    git_data = mine_git_history(repo, since_months=since_months)
    git_df = pd.DataFrame(list(git_data.values()))
    if git_df.empty:
        raise SystemExit("No git data mined. Check repo path or since_months.")
    print(f"[+] Git rows: {len(git_df)}")

    print(f"[+] Computing static metrics...")
    sm = crawl_repo_static_metrics(repo)
    sm_df = pd.DataFrame(sm)
    print(f"[+] Static rows: {len(sm_df)}")

    # Join on file path directly
    df = pd.merge(sm_df, git_df, on="file", how="inner")

    if df.empty:
        # Fallback: join on basename if paths differ
        sm_df["__base"] = sm_df["file"].apply(lambda p: Path(p).name)
        git_df["__base"] = git_df["file"].apply(lambda p: Path(p).name)
        df = pd.merge(sm_df, git_df, on="__base", how="inner", suffixes=("_static", "_git"))
        if not df.empty:
            df["file"] = df["file_static"]
            keep = ["file","loc","sloc","comments","multi","blank","avg_cc","max_cc","mi",
                    "commits","churn_added","churn_deleted","distinct_authors","last_modified_days","buggy_label"]
            df = df.assign(**{k: df[k] for k in keep})[keep]

    if df.empty:
        raise SystemExit("Join produced empty dataframe. Check path normalization.")

    df = df.fillna(0)
    df = df[df["loc"] > 0].reset_index(drop=True)

    save_csv(df, out_csv)
    print(f"[+] Saved dataset to: {out_csv}  (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to local git repo")
    ap.add_argument("--since_months", type=int, default=12, help="History window size")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()
    build_dataset(args.repo, args.since_months, args.out)
