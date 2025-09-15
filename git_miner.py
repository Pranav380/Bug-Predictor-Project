\
from __future__ import annotations
from typing import Dict, Set
from collections import defaultdict
from datetime import datetime, timedelta
import re

from pydriller import RepositoryMining

BUGFIX_KEYWORDS = [
    r"\bfix(es|ed)?\b",
    r"\bbug(s)?\b",
    r"\bhotfix\b",
    r"\bdefect(s)?\b",
    r"\bissue(s)?\b",
    r"\bresolve(d|s)?\b",
    r"\bpatch\b",
]

def _is_bugfix_message(msg: str) -> bool:
    if not msg:
        return False
    m = msg.lower()
    return any(re.search(p, m) for p in BUGFIX_KEYWORDS)

def mine_git_history(repo_path: str, since_months: int = 12) -> Dict[str, Dict]:
    """Aggregate process metrics and a weak bug label per file.

    Returns a dict keyed by file path with:
      - commits, churn_added, churn_deleted, distinct_authors, last_modified_days
      - buggy_label = 1 if file appears in a "bug-fix" commit (heuristic)
    """
    since_dt = datetime.now() - timedelta(days=since_months*30)
    per_file = defaultdict(lambda: {
        "file": "",
        "commits": 0,
        "churn_added": 0,
        "churn_deleted": 0,
        "distinct_authors": 0,
        "last_modified_days": 10_000,
        "buggy_label": 0,
    })
    authors_per_file: Dict[str, Set[str]] = defaultdict(set)
    last_modified: Dict[str, datetime] = {}

    for commit in RepositoryMining(repo_path, since=since_dt).traverse_commits():
        is_bugfix = _is_bugfix_message(commit.msg or "")
        for mod in commit.modifications:
            path = mod.new_path or mod.old_path
            if not path:
                continue
            rec = per_file[path]
            rec["file"] = path
            rec["commits"] += 1
            rec["churn_added"] += max(0, mod.added)
            rec["churn_deleted"] += max(0, mod.removed)
            author_id = (commit.author.email or commit.author.name or "unknown").lower()
            authors_per_file[path].add(author_id)
            lm_prev = last_modified.get(path)
            last_modified[path] = max(lm_prev, commit.author_date) if lm_prev else commit.author_date
            if is_bugfix:
                rec["buggy_label"] = 1

    now = datetime.now()
    for path, rec in per_file.items():
        rec["distinct_authors"] = len(authors_per_file[path])
        lm = last_modified.get(path)
        rec["last_modified_days"] = (now - lm).days if lm else 10_000

    return dict(per_file)
