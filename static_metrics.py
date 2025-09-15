from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from radon.raw import analyze as raw_analyze
from radon.complexity import cc_visit
from radon.metrics import mi_visit

TEXT_EXTS = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".cs", ".rb", ".php"}

def is_text_code_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS

def read_file_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def compute_static_metrics_for_file(path: Path) -> Dict:
    text = read_file_text(path)
    if not text:
        return {
            "file": str(path),
            "loc": 0,
            "sloc": 0,
            "comments": 0,
            "multi": 0,
            "blank": 0,
            "avg_cc": 0.0,
            "max_cc": 0.0,
            "mi": 0.0,
        }
    raw = raw_analyze(text)
    ccs = cc_visit(text)
    avg_cc = sum([c.complexity for c in ccs]) / len(ccs) if ccs else 0.0
    max_cc = max([c.complexity for c in ccs]) if ccs else 0.0
    mi_scores = mi_visit(text, multi=True)
    mi = float(sum(mi_scores) / len(mi_scores)) if mi_scores else 0.0
    return {
        "file": str(path),
        "loc": raw.loc,
        "sloc": raw.sloc,
        "comments": raw.comments,
        "multi": raw.multi,
        "blank": raw.blank,
        "avg_cc": avg_cc,
        "max_cc": max_cc,
        "mi": mi,
    }

def crawl_repo_static_metrics(repo_path: str) -> List[Dict]:
    repo = Path(repo_path)
    results: List[Dict] = []
    for p in repo.rglob("*"):
        if p.is_file() and is_text_code_file(p):
            results.append(compute_static_metrics_for_file(p))
    return results
