"""Summarize mixed-case probe JSON reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _as_dict(value: object) -> dict[str, Any]:
    """Return value when it is a dict, otherwise an empty dict."""

    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    """Return value when it is a list, otherwise an empty list."""

    return value if isinstance(value, list) else []


def _metric_delta(after: object, before: object, key: str) -> float | None:
    """Return after-minus-before for one metric when both values exist."""

    after_metrics = _as_dict(after)
    before_metrics = _as_dict(before)
    if key not in after_metrics or key not in before_metrics:
        return None
    return float(after_metrics[key]) - float(before_metrics[key])


def _family_delta(row: dict[str, Any]) -> float:
    """Return the final-test exact delta for one family row."""

    if isinstance(row.get("delta"), (int, float)):
        return float(row["delta"])
    return _metric_delta(row.get("after_metrics"), row.get("before_metrics"), "test_accuracy") or 0.0


def _compact_family_row(row: dict[str, Any]) -> dict[str, object]:
    """Return the decision fields needed to compare one family row."""

    return {
        "family": row.get("family"),
        "accepted": bool(row.get("accepted")),
        "selection_delta": row.get("selection_delta"),
        "confirmation_delta": row.get("confirmation_delta"),
        "delta": row.get("delta"),
        "rejection_reason": row.get("rejection_reason"),
        "before_metrics": row.get("before_metrics"),
        "after_metrics": row.get("after_metrics"),
    }


def top_family_rows(families: object, limit: int = 5) -> list[dict[str, object]]:
    """Return family rows with the largest final exact deltas."""

    rows = [_as_dict(row) for row in _as_list(families)]
    ranked = sorted(rows, key=_family_delta, reverse=True)
    return [_compact_family_row(row) for row in ranked[:limit]]


def _all_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every family decision row from every sweep row."""

    family_rows: list[dict[str, Any]] = []
    for row in rows:
        family_rows.extend(_as_dict(family) for family in _as_list(row.get("families")))
    return family_rows


def _rejection_reason_counts(families: list[dict[str, Any]]) -> dict[str, int]:
    """Count rejection reasons across family decision rows."""

    reasons = Counter(
        str(row.get("rejection_reason"))
        for row in families
        if not bool(row.get("accepted")) and row.get("rejection_reason")
    )
    return dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))


def _accepted_family_counts(families: list[dict[str, Any]]) -> dict[str, int]:
    """Count accepted family rows across a sweep."""

    counts = Counter(str(row.get("family")) for row in families if bool(row.get("accepted")) and row.get("family"))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_sweep(report: dict[str, Any]) -> dict[str, object]:
    """Return the compact fields for a mixed-case sweep report."""

    rows = [_as_dict(row) for row in _as_list(report.get("rows"))]
    best = _as_dict(report.get("best"))
    family_rows = _all_family_rows(rows)
    return {
        "kind": "sweep",
        "completed_runs": report.get("completed_runs", len(rows)),
        "planned_runs": report.get("planned_runs"),
        "promotable_count": report.get("promotable_count", sum(1 for row in rows if row.get("promotable"))),
        "best_promotable": bool(best.get("promotable")),
        "best_test_delta": float(best.get("test_delta", 0.0)) if best else None,
        "families": report.get("families"),
        "best_parameters": best.get("parameters"),
        "best_base": best.get("base"),
        "best_reranked": best.get("reranked"),
        "top_family_rows": top_family_rows(best.get("families")),
        "top_family_rows_all_runs": top_family_rows(family_rows),
        "rejection_reason_counts": _rejection_reason_counts(family_rows),
        "accepted_family_counts": _accepted_family_counts(family_rows),
    }


def summarize_residual(report: dict[str, Any]) -> dict[str, object]:
    """Return the compact fields for a residual-cluster probe report."""

    clusters = [_as_dict(row) for row in _as_list(report.get("clusters"))]
    accepted = [row for row in clusters if bool(row.get("accepted"))]
    return {
        "kind": "residual_clusters",
        "promotable": bool(report.get("promotable")),
        "test_delta": float(report.get("test_delta", 0.0)),
        "accepted_count": len(accepted),
        "accepted_clusters": [row.get("cluster") for row in accepted],
        "base": report.get("base"),
        "reranked": report.get("reranked"),
        "cluster_rejections": [
            {
                "cluster": row.get("cluster"),
                "selection_delta": row.get("selection_delta"),
                "confirmation_delta": row.get("confirmation_delta"),
                "delta": row.get("delta"),
                "reason": row.get("rejection_reason"),
            }
            for row in clusters
            if not bool(row.get("accepted"))
        ],
    }


def summarize_ensemble(report: dict[str, Any]) -> dict[str, object]:
    """Return compact fields for a checkpoint-ensemble probe report."""

    best = _as_dict(report.get("best"))
    baseline = _as_dict(report.get("baseline"))
    candidates = [_as_dict(row) for row in _as_list(report.get("candidates"))]
    accepted = [row for row in candidates if bool(row.get("accepted"))]
    return {
        "kind": "checkpoint_ensemble",
        "candidate_count": report.get("candidate_count", len(candidates)),
        "unique_checkpoint_count": report.get("unique_checkpoint_count"),
        "duplicate_checkpoint_count": report.get("duplicate_checkpoint_count"),
        "accepted_count": len(accepted),
        "best_path": best.get("path"),
        "best_test_delta": _metric_delta(best.get("metrics"), baseline, "test_accuracy"),
        "best_metrics": best.get("metrics"),
    }


def summarize_probe(path: Path) -> dict[str, object]:
    """Return a compact summary for one mixed-case probe JSON report."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError(f"{path} does not contain a JSON object.")
    if "rows" in report:
        summary = summarize_sweep(report)
    elif "clusters" in report:
        summary = summarize_residual(report)
    elif "candidates" in report:
        summary = summarize_ensemble(report)
    else:
        raise RuntimeError(f"{path} is not a recognized mixed-case probe report.")
    return {"path": str(path), **summary}


def summarize_probes(paths: list[Path]) -> dict[str, object]:
    """Return summaries plus aggregate accepted/promotable counts."""

    summaries = [summarize_probe(path) for path in paths]
    return {
        "probe_count": len(summaries),
        "promotable_count": sum(
            1
            for summary in summaries
            if bool(summary.get("promotable")) or bool(summary.get("best_promotable"))
        ),
        "accepted_count": sum(int(summary.get("accepted_count", 0) or 0) for summary in summaries),
        "summaries": summaries,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_probes(args.reports), indent=2))


if __name__ == "__main__":
    main()
