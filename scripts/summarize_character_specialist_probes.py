"""Summarize character image-specialist probe JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_dict(value: object) -> dict[str, Any]:
    """Return value when it is a dict, otherwise an empty dict."""

    return value if isinstance(value, dict) else {}


def best_threshold_row(rows: list[object]) -> dict[str, Any] | None:
    """Return the threshold row with the best exact gain and replacement balance."""

    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return None
    return max(
        dict_rows,
        key=lambda row: (
            float(row.get("gain", 0.0)),
            int(_as_dict(row.get("replacement_report")).get("fixed", 0))
            - int(_as_dict(row.get("replacement_report")).get("broken", 0)),
        ),
    )


def summarize_probe(path: Path) -> dict[str, object]:
    """Return a compact, stable summary for one specialist probe report."""

    report = json.loads(path.read_text(encoding="utf-8"))
    selection = _as_dict(report.get("threshold_selection"))
    confirmation = _as_dict(report.get("confirmation"))
    best_selection = best_threshold_row(list(selection.get("evaluated_thresholds", [])))
    delta = _as_dict(report.get("delta"))
    diagnostics = [
        _compact_family_diagnostic(row)
        for row in report.get("family_validation_diagnostics", [])
        if isinstance(row, dict)
    ]
    return {
        "path": str(path),
        "promotable": bool(report.get("promotable")),
        "thresholds": report.get("thresholds"),
        "best_selection": best_selection,
        "confirmation_gain": confirmation.get("gain"),
        "confirmation_report": confirmation.get("replacement_report"),
        "confirmation_confirmed": confirmation.get("confirmed"),
        "validation_delta": delta.get("validation_accuracy", 0.0),
        "letter_delta": delta.get("letter_validation_accuracy", 0.0),
        "digit_delta": delta.get("digit_validation_accuracy", 0.0),
        "punctuation_delta": delta.get("punctuation_validation_accuracy", 0.0),
        "family_reports": report.get("family_reports"),
        "family_validation_diagnostics": diagnostics,
    }


def _compact_family_diagnostic(row: dict[str, Any]) -> dict[str, object]:
    """Return the decision fields needed to compare one family diagnostic."""

    replacement_report = _as_dict(row.get("replacement_report"))
    return {
        "family": row.get("family"),
        "validation_delta": _as_dict(row.get("delta")).get("validation_accuracy", 0.0),
        "replacement_report": replacement_report,
        "family_reports": row.get("family_reports"),
        "protected_failures": row.get("protected_failures", []),
    }


def summarize_probes(paths: list[Path]) -> dict[str, object]:
    """Return compact summaries plus aggregate promotable counts."""

    summaries = [summarize_probe(path) for path in paths]
    return {
        "probe_count": len(summaries),
        "promotable_count": sum(1 for item in summaries if bool(item["promotable"])),
        "confirmed_count": sum(1 for item in summaries if bool(item["confirmation_confirmed"])),
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
