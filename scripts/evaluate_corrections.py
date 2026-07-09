"""Evaluate the web recognizer against saved user correction records."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import main
from mnist_model import get_device


@dataclass(frozen=True)
class CorrectionCase:
    """One correction record that has enough data for app-level evaluation."""

    filename: str
    image_path: Path
    target: str


def load_cases(
    corrections_path: Path = main.CORRECTIONS_PATH,
    upload_dir: Path = main.CORRECTION_UPLOAD_DIR,
) -> list[CorrectionCase]:
    """Load sequence corrections that can be replayed through the app."""

    if not corrections_path.exists():
        return []
    cases: list[CorrectionCase] = []
    for line in corrections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("correction_kind") != "sequence":
            continue
        image_id = str(record.get("image_id", ""))
        target = str(record.get("corrected_label", ""))
        if not image_id or not target:
            continue
        image_path = upload_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        cases.append(CorrectionCase(filename=str(record.get("filename", image_path.name)), image_path=image_path, target=target))
    return cases


def load_web_models() -> tuple[object, object]:
    """Initialize the same model stack used by main.run()."""

    device = get_device()
    main.MnistWebHandler.device = device
    main.MnistWebHandler.model, main.MnistWebHandler.labels = main.load_character_model(device=device)
    main.MnistWebHandler.letter_model, main.MnistWebHandler.letter_labels = main.load_letter_model(device=device)
    main.MnistWebHandler.alnum_model, main.MnistWebHandler.alnum_labels = main.load_mixedcase_model(device=device)
    if main.MnistWebHandler.alnum_model is None:
        main.MnistWebHandler.alnum_model, main.MnistWebHandler.alnum_labels = main.load_alnum_model(device=device)
    main.MnistWebHandler.recognizer_kind = "characters" if main.MnistWebHandler.labels is not None else "digits"
    return main.MnistWebHandler.model, device


def evaluate_cases(cases: list[CorrectionCase]) -> dict[str, object]:
    """Run saved correction images through the web classifier."""

    if not cases:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "results": []}
    model, device = load_web_models()
    results = []
    correct = 0
    for case in cases:
        classified = main.classify_files([(case.filename, case.image_path.read_bytes())], model, device)[0]
        prediction = str(classified.get("sequence", ""))
        is_correct = prediction == case.target
        correct += int(is_correct)
        results.append(
            {
                "filename": case.filename,
                "target": case.target,
                "prediction": prediction,
                "correct": is_correct,
            }
        )
    return {"total": len(cases), "correct": correct, "accuracy": 100.0 * correct / len(cases), "results": results}


def main_cli() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Evaluate saved correction uploads against the web recognizer.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    report = evaluate_cases(load_cases())
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"correction_accuracy={report['accuracy']:.2f}% ({report['correct']}/{report['total']})")
    for result in report["results"]:
        status = "ok" if result["correct"] else "miss"
        print(f"{status}: {result['filename']} target={result['target']!r} prediction={result['prediction']!r}")


if __name__ == "__main__":
    main_cli()
