"""Estimate where mixed-case recognition can still gain accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_AMBIGUITY_GROUPS,
    MIXEDCASE_LABELS,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_mixedcase_model,
)
from mnist_model import get_device  # noqa: E402


def _family_name(group: frozenset[str]) -> str:
    """Return a stable label for one visual-twin family."""

    return "".join(sorted(group))


def _family_lookup(groups: list[frozenset[str]]) -> dict[tuple[str, str], str]:
    """Map ordered label pairs to their visual-family name."""

    lookup: dict[tuple[str, str], str] = {}
    for group in groups:
        name = _family_name(group)
        for expected in group:
            for predicted in group:
                if expected != predicted:
                    lookup[(expected, predicted)] = name
    return lookup


def _group(label: str) -> str:
    """Return the broad mixed-case split for one label."""

    if label.isdigit():
        return "digit"
    if label.isupper():
        return "upper"
    return "lower"


def headroom_report(
    expected_labels: list[str],
    predicted_labels: list[str],
    families: list[frozenset[str]] = MIXEDCASE_AMBIGUITY_GROUPS,
) -> dict[str, object]:
    """Return exact and oracle-recoverable mixed-case error budgets."""

    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Expected and predicted label lists must have the same length.")
    family_lookup = _family_lookup(families)
    total = max(len(expected_labels), 1)
    exact = 0
    case_oracle = 0
    visual_oracle = 0
    case_or_visual_oracle = 0
    family_recoverable: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    split_total: Counter[str] = Counter()
    split_exact: Counter[str] = Counter()
    split_case_or_visual: Counter[str] = Counter()

    for expected, predicted in zip(expected_labels, predicted_labels):
        expected_split = _group(expected)
        split_total[expected_split] += 1
        is_exact = expected == predicted
        is_case_match = (
            is_exact
            or (expected.isalpha() and predicted.isalpha() and expected.lower() == predicted.lower())
        )
        family_name = family_lookup.get((expected, predicted))
        is_visual_match = is_exact or family_name is not None
        is_case_or_visual_match = is_case_match or is_visual_match
        exact += int(is_exact)
        case_oracle += int(is_case_match)
        visual_oracle += int(is_visual_match)
        case_or_visual_oracle += int(is_case_or_visual_match)
        split_exact[expected_split] += int(is_exact)
        split_case_or_visual[expected_split] += int(is_case_or_visual_match)
        if family_name is not None and not is_exact:
            family_recoverable[family_name] += 1
            family_total[family_name] += 1

    def percent(count: int) -> float:
        return 100.0 * count / total

    split_rows = {}
    for split in ("digit", "upper", "lower"):
        split_count = max(split_total[split], 1)
        split_rows[split] = {
            "exact_accuracy": 100.0 * split_exact[split] / split_count,
            "case_or_visual_oracle_accuracy": 100.0 * split_case_or_visual[split] / split_count,
            "recoverable_errors": split_case_or_visual[split] - split_exact[split],
            "total": split_total[split],
        }

    return {
        "total": len(expected_labels),
        "exact_accuracy": percent(exact),
        "case_oracle_accuracy": percent(case_oracle),
        "visual_oracle_accuracy": percent(visual_oracle),
        "case_or_visual_oracle_accuracy": percent(case_or_visual_oracle),
        "case_or_visual_recoverable_errors": case_or_visual_oracle - exact,
        "remaining_non_family_errors": len(expected_labels) - case_or_visual_oracle,
        "splits": split_rows,
        "families": [
            {
                "family": family,
                "recoverable_errors": count,
                "total_family_errors": family_total[family],
                "error_percent": 100.0 * count / max(len(expected_labels) - exact, 1),
            }
            for family, count in family_recoverable.most_common()
        ],
    }


def deployed_predictions(batch_size: int) -> tuple[list[str], list[str]]:
    """Run the deployed mixed-case stack on the mixed-case benchmark tensors."""

    device = get_device()
    model, labels = load_mixedcase_model(device=device)
    if model is None or labels is None:
        raise RuntimeError("mixedcase_cnn.pt is missing or could not be loaded.")
    if list(labels) != list(MIXEDCASE_LABELS):
        raise RuntimeError("Mixed-case checkpoint labels do not match expected order.")
    mnist_images, mnist_targets = build_or_load_mnist_cache(train=False)
    byclass_images, byclass_targets = build_or_load_emnist_byclass_mixedcase_cache(train=False)
    loader = DataLoader(
        TensorDataset(
            torch.cat([mnist_images, byclass_images]),
            torch.cat([mnist_targets, byclass_targets]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            expected_labels.extend(labels[int(index)] for index in targets.tolist())
            predicted_labels.extend(labels[int(index)] for index in predictions.tolist())
    return expected_labels, predicted_labels


def analyze_deployed_headroom(batch_size: int = 4096) -> dict[str, object]:
    """Return headroom metrics for the deployed mixed-case stack."""

    expected, predicted = deployed_predictions(batch_size)
    return headroom_report(expected, predicted)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Analyze mixed-case exact-accuracy headroom.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze_deployed_headroom(batch_size=args.batch_size)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        "mixedcase headroom "
        f"exact={report['exact_accuracy']:.2f}% "
        f"case_oracle={report['case_oracle_accuracy']:.2f}% "
        f"visual_oracle={report['visual_oracle_accuracy']:.2f}% "
        f"case_or_visual_oracle={report['case_or_visual_oracle_accuracy']:.2f}%"
    )
    print(f"recoverable_family_errors={report['case_or_visual_recoverable_errors']}")
    print(f"remaining_non_family_errors={report['remaining_non_family_errors']}")
    for row in report["families"][:10]:
        print(f"  {row['family']}: {row['recoverable_errors']} recoverable errors")


if __name__ == "__main__":
    main()
