"""Report exact mixed-case confusion patterns for the deployed checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_LABELS,
    build_or_load_emnist_byclass_mixedcase_cache,
    build_or_load_mnist_cache,
    load_mixedcase_model,
    mixedcase_labels_match_with_ambiguity,
    mixedcase_labels_match_with_visual_ambiguity,
)
from mnist_model import get_device  # noqa: E402


def _group(label: str) -> str:
    """Return the broad mixed-case label group."""

    if label.isdigit():
        return "digit"
    if label.isupper():
        return "upper"
    return "lower"


def analyze_confusions(batch_size: int = 2048, top: int = 25) -> dict[str, object]:
    """Evaluate the mixed-case model and summarize confusion counts."""

    device = get_device()
    model, labels = load_mixedcase_model(device=device)
    if model is None or labels is None:
        raise RuntimeError("mixedcase_cnn.pt is missing or could not be loaded.")

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

    total = 0
    exact = 0
    casefold = 0
    visual = 0
    case_or_visual = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()
    per_label_total: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()
    group_total: Counter[str] = Counter()
    group_correct: Counter[str] = Counter()
    group_confusions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            for expected_index, predicted_index in zip(targets.tolist(), predictions.tolist()):
                expected = str(labels[int(expected_index)])
                predicted = str(labels[int(predicted_index)])
                expected_group = _group(expected)
                is_exact = expected == predicted
                is_casefold = is_exact or (
                    expected.isalpha() and predicted.isalpha() and expected.lower() == predicted.lower()
                )
                is_visual = mixedcase_labels_match_with_visual_ambiguity(expected, predicted)
                is_case_or_visual = mixedcase_labels_match_with_ambiguity(expected, predicted)

                total += 1
                exact += int(is_exact)
                casefold += int(is_casefold)
                visual += int(is_visual)
                case_or_visual += int(is_case_or_visual)
                per_label_total[expected] += 1
                group_total[expected_group] += 1
                if is_exact:
                    per_label_correct[expected] += 1
                    group_correct[expected_group] += 1
                else:
                    confusion_counts[(expected, predicted)] += 1
                    group_confusions[expected_group][(expected, predicted)] += 1

    worst_labels = []
    for label in labels:
        label_total = per_label_total[str(label)]
        if not label_total:
            continue
        correct = per_label_correct[str(label)]
        worst_labels.append(
            {
                "label": str(label),
                "accuracy": 100.0 * correct / label_total,
                "correct": correct,
                "total": label_total,
            }
        )
    worst_labels.sort(key=lambda item: (float(item["accuracy"]), str(item["label"])))

    return {
        "total": total,
        "exact_accuracy": 100.0 * exact / max(total, 1),
        "casefold_accuracy": 100.0 * casefold / max(total, 1),
        "visual_ambiguity_accuracy": 100.0 * visual / max(total, 1),
        "case_or_visual_accuracy": 100.0 * case_or_visual / max(total, 1),
        "group_accuracy": {
            group: 100.0 * group_correct[group] / max(group_total[group], 1)
            for group in ("digit", "upper", "lower")
        },
        "top_confusions": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in confusion_counts.most_common(top)
        ],
        "top_confusions_by_group": {
            group: [
                {"expected": expected, "predicted": predicted, "count": count}
                for (expected, predicted), count in group_confusions[group].most_common(top)
            ]
            for group in ("digit", "upper", "lower")
        },
        "worst_labels": worst_labels[:top],
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Analyze mixed-case exact confusions.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = analyze_confusions(batch_size=args.batch_size, top=args.top)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(
        "mixedcase "
        f"exact={report['exact_accuracy']:.2f}% "
        f"casefold={report['casefold_accuracy']:.2f}% "
        f"visual={report['visual_ambiguity_accuracy']:.2f}% "
        f"case_or_visual={report['case_or_visual_accuracy']:.2f}%"
    )
    print("groups:")
    for group, accuracy in report["group_accuracy"].items():
        print(f"  {group}: {accuracy:.2f}%")
    print("top confusions:")
    for item in report["top_confusions"]:
        print(f"  {item['expected']} -> {item['predicted']}: {item['count']}")
    print("worst labels:")
    for item in report["worst_labels"]:
        print(f"  {item['label']}: {item['accuracy']:.2f}% ({item['correct']}/{item['total']})")


if __name__ == "__main__":
    main()
