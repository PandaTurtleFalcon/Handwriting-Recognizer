"""Report character-model confusion patterns, especially punctuation misses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from character_model import (  # noqa: E402
    DATASET_ROOT,
    METRICS_PATH,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
)
from mnist_model import get_device  # noqa: E402


def _group(label: str) -> str:
    """Return the broad character group for a label."""

    if label.isdigit():
        return "digits"
    if label.isalpha():
        return "letters"
    return "punctuation"


def _metric_extra_roots() -> list[Path]:
    """Return extra roots used by the saved character metrics, when available."""

    if not METRICS_PATH.exists():
        return []
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return [Path(path) for path in metrics.get("extra_roots", []) if Path(path).exists()]


def analyze_confusions(
    batch_size: int = 256,
    top: int = 25,
    extra_roots: list[Path] | None = None,
) -> dict[str, object]:
    """Evaluate the deployed character model on its validation split."""

    device = get_device()
    model, labels = load_character_model(device=device)
    if model is None or labels is None:
        raise RuntimeError("character_cnn.pt is missing or could not be loaded.")

    selected_extra_roots = _metric_extra_roots() if extra_roots is None else extra_roots
    images, targets, cache_labels = build_or_load_combined_cache(DATASET_ROOT, selected_extra_roots)
    if list(cache_labels) != list(labels):
        raise RuntimeError("Character cache labels do not match deployed checkpoint labels.")

    indices = list(range(len(targets)))
    _, validation_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=42,
        stratify=targets.numpy(),
    )
    validation_index_tensor = torch.tensor(validation_indices, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(images[validation_index_tensor], targets[validation_index_tensor]),
        batch_size=batch_size,
        shuffle=False,
    )

    total = 0
    exact = 0
    ambiguity = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()
    group_total: Counter[str] = Counter()
    group_correct: Counter[str] = Counter()
    group_ambiguity: Counter[str] = Counter()
    group_confusions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    per_label_total: Counter[str] = Counter()
    per_label_correct: Counter[str] = Counter()

    with torch.no_grad():
        for batch_images, batch_targets in loader:
            outputs = model(batch_images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            for expected_index, predicted_index in zip(batch_targets.tolist(), predictions.tolist()):
                expected = str(labels[int(expected_index)])
                predicted = str(labels[int(predicted_index)])
                group = _group(expected)
                is_exact = expected == predicted
                is_ambiguity = labels_match_with_ambiguity(expected, predicted)

                total += 1
                exact += int(is_exact)
                ambiguity += int(is_ambiguity)
                group_total[group] += 1
                group_correct[group] += int(is_exact)
                group_ambiguity[group] += int(is_ambiguity)
                per_label_total[expected] += 1
                per_label_correct[expected] += int(is_exact)
                if not is_exact:
                    confusion_counts[(expected, predicted)] += 1
                    group_confusions[group][(expected, predicted)] += 1

    worst_labels = []
    for label in labels:
        label_total = per_label_total[str(label)]
        if not label_total:
            continue
        correct = per_label_correct[str(label)]
        worst_labels.append(
            {
                "label": str(label),
                "group": _group(str(label)),
                "accuracy": 100.0 * correct / label_total,
                "correct": correct,
                "total": label_total,
            }
        )
    worst_labels.sort(key=lambda item: (float(item["accuracy"]), str(item["label"])))

    return {
        "total": total,
        "exact_accuracy": 100.0 * exact / max(total, 1),
        "ambiguity_aware_accuracy": 100.0 * ambiguity / max(total, 1),
        "group_accuracy": {
            group: 100.0 * group_correct[group] / max(group_total[group], 1)
            for group in ("digits", "letters", "punctuation")
        },
        "group_ambiguity_accuracy": {
            group: 100.0 * group_ambiguity[group] / max(group_total[group], 1)
            for group in ("digits", "letters", "punctuation")
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
            for group in ("digits", "letters", "punctuation")
        },
        "worst_labels": worst_labels[:top],
        "extra_roots": [str(path) for path in selected_extra_roots],
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Analyze character-model validation confusions.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-only", action="store_true", help="Ignore extra roots from character metrics.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = analyze_confusions(
        batch_size=args.batch_size,
        top=args.top,
        extra_roots=[] if args.base_only else None,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(
        "character "
        f"exact={report['exact_accuracy']:.2f}% "
        f"ambiguity={report['ambiguity_aware_accuracy']:.2f}%"
    )
    print("groups:")
    for group, accuracy in report["group_accuracy"].items():
        ambiguity = report["group_ambiguity_accuracy"][group]
        print(f"  {group}: exact={accuracy:.2f}% ambiguity={ambiguity:.2f}%")
    for group in ("digits", "letters", "punctuation"):
        print(f"top {group} confusions:")
        for item in report["top_confusions_by_group"][group]:
            print(f"  {item['expected']} -> {item['predicted']}: {item['count']}")
    print("worst labels:")
    for item in report["worst_labels"][: args.top]:
        print(f"  {item['label']}: {item['accuracy']:.2f}% ({item['correct']}/{item['total']})")


if __name__ == "__main__":
    main()
