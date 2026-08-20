"""Report character-model confusion patterns, especially punctuation misses."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

METRICS_PATH = PROJECT_DIR / "character_training_metrics.json"
DATASET_ROOT = None
build_or_load_combined_cache = None
get_device = None
load_character_model = None
train_test_split = None
match_with_ambiguity = None

DEFAULT_VISUAL_FAMILIES = ("!/1Iil|", "0Oo", "5Ss", "Cc", "Uuv", "Pp", "2Zz")
CONFIDENCE_BANDS = ((0.7, "<0.70"), (0.9, "0.70-0.90"), (float("inf"), ">=0.90"))
MARGIN_BANDS = ((0.05, "<0.05"), (0.15, "0.05-0.15"), (0.35, "0.15-0.35"), (float("inf"), ">=0.35"))


def labels_match_with_ambiguity(expected: str, predicted: str) -> bool:
    """Return true when labels are exact or known visual twins."""

    global match_with_ambiguity

    if match_with_ambiguity is None:
        from character_model import labels_match_with_ambiguity as character_labels_match_with_ambiguity

        match_with_ambiguity = character_labels_match_with_ambiguity
    return bool(match_with_ambiguity(expected, predicted))


def _group(label: str) -> str:
    """Return the broad character group for a label."""

    if label.isdigit():
        return "digits"
    if label.isalpha():
        return "letters"
    return "punctuation"


def _band_name(value: float, bands: Iterable[tuple[float, str]]) -> str:
    """Return the first named bucket whose upper bound contains value."""

    for upper_bound, label in bands:
        if value < upper_bound:
            return label
    return "unknown"


def parse_families(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated visual-family labels for detailed diagnostics."""

    if value is None:
        return DEFAULT_VISUAL_FAMILIES
    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or DEFAULT_VISUAL_FAMILIES


def _metric_extra_roots() -> list[Path]:
    """Return extra roots used by the saved character metrics, when available."""

    if not METRICS_PATH.exists():
        return []
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return [Path(path) for path in metrics.get("extra_roots", []) if Path(path).exists()]


def family_error_details(
    expected_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    margins: list[float],
    families: tuple[str, ...] = DEFAULT_VISUAL_FAMILIES,
    top: int = 12,
) -> list[dict[str, object]]:
    """Summarize exact misses whose expected/predicted labels share a family."""

    details: list[dict[str, object]] = []
    for family in families:
        members = set(dict.fromkeys(family))
        pair_counts: Counter[tuple[str, str]] = Counter()
        split_counts: Counter[str] = Counter()
        confidence_counts: Counter[str] = Counter()
        margin_counts: Counter[str] = Counter()
        confidence_total = 0.0
        margin_total = 0.0
        for expected, predicted, confidence, margin in zip(
            expected_labels,
            predicted_labels,
            confidences,
            margins,
        ):
            if expected == predicted or expected not in members or predicted not in members:
                continue
            pair_counts[(expected, predicted)] += 1
            split_counts[_group(expected)] += 1
            confidence_counts[_band_name(float(confidence), CONFIDENCE_BANDS)] += 1
            margin_counts[_band_name(float(margin), MARGIN_BANDS)] += 1
            confidence_total += float(confidence)
            margin_total += float(margin)
        recoverable = sum(pair_counts.values())
        if not recoverable:
            details.append(
                {
                    "family": family,
                    "recoverable_errors": 0,
                    "split_recoverable_errors": {},
                    "confidence_bands": {},
                    "margin_bands": {},
                    "mean_confidence": None,
                    "mean_margin": None,
                    "top_pairs": [],
                }
            )
            continue
        details.append(
            {
                "family": family,
                "recoverable_errors": recoverable,
                "split_recoverable_errors": dict(split_counts),
                "confidence_bands": dict(confidence_counts),
                "margin_bands": dict(margin_counts),
                "mean_confidence": confidence_total / recoverable,
                "mean_margin": margin_total / recoverable,
                "top_pairs": [
                    {"expected": expected, "predicted": predicted, "count": count}
                    for (expected, predicted), count in pair_counts.most_common(top)
                ],
            }
        )
    details.sort(key=lambda item: int(item["recoverable_errors"]), reverse=True)
    return details


def analyze_confusions(
    batch_size: int = 256,
    top: int = 25,
    extra_roots: list[Path] | None = None,
    families: tuple[str, ...] = DEFAULT_VISUAL_FAMILIES,
) -> dict[str, object]:
    """Evaluate the deployed character model on its validation split."""

    global DATASET_ROOT
    global build_or_load_combined_cache
    global get_device
    global load_character_model
    global train_test_split

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if train_test_split is None:
        from character_model import stratified_split_indices

        train_test_split = stratified_split_indices
    if DATASET_ROOT is None:
        from character_model import DATASET_ROOT as character_dataset_root

        DATASET_ROOT = character_dataset_root
    if build_or_load_combined_cache is None:
        from character_model import build_or_load_combined_cache as character_cache_builder

        build_or_load_combined_cache = character_cache_builder
    if load_character_model is None:
        from character_model import load_character_model as character_model_loader

        load_character_model = character_model_loader
    if get_device is None:
        from mnist_model import get_device as mnist_device_loader

        get_device = mnist_device_loader

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
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    confidences: list[float] = []
    margins: list[float] = []

    with torch.no_grad():
        for batch_images, batch_targets in loader:
            outputs = model(batch_images.to(device))
            probabilities = outputs.softmax(dim=1).cpu()
            top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
            predictions = outputs.argmax(dim=1).cpu()
            for row, (expected_index, predicted_index) in enumerate(zip(batch_targets.tolist(), predictions.tolist())):
                expected = str(labels[int(expected_index)])
                predicted = str(labels[int(predicted_index)])
                group = _group(expected)
                is_exact = expected == predicted
                is_ambiguity = labels_match_with_ambiguity(expected, predicted)
                confidence = float(top2[row, 0].item())
                margin = float((top2[row, 0] - top2[row, 1]).item()) if top2.shape[1] > 1 else 1.0

                total += 1
                exact += int(is_exact)
                ambiguity += int(is_ambiguity)
                expected_labels.append(expected)
                predicted_labels.append(predicted)
                confidences.append(confidence)
                margins.append(margin)
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
        "family_error_details": family_error_details(
            expected_labels,
            predicted_labels,
            confidences,
            margins,
            families=families,
            top=top,
        ),
        "extra_roots": [str(path) for path in selected_extra_roots],
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Analyze character-model validation confusions.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-only", action="store_true", help="Ignore extra roots from character metrics.")
    parser.add_argument("--families", default=",".join(DEFAULT_VISUAL_FAMILIES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = analyze_confusions(
        batch_size=args.batch_size,
        top=args.top,
        extra_roots=[] if args.base_only else None,
        families=parse_families(args.families),
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
    print("visual-family recoverable errors:")
    for item in report["family_error_details"][: args.top]:
        print(
            f"  {item['family']}: {item['recoverable_errors']} "
            f"mean_conf={item['mean_confidence']} mean_margin={item['mean_margin']}"
        )


if __name__ == "__main__":
    main()
