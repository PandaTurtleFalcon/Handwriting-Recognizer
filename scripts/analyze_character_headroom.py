"""Estimate where 93-class character recognition can still gain accuracy."""

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

from character_model import (  # noqa: E402
    AMBIGUITY_GROUPS,
    DATASET_ROOT,
    build_or_load_combined_cache,
    labels_match_with_ambiguity,
    load_character_model,
    stratified_split_indices,
)
from mnist_model import get_device  # noqa: E402
from scripts.analyze_character_confusions import _metric_extra_roots  # noqa: E402


def _group(label: str) -> str:
    """Return the broad character split for one label."""

    if label.isdigit():
        return "digit"
    if label.isalpha():
        return "letter"
    return "punctuation"


def _family_name(group: frozenset[str]) -> str:
    """Return a stable display name for a visual-twin family."""

    return "".join(sorted(group))


def _family_lookup(groups: list[frozenset[str]]) -> dict[tuple[str, str], str]:
    """Map ordered visual-twin pairs to their family name."""

    lookup: dict[tuple[str, str], str] = {}
    for group in groups:
        name = _family_name(group)
        for expected in group:
            for predicted in group:
                if expected != predicted:
                    lookup[(expected, predicted)] = name
    return lookup


def _empty_split_counts() -> dict[str, int]:
    """Return empty recoverable counts for character benchmark splits."""

    return {"digit": 0, "letter": 0, "punctuation": 0}


def headroom_report(
    expected_labels: list[str],
    predicted_labels: list[str],
    families: list[frozenset[str]] = AMBIGUITY_GROUPS,
    target_accuracy: float = 95.0,
) -> dict[str, object]:
    """Return exact and visual-family-recoverable character error budgets."""

    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Expected and predicted label lists must have the same length.")

    family_lookup = _family_lookup(families)
    exact = 0
    ambiguity = 0
    family_recoverable: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_split_recoverable: dict[str, dict[str, int]] = {}
    split_total: Counter[str] = Counter()
    split_exact: Counter[str] = Counter()
    split_ambiguity: Counter[str] = Counter()
    split_recoverable: Counter[str] = Counter()

    for expected, predicted in zip(expected_labels, predicted_labels):
        split = _group(expected)
        split_total[split] += 1
        is_exact = expected == predicted
        is_ambiguous_match = labels_match_with_ambiguity(expected, predicted)
        family_name = family_lookup.get((expected, predicted))
        exact += int(is_exact)
        ambiguity += int(is_ambiguous_match)
        split_exact[split] += int(is_exact)
        split_ambiguity[split] += int(is_ambiguous_match)
        if family_name is not None and not is_exact:
            family_recoverable[family_name] += 1
            family_total[family_name] += 1
            family_split_recoverable.setdefault(family_name, _empty_split_counts())[split] += 1
            split_recoverable[split] += 1

    total = max(len(expected_labels), 1)

    def percent(count: int) -> float:
        return 100.0 * count / total

    splits: dict[str, dict[str, float | int]] = {}
    for split in ("digit", "letter", "punctuation"):
        split_count = max(split_total[split], 1)
        splits[split] = {
            "exact_accuracy": 100.0 * split_exact[split] / split_count,
            "ambiguity_aware_accuracy": 100.0 * split_ambiguity[split] / split_count,
            "recoverable_errors": split_recoverable[split],
            "total": split_total[split],
        }

    family_rows = []
    cumulative_correct = exact
    cumulative_rows = []
    for family, count in family_recoverable.most_common():
        split_recoverable = family_split_recoverable.get(family, _empty_split_counts())
        cumulative_correct += count
        cumulative_accuracy = percent(cumulative_correct)
        row = {
            "family": family,
            "recoverable_errors": count,
            "total_family_errors": family_total[family],
            "error_percent": 100.0 * count / max(len(expected_labels) - exact, 1),
            "accuracy_gain": percent(count),
            "split_recoverable_errors": split_recoverable,
        }
        family_rows.append(row)
        cumulative_rows.append(
            {
                "families": [str(item["family"]) for item in family_rows],
                "family": family,
                "cumulative_recoverable_errors": sum(int(item["recoverable_errors"]) for item in family_rows),
                "cumulative_accuracy": cumulative_accuracy,
                "reaches_target": cumulative_accuracy >= target_accuracy,
                "reaches_95": cumulative_accuracy >= 95.0,
            }
        )

    families_to_target = next((row for row in cumulative_rows if bool(row["reaches_target"])), None)
    families_to_95 = next((row for row in cumulative_rows if bool(row["reaches_95"])), None)
    exact_accuracy = percent(exact)
    visual_oracle_accuracy = percent(ambiguity)

    return {
        "total": len(expected_labels),
        "target_accuracy": target_accuracy,
        "exact_accuracy": exact_accuracy,
        "ambiguity_aware_accuracy": visual_oracle_accuracy,
        "visual_oracle_accuracy": visual_oracle_accuracy,
        "accuracy_gap_to_target": max(0.0, target_accuracy - exact_accuracy),
        "visual_oracle_gap_to_target": max(0.0, target_accuracy - visual_oracle_accuracy),
        "visual_recoverable_errors": ambiguity - exact,
        "remaining_non_family_errors": len(expected_labels) - ambiguity,
        "splits": splits,
        "families": family_rows,
        "cumulative_family_oracle": cumulative_rows,
        "families_to_reach_target": families_to_target,
        "families_to_reach_95": families_to_95,
    }


def deployed_predictions(batch_size: int) -> tuple[list[str], list[str]]:
    """Run the deployed character stack on the validation tensors."""

    device = get_device()
    model, labels = load_character_model(device=device)
    if model is None or labels is None:
        raise RuntimeError("character_cnn.pt is missing or could not be loaded.")

    images, targets, cache_labels = build_or_load_combined_cache(DATASET_ROOT, _metric_extra_roots())
    if list(cache_labels) != list(labels):
        raise RuntimeError("Character cache labels do not match deployed checkpoint labels.")

    indices = list(range(len(targets)))
    _, validation_indices = stratified_split_indices(
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
    expected_labels: list[str] = []
    predicted_labels: list[str] = []
    with torch.no_grad():
        for images_batch, targets_batch in loader:
            outputs = model(images_batch.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            expected_labels.extend(labels[int(index)] for index in targets_batch.tolist())
            predicted_labels.extend(labels[int(index)] for index in predictions.tolist())
    return expected_labels, predicted_labels


def analyze_deployed_headroom(batch_size: int = 4096) -> dict[str, object]:
    """Return headroom metrics for the deployed character stack."""

    expected, predicted = deployed_predictions(batch_size)
    return headroom_report(expected, predicted)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Analyze character exact-accuracy headroom.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--target", type=float, default=95.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    expected, predicted = deployed_predictions(args.batch_size)
    report = headroom_report(expected, predicted, target_accuracy=args.target)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(
        "character headroom "
        f"exact={report['exact_accuracy']:.2f}% "
        f"ambiguity={report['ambiguity_aware_accuracy']:.2f}%"
    )
    print(f"recoverable_family_errors={report['visual_recoverable_errors']}")
    print(f"remaining_non_family_errors={report['remaining_non_family_errors']}")
    if report["families_to_reach_target"] is not None:
        threshold = report["families_to_reach_target"]
        print(
            "families_to_reach_target="
            f"{','.join(threshold['families'])} "
            f"cumulative_accuracy={threshold['cumulative_accuracy']:.2f}%"
        )
    for row in report["families"][:10]:
        print(f"  {row['family']}: {row['recoverable_errors']} recoverable errors")


if __name__ == "__main__":
    main()
