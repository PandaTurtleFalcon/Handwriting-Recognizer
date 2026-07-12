"""Daily fine-tune entrypoint for user-labeled correction data."""

from __future__ import annotations

import argparse
import sys
import json
import shutil
from pathlib import Path
from collections import Counter

from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (
    CORRECTION_UPLOAD_DIR,
    CORRECTIONS_PATH,
    LABELS,
    MIXEDCASE_LABELS,
    _correction_training_items,
    _record_with_legacy_sequence_boxes,
    load_correction_cache,
    train,
    train_mixedcase,
)
from character_model import DATASET_ROOT as CHARACTER_DATASET_ROOT
from character_model import LABELS_PATH as CHARACTER_LABELS_PATH
from character_model import train_character_model
from main import PRACTICE_PRIORITY_LABELS, PRACTICE_TARGET_PER_LABEL


CHARACTER_CORRECTION_ROOT = PROJECT_DIR / "data" / "corrections" / "character_ascii"
HASY_CHARACTER_ROOT = PROJECT_DIR / "data" / "extra_hasyv2" / "character_ascii"
DEFAULT_MIN_CHARACTER_CORRECTIONS = 10
DEFAULT_MIN_ALNUM_CORRECTIONS = 10
DEFAULT_PRIORITY_LABELS = "".join(PRACTICE_PRIORITY_LABELS)
DEFAULT_MIXEDCASE_PRIORITY_LABELS = "".join(PRACTICE_PRIORITY_LABELS)


def export_character_correction_folder(
    labels: list[str],
    output_root: Path = CHARACTER_CORRECTION_ROOT,
    corrections_path: Path = CORRECTIONS_PATH,
    upload_dir: Path = CORRECTION_UPLOAD_DIR,
) -> int:
    """Export saved correction crops as ASCII-code folders for character training."""

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not corrections_path.exists():
        return 0

    label_to_index = {label: index for index, label in enumerate(labels)}
    exported = 0
    for record_index, line in enumerate(corrections_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        image_path = upload_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            source_image = ImageOps.exif_transpose(image).convert("RGB")
            training_record = _record_with_legacy_sequence_boxes(record, source_image)
            for item_index, (corrected_label, bbox) in enumerate(
                _correction_training_items(training_record, label_to_index),
                start=1,
            ):
                try:
                    x0 = max(0, int(round(float(bbox.get("x", 0)))))
                    y0 = max(0, int(round(float(bbox.get("y", 0)))))
                    width = max(1, int(round(float(bbox.get("width", 0)))))
                    height = max(1, int(round(float(bbox.get("height", 0)))))
                except (TypeError, ValueError):
                    continue
                class_dir = output_root / str(ord(corrected_label))
                class_dir.mkdir(parents=True, exist_ok=True)
                crop = source_image.crop((x0, y0, x0 + width, y0 + height)).convert("L")
                crop.save(class_dir / f"{record_index:05d}_{item_index:02d}.png")
                exported += 1
    return exported


def exportable_character_correction_counts(
    labels: list[str],
    corrections_path: Path = CORRECTIONS_PATH,
    upload_dir: Path = CORRECTION_UPLOAD_DIR,
) -> Counter[str]:
    """Count trainable character corrections by label without exporting crops."""

    counts: Counter[str] = Counter()
    if not corrections_path.exists():
        return counts
    label_to_index = {label: index for index, label in enumerate(labels)}
    for line in corrections_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        image_id = str(record.get("image_id", ""))
        if not image_id:
            continue
        image_path = upload_dir / f"{image_id}.png"
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as image:
                source_image = ImageOps.exif_transpose(image).convert("RGB")
                training_record = _record_with_legacy_sequence_boxes(record, source_image)
                for corrected_label, _ in _correction_training_items(training_record, label_to_index):
                    counts[corrected_label] += 1
        except OSError:
            continue
    return counts


def load_character_labels() -> list[str]:
    """Load the deployed 93-class label list for correction export."""

    return [str(label) for label in json.loads(CHARACTER_LABELS_PATH.read_text(encoding="utf-8"))]


def build_parser() -> argparse.ArgumentParser:
    """Create the correction-training CLI parser."""

    parser = argparse.ArgumentParser(description="Fine-tune recognizers from saved user correction data.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report usable correction counts without exporting crops or training.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --dry-run, emit a machine-readable correction readiness report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even when the correction set is smaller than the default safety threshold.",
    )
    parser.add_argument(
        "--min-character-corrections",
        type=int,
        default=DEFAULT_MIN_CHARACTER_CORRECTIONS,
        help="Minimum exported character crops required before character fine-tuning starts.",
    )
    parser.add_argument(
        "--min-alnum-corrections",
        type=int,
        default=DEFAULT_MIN_ALNUM_CORRECTIONS,
        help="Minimum correction items required before folded or mixed-case fine-tuning starts.",
    )
    parser.add_argument(
        "--priority-labels",
        default=DEFAULT_PRIORITY_LABELS,
        help="Character labels to highlight in dry-run correction coverage.",
    )
    parser.add_argument(
        "--mixedcase-priority-labels",
        default=DEFAULT_MIXEDCASE_PRIORITY_LABELS,
        help="Mixed-case labels to highlight in dry-run correction coverage.",
    )
    return parser


def count_exported_character_crops(root: Path = CHARACTER_CORRECTION_ROOT) -> int:
    """Count already-exported correction crop images."""

    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.png") if path.is_file())


def exported_character_crop_counts(root: Path = CHARACTER_CORRECTION_ROOT) -> Counter[str]:
    """Count exported correction crops by character label."""

    counts: Counter[str] = Counter()
    if not root.exists():
        return counts
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = chr(int(class_dir.name))
        counts[label] += sum(1 for path in class_dir.rglob("*.png") if path.is_file())
    return counts


def format_priority_coverage(counts: Counter[str], priority_labels: str) -> str:
    """Return a compact label=count coverage string for weak labels."""

    seen: set[str] = set()
    parts = []
    for label in priority_labels:
        if label in seen:
            continue
        seen.add(label)
        parts.append(f"{label}:{counts.get(label, 0)}")
    return ", ".join(parts)


def filter_priority_labels(priority_labels: str, allowed_labels: list[str]) -> str:
    """Keep priority labels that are valid for a specific recognizer label set."""

    allowed = set(allowed_labels)
    seen: set[str] = set()
    filtered = []
    for label in priority_labels:
        if label in seen or label not in allowed:
            continue
        seen.add(label)
        filtered.append(label)
    return "".join(filtered)


def correction_readiness_summary(
    counts: Counter[str],
    priority_labels: str,
    target_per_label: int = PRACTICE_TARGET_PER_LABEL,
) -> dict[str, int | bool]:
    """Summarize whether weak-label correction coverage is ready for training."""

    unique_labels = list(dict.fromkeys(priority_labels))
    ready_labels = sum(1 for label in unique_labels if counts.get(label, 0) >= target_per_label)
    total_samples = sum(counts.get(label, 0) for label in unique_labels)
    target_total = len(unique_labels) * target_per_label
    coverage_percent = 100.0 * total_samples / target_total if target_total else 0.0
    return {
        "ready": ready_labels == len(unique_labels) and len(unique_labels) > 0,
        "ready_labels": ready_labels,
        "total_labels": len(unique_labels),
        "not_ready_labels": max(0, len(unique_labels) - ready_labels),
        "samples": total_samples,
        "target_samples": target_total,
        "needed_samples": max(0, target_total - total_samples),
        "coverage_percent": coverage_percent,
    }


def next_needed_labels(
    counts: Counter[str],
    priority_labels: str,
    target_per_label: int = PRACTICE_TARGET_PER_LABEL,
    limit: int = 8,
) -> list[dict[str, int | str]]:
    """Return the highest-need labels to collect next."""

    rows = []
    for index, label in enumerate(dict.fromkeys(priority_labels)):
        count = counts.get(label, 0)
        needed = max(0, target_per_label - count)
        if needed <= 0:
            continue
        rows.append({"label": label, "count": count, "needed": needed, "rank": index})
    rows.sort(key=lambda item: (-int(item["needed"]), int(item["count"]), int(item["rank"])))
    return [
        {
            "label": str(item["label"]),
            "count": int(item["count"]),
            "target": target_per_label,
            "needed": int(item["needed"]),
            "coverage_percent": 100.0 * int(item["count"]) / target_per_label if target_per_label else 100.0,
        }
        for item in rows[:limit]
    ]


def correction_recommendation(
    readiness: dict[str, int | bool],
    next_needed: list[dict[str, int | str]],
) -> dict[str, str | None]:
    """Return the next correction-training action for automation."""

    if bool(readiness.get("ready")):
        return {"recommended_action": "train_corrections", "recommended_label": None}
    next_label = str(next_needed[0]["label"]) if next_needed else None
    return {"recommended_action": "collect_corrections", "recommended_label": next_label}


def format_readiness_summary(name: str, summary: dict[str, int | bool]) -> str:
    """Return a compact readiness line for correction dry-runs."""

    status = "ready" if summary["ready"] else "not_ready"
    return (
        f"{name} correction readiness: {status} "
        f"labels={summary['ready_labels']}/{summary['total_labels']} "
        f"not_ready={summary.get('not_ready_labels', 0)} "
        f"samples={summary['samples']}/{summary['target_samples']} "
        f"needed={summary['needed_samples']} "
        f"coverage={float(summary.get('coverage_percent', 0.0)):.2f}%"
    )


def format_recommendation_summary(name: str, report_section: dict[str, object]) -> str:
    """Return a compact next-action line for correction dry-runs."""

    action = str(report_section.get("recommended_action") or "collect_corrections")
    label = report_section.get("recommended_label")
    suffix = f" label={label}" if label is not None else ""
    return f"{name} correction recommendation: action={action}{suffix}"


def format_next_needed_summary(name: str, report_section: dict[str, object]) -> str:
    """Return a compact next-needed label list for correction dry-runs."""

    next_needed = report_section.get("next_needed")
    if not isinstance(next_needed, list) or not next_needed:
        return f"{name} correction next_needed: none"
    labels = []
    for item in next_needed:
        if not isinstance(item, dict):
            continue
        labels.append(f"{item.get('label')}:{item.get('needed')}")
    return f"{name} correction next_needed: {', '.join(labels) if labels else 'none'}"


def dry_run_report(
    character_counts: Counter[str],
    folded_counts: Counter[str],
    mixed_counts: Counter[str],
    folded_item_count: int,
    mixed_item_count: int,
    character_priority_labels: str,
    mixedcase_priority_labels: str,
) -> dict[str, object]:
    """Build the correction dry-run report shared by text and JSON output."""

    folded_priority_labels = filter_priority_labels(character_priority_labels.upper(), LABELS)
    mixed_priority_labels = filter_priority_labels(mixedcase_priority_labels, list(MIXEDCASE_LABELS))
    character_readiness = correction_readiness_summary(character_counts, character_priority_labels)
    character_next_needed = next_needed_labels(character_counts, character_priority_labels)
    character_recommendation = correction_recommendation(character_readiness, character_next_needed)
    folded_readiness = correction_readiness_summary(folded_counts, folded_priority_labels)
    folded_next_needed = next_needed_labels(folded_counts, folded_priority_labels)
    folded_recommendation = correction_recommendation(folded_readiness, folded_next_needed)
    mixed_readiness = correction_readiness_summary(mixed_counts, mixed_priority_labels)
    mixed_next_needed = next_needed_labels(mixed_counts, mixed_priority_labels)
    mixed_recommendation = correction_recommendation(mixed_readiness, mixed_next_needed)
    recommendations = [character_recommendation, folded_recommendation, mixed_recommendation]
    summary_action = (
        "train_corrections"
        if all(item["recommended_action"] == "train_corrections" for item in recommendations)
        else "collect_corrections"
    )
    summary_label = next(
        (item["recommended_label"] for item in recommendations if item["recommended_label"] is not None),
        None,
    )
    summary_batch_labels = [
        str(item["label"])
        for item in character_next_needed
        if isinstance(item, dict) and item.get("label") is not None
    ]
    summary_batch_samples = sum(int(item["count"]) for item in character_next_needed)
    summary_batch_target_samples = sum(int(item["target"]) for item in character_next_needed)
    summary_batch_needed_samples = sum(int(item["needed"]) for item in character_next_needed)
    summary_batch_coverage_percent = (
        100.0 * summary_batch_samples / summary_batch_target_samples if summary_batch_target_samples else 100.0
    )
    summary_needed_samples = int(character_readiness.get("needed_samples", 0))
    summary_not_ready_labels = int(character_readiness.get("not_ready_labels", 0))
    summary_blocked_reason = (
        ""
        if summary_action == "train_corrections"
        else f"Need {summary_needed_samples} more labeled samples across {summary_not_ready_labels} labels before training."
    )
    return {
        "summary": {
            "character_crops": sum(character_counts.values()),
            "folded_items": folded_item_count,
            "mixedcase_items": mixed_item_count,
            "recommended_action": summary_action,
            "recommended_label": summary_label,
            "recommended_batch_labels": summary_batch_labels,
            "recommended_batch_size": len(summary_batch_labels),
            "recommended_batch_samples": summary_batch_samples,
            "recommended_batch_target_samples": summary_batch_target_samples,
            "recommended_batch_needed_samples": summary_batch_needed_samples,
            "recommended_batch_coverage_percent": summary_batch_coverage_percent,
            "training_blocked_reason": summary_blocked_reason,
        },
        "character": {
            "coverage": dict(character_counts),
            "priority_labels": list(dict.fromkeys(character_priority_labels)),
            "readiness": character_readiness,
            "next_needed": character_next_needed,
            **character_recommendation,
        },
        "folded_alnum": {
            "coverage": dict(folded_counts),
            "priority_labels": list(dict.fromkeys(folded_priority_labels)),
            "readiness": folded_readiness,
            "next_needed": folded_next_needed,
            **folded_recommendation,
        },
        "mixedcase": {
            "coverage": dict(mixed_counts),
            "priority_labels": list(dict.fromkeys(mixed_priority_labels)),
            "readiness": mixed_readiness,
            "next_needed": mixed_next_needed,
            **mixed_recommendation,
        },
    }


def print_text_dry_run_report(report: dict[str, object]) -> None:
    """Print the human-readable dry-run report."""

    summary = report["summary"]
    character = report["character"]
    folded = report["folded_alnum"]
    mixed = report["mixedcase"]
    print(
        "Correction summary: "
        f"character_crops={summary['character_crops']}, "
        f"folded_items={summary['folded_items']}, "
        f"mixedcase_items={summary['mixedcase_items']}"
    )
    print(
        "Character priority coverage: "
        f"{format_priority_coverage(Counter(character['coverage']), ''.join(character['priority_labels']))}"
    )
    print(format_readiness_summary("Character", character["readiness"]))
    print(format_recommendation_summary("Character", character))
    print(format_next_needed_summary("Character", character))
    print(
        "Folded alnum priority coverage: "
        f"{format_priority_coverage(Counter(folded['coverage']), ''.join(folded['priority_labels']))}"
    )
    print(format_readiness_summary("Folded alnum", folded["readiness"]))
    print(format_recommendation_summary("Folded alnum", folded))
    print(format_next_needed_summary("Folded alnum", folded))
    print(
        "Mixed-case priority coverage: "
        f"{format_priority_coverage(Counter(mixed['coverage']), ''.join(mixed['priority_labels']))}"
    )
    print(format_readiness_summary("Mixed-case", mixed["readiness"]))
    print(format_recommendation_summary("Mixed-case", mixed))
    print(format_next_needed_summary("Mixed-case", mixed))


def correction_item_label_counts(
    labels: list[str],
    corrections: tuple[object, object] | None,
) -> Counter[str]:
    """Count loaded correction items by their decoded target label."""

    counts: Counter[str] = Counter()
    if corrections is None:
        return counts
    _, targets = corrections
    for target in targets:
        try:
            label = labels[int(target)]
        except (IndexError, TypeError, ValueError):
            continue
        counts[label] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    """Fine-tune alphanumeric models when usable correction crops exist."""

    args = build_parser().parse_args(argv)

    folded_corrections = load_correction_cache(LABELS)
    mixed_corrections = load_correction_cache(list(MIXEDCASE_LABELS))
    character_labels = load_character_labels() if CHARACTER_LABELS_PATH.exists() else []
    if args.dry_run:
        character_counts = exportable_character_correction_counts(character_labels) if character_labels else Counter()
        folded_counts = correction_item_label_counts(LABELS, folded_corrections)
        mixed_counts = correction_item_label_counts(list(MIXEDCASE_LABELS), mixed_corrections)
        report = dry_run_report(
            character_counts,
            folded_counts,
            mixed_counts,
            0 if folded_corrections is None else len(folded_corrections[1]),
            0 if mixed_corrections is None else len(mixed_corrections[1]),
            args.priority_labels,
            args.mixedcase_priority_labels,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_text_dry_run_report(report)
        return

    character_count = export_character_correction_folder(character_labels) if character_labels else 0
    if folded_corrections is None and mixed_corrections is None and character_count == 0:
        print("No character-level corrections with saved source images yet; skipping training.")
        return

    if character_count:
        if character_count < args.min_character_corrections and not args.force:
            print(
                f"Only {character_count} character correction samples are available; "
                f"need at least {args.min_character_corrections} before daily character fine-tuning. "
                "Use --force to override."
            )
        else:
            print(f"Fine-tuning primary character model with {character_count} correction samples.")
            extra_roots = [CHARACTER_CORRECTION_ROOT]
            if HASY_CHARACTER_ROOT.exists():
                extra_roots.insert(0, HASY_CHARACTER_ROOT)
            train_character_model(
                epochs=2,
                batch_size=128,
                min_accuracy=0,
                dataset_root=CHARACTER_DATASET_ROOT,
                model_type="widecnn",
                device_name="auto",
                learning_rate=0.00008,
                label_smoothing=0.02,
                seed=101,
                warm_start=True,
                augment=True,
                extra_roots=extra_roots,
            )

    if folded_corrections is not None:
        folded_count = len(folded_corrections[1])
        if folded_count < args.min_alnum_corrections and not args.force:
            print(
                f"Only {folded_count} folded alnum correction samples are available; "
                f"need at least {args.min_alnum_corrections} before daily folded fine-tuning. "
                "Use --force to override."
            )
        else:
            print(f"Fine-tuning folded alnum model with {folded_count} correction samples.")
            train(
                epochs=3,
                batch_size=2048,
                min_accuracy=0,
                learning_rate=0.00008,
                seed=101,
                augment=False,
                model_type="cnn",
                samples_per_class=2500,
                device_name="auto",
                include_corrections=True,
                warm_start=True,
            )

    if mixed_corrections is not None:
        mixed_count = len(mixed_corrections[1])
        if mixed_count < args.min_alnum_corrections and not args.force:
            print(
                f"Only {mixed_count} mixed-case correction samples are available; "
                f"need at least {args.min_alnum_corrections} before daily mixed-case fine-tuning. "
                "Use --force to override."
            )
        else:
            print(f"Fine-tuning mixed-case model with {mixed_count} correction samples.")
            train_mixedcase(
                epochs=3,
                batch_size=2048,
                min_accuracy=0,
                learning_rate=0.00008,
                seed=101,
                model_type="cnn",
                samples_per_class=2500,
                device_name="auto",
                include_corrections=True,
                warm_start=True,
            )


if __name__ == "__main__":
    main()
