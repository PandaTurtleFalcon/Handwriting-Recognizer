"""Daily fine-tune entrypoint for user-labeled correction data."""

from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path

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


CHARACTER_CORRECTION_ROOT = PROJECT_DIR / "data" / "corrections" / "character_ascii"
HASY_CHARACTER_ROOT = PROJECT_DIR / "data" / "extra_hasyv2" / "character_ascii"


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


def load_character_labels() -> list[str]:
    """Load the deployed 93-class label list for correction export."""

    return [str(label) for label in json.loads(CHARACTER_LABELS_PATH.read_text(encoding="utf-8"))]


def main() -> None:
    """Fine-tune alphanumeric models when usable correction crops exist."""

    folded_corrections = load_correction_cache(LABELS)
    mixed_corrections = load_correction_cache(list(MIXEDCASE_LABELS))
    character_labels = load_character_labels() if CHARACTER_LABELS_PATH.exists() else []
    character_count = export_character_correction_folder(character_labels) if character_labels else 0
    if folded_corrections is None and mixed_corrections is None and character_count == 0:
        print("No character-level corrections with saved source images yet; skipping training.")
        return

    if character_count:
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
        print(f"Fine-tuning folded alnum model with {len(folded_corrections[1])} correction samples.")
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
        print(f"Fine-tuning mixed-case model with {len(mixed_corrections[1])} correction samples.")
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
