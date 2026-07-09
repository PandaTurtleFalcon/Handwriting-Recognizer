"""Daily fine-tune entrypoint for user-labeled correction data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import LABELS, MIXEDCASE_LABELS, load_correction_cache, train, train_mixedcase


def main() -> None:
    """Fine-tune alphanumeric models when usable correction crops exist."""

    folded_corrections = load_correction_cache(LABELS)
    mixed_corrections = load_correction_cache(list(MIXEDCASE_LABELS))
    if folded_corrections is None and mixed_corrections is None:
        print("No character-level corrections with saved source images yet; skipping training.")
        return

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
