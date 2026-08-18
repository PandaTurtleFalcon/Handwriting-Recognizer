"""Build a balanced extra-data pack for hard mixed-case visual families."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alnum_model import (  # noqa: E402
    MIXEDCASE_LABELS,
    build_or_load_chars74k_mixedcase_cache,
    build_or_load_emnist_byclass_mixedcase_cache,
    load_mixedcase_extra_cache,
)


DEFAULT_FAMILIES = ("1Iil", "0Oo", "5Ss", "Cc", "MNmn", "9qg", "Uuv", "2Zz", "4Yy")
DEFAULT_SOURCES = (
    "emnist-train",
    "chars74k",
    "data/the_dataset/the_version4_mixedcase.pt",
    "data/cvl_top6_family_letters.pt",
    "data/unipen_chars/curated_mixedcase_62_2b8c2762df04.pt",
)


def parse_families(value: str) -> tuple[str, ...]:
    """Parse a comma-separated family list, preserving each family's label order."""

    families = tuple(part.strip() for part in value.split(",") if part.strip())
    return families or DEFAULT_FAMILIES


def family_label_indices(families: tuple[str, ...]) -> tuple[int, ...]:
    """Return unique mixed-case label indices selected by visual families."""

    label_to_index = {label: index for index, label in enumerate(MIXEDCASE_LABELS)}
    selected: list[int] = []
    for family in families:
        for label in family:
            if label in label_to_index and label_to_index[label] not in selected:
                selected.append(label_to_index[label])
    if not selected:
        raise ValueError("No requested family labels exist in MIXEDCASE_LABELS.")
    return tuple(selected)


def load_named_source(source: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one named source or `.pt` extra cache."""

    if source == "emnist-train":
        return build_or_load_emnist_byclass_mixedcase_cache(train=True)
    if source == "emnist-test":
        return build_or_load_emnist_byclass_mixedcase_cache(train=False)
    if source == "chars74k":
        return build_or_load_chars74k_mixedcase_cache()
    return load_mixedcase_extra_cache(Path(source))


def balanced_indices_for_labels(
    targets: torch.Tensor,
    label_indices: tuple[int, ...],
    max_per_label: int,
    seed: int,
) -> torch.Tensor:
    """Return shuffled balanced indices for selected labels."""

    generator = torch.Generator().manual_seed(seed)
    selected_parts = []
    for label_index in label_indices:
        indices = torch.where(targets == label_index)[0]
        if not int(indices.numel()):
            continue
        shuffled = indices[torch.randperm(int(indices.numel()), generator=generator)]
        selected_parts.append(shuffled[:max_per_label])
    if not selected_parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(selected_parts)


def build_family_pack(
    sources: tuple[str, ...],
    families: tuple[str, ...],
    max_per_label_per_source: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a balanced tensor pack from selected source datasets."""

    label_indices = family_label_indices(families)
    image_parts = []
    target_parts = []
    source_reports = []
    for source_index, source in enumerate(sources):
        images, targets = load_named_source(source)
        selected = balanced_indices_for_labels(
            targets,
            label_indices,
            max_per_label_per_source,
            seed + source_index,
        )
        if int(selected.numel()):
            image_parts.append(images.index_select(0, selected).float())
            target_parts.append(targets.index_select(0, selected).long())
        counts = Counter(int(target) for target in targets.index_select(0, selected).tolist())
        source_reports.append(
            {
                "source": source,
                "selected": int(selected.numel()),
                "counts": {MIXEDCASE_LABELS[index]: counts.get(index, 0) for index in label_indices},
            }
        )
    if not image_parts:
        raise RuntimeError("No samples were selected for the requested families.")
    images = torch.cat(image_parts)
    targets = torch.cat(target_parts)
    totals = Counter(int(target) for target in targets.tolist())
    metadata = {
        "families": list(families),
        "labels": [MIXEDCASE_LABELS[index] for index in label_indices],
        "sources": source_reports,
        "counts": {MIXEDCASE_LABELS[index]: totals.get(index, 0) for index in label_indices},
        "max_per_label_per_source": max_per_label_per_source,
        "seed": seed,
    }
    return images, targets, metadata


def main() -> None:
    """Run the pack builder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-per-label-per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "tmp" / "mixedcase_top_family_pack.pt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    families = parse_families(args.families)
    images, targets, metadata = build_family_pack(
        sources,
        families,
        max_per_label_per_source=args.max_per_label_per_source,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "targets": targets, "metadata": metadata}, args.output)
    report = {"output": str(args.output), "samples": int(targets.numel()), **metadata}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"wrote {args.output} with {targets.numel()} samples")


if __name__ == "__main__":
    main()
