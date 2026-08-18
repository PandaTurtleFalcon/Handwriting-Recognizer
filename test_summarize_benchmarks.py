import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.summarize_benchmarks import (
    summarize_app_hardcases,
    summarize_correction_memory,
    summarize_correction_training,
    summarize_saved_metrics,
    summarize_uploaded_hardcases,
)


class BenchmarkSummaryTests(unittest.TestCase):
    """Regression tests for saved benchmark gate summaries."""

    def test_summarizes_matching_mixedcase_hybrid_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_weights = root / "mixedcase_cnn.pt"
            folded_weights = root / "alnum_cnn.pt"
            mixed_weights.write_bytes(b"mixed checkpoint")
            folded_weights.write_bytes(b"folded checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"validation_accuracy": 90.0}}))
            (root / "mixedcase_hybrid.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "labels": [str(index) for index in range(10)]
                        + [chr(ord("A") + index) for index in range(26)]
                        + [chr(ord("a") + index) for index in range(26)],
                        "mixedcase_checkpoint_sha256": hashlib.sha256(b"mixed checkpoint").hexdigest(),
                        "folded_checkpoint_sha256": hashlib.sha256(b"folded checkpoint").hexdigest(),
                        "best_checkpoint": {
                            "test_accuracy": 91.25,
                            "case_or_ambiguity_aware_test_accuracy": 98.25,
                            "digit_test_accuracy": 95.0,
                            "upper_test_accuracy": 92.0,
                            "lower_test_accuracy": 80.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 91.25)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 98.25)

    def test_summarizes_matching_mixedcase_family_reranker_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_weights = root / "mixedcase_cnn.pt"
            folded_weights = root / "alnum_cnn.pt"
            bias_path = root / "mixedcase_logit_bias.pt"
            pair_rules_path = root / "mixedcase_pair_rules.json"
            hybrid_path = root / "mixedcase_hybrid.json"
            mixed_weights.write_bytes(b"mixed checkpoint")
            folded_weights.write_bytes(b"folded checkpoint")
            bias_path.write_bytes(b"current bias")
            pair_rules_path.write_bytes(b"current pair rules")
            hybrid_path.write_bytes(b"current hybrid")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"validation_accuracy": 90.0}}))
            torch.save(
                {
                    "enabled": True,
                    "labels": [str(index) for index in range(10)]
                    + [chr(ord("A") + index) for index in range(26)]
                    + [chr(ord("a") + index) for index in range(26)],
                    "mixedcase_checkpoint_sha256": hashlib.sha256(b"mixed checkpoint").hexdigest(),
                    "folded_checkpoint_sha256": hashlib.sha256(b"folded checkpoint").hexdigest(),
                    "mixedcase_logit_bias_sha256": hashlib.sha256(b"current bias").hexdigest(),
                    "mixedcase_pair_rules_sha256": hashlib.sha256(b"current pair rules").hexdigest(),
                    "mixedcase_hybrid_sha256": hashlib.sha256(b"current hybrid").hexdigest(),
                    "best_checkpoint": {
                        "test_accuracy": 91.5,
                        "case_or_ambiguity_aware_test_accuracy": 98.3,
                        "digit_test_accuracy": 95.1,
                        "upper_test_accuracy": 92.2,
                        "lower_test_accuracy": 80.2,
                    },
                },
                root / "mixedcase_family_reranker.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 91.5)
        self.assertEqual(by_name["mixedcase_lower_exact"]["value"], 80.2)

    def test_summarizes_stale_mixedcase_hybrid_as_base_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mixedcase_cnn.pt").write_bytes(b"mixed checkpoint")
            (root / "alnum_cnn.pt").write_bytes(b"folded checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"validation_accuracy": 90.0}}))
            (root / "mixedcase_hybrid.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "labels": [str(index) for index in range(10)]
                        + [chr(ord("A") + index) for index in range(26)]
                        + [chr(ord("a") + index) for index in range(26)],
                        "mixedcase_checkpoint_sha256": "stale",
                        "folded_checkpoint_sha256": "stale",
                        "best_checkpoint": {"test_accuracy": 91.25},
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 80.0)

    def test_ignores_mixedcase_hybrid_when_dependency_hash_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_weights = root / "mixedcase_cnn.pt"
            folded_weights = root / "alnum_cnn.pt"
            bias_path = root / "mixedcase_logit_bias.pt"
            pair_rules_path = root / "mixedcase_pair_rules.json"
            mixed_weights.write_bytes(b"mixed checkpoint")
            folded_weights.write_bytes(b"folded checkpoint")
            bias_path.write_bytes(b"current bias")
            pair_rules_path.write_bytes(b"current pair rules")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"validation_accuracy": 90.0}}))
            (root / "mixedcase_hybrid.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "labels": [str(index) for index in range(10)]
                        + [chr(ord("A") + index) for index in range(26)]
                        + [chr(ord("a") + index) for index in range(26)],
                        "mixedcase_checkpoint_sha256": hashlib.sha256(b"mixed checkpoint").hexdigest(),
                        "folded_checkpoint_sha256": hashlib.sha256(b"folded checkpoint").hexdigest(),
                        "mixedcase_logit_bias_sha256": "stale",
                        "mixedcase_pair_rules_sha256": hashlib.sha256(b"current pair rules").hexdigest(),
                        "best_checkpoint": {"test_accuracy": 91.25},
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 80.0)

    def test_summarizes_pass_fail_saved_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}),
                encoding="utf-8",
            )
            (root / "alnum_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}),
                encoding="utf-8",
            )
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "test_accuracy": 80.0,
                            "case_or_ambiguity_aware_test_accuracy": 97.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 92.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "digit_validation_accuracy": 94.0,
                            "letter_validation_accuracy": 91.0,
                            "punctuation_validation_accuracy": 95.2,
                            "punctuation_ambiguity_aware_validation_accuracy": 98.6,
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertTrue(by_name["digit_specialist_exact"]["passed"])
        self.assertFalse(by_name["mixedcase_exact"]["passed"])
        self.assertTrue(by_name["mixedcase_case_or_visual"]["passed"])
        self.assertFalse(by_name["character_exact"]["passed"])
        self.assertFalse(by_name["character_digit_exact"]["passed"])
        self.assertFalse(by_name["character_letter_exact"]["passed"])
        self.assertTrue(by_name["punctuation_exact"]["passed"])

    def test_summarizes_matching_character_calibration_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_labels.json").write_text(json.dumps(["A", "B"]))
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            torch.save(
                {
                    "labels": ["A", "B"],
                    "best_checkpoint": {
                        "validation_accuracy": 93.0,
                        "ambiguity_aware_validation_accuracy": 99.0,
                        "digit_validation_accuracy": 95.5,
                        "letter_validation_accuracy": 92.5,
                        "punctuation_validation_accuracy": 96.0,
                        "punctuation_ambiguity_aware_validation_accuracy": 99.5,
                    },
                },
                root / "character_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_exact"]["value"], 93.0)
        self.assertEqual(by_name["character_digit_exact"]["value"], 95.5)
        self.assertEqual(by_name["character_letter_exact"]["value"], 92.5)
        self.assertEqual(by_name["punctuation_exact"]["value"], 96.0)

    def test_summarizes_matching_character_pair_rule_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_labels.json").write_text(json.dumps(["A", "B"]))
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            (root / "character_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": ["A", "B"],
                        "best_checkpoint": {
                            "validation_accuracy": 93.7,
                            "ambiguity_aware_validation_accuracy": 99.0,
                            "digit_validation_accuracy": 95.5,
                            "letter_validation_accuracy": 92.9,
                            "punctuation_validation_accuracy": 96.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.5,
                        },
                    }
                )
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_exact"]["value"], 93.7)
        self.assertEqual(by_name["character_letter_exact"]["value"], 92.9)

    def test_pair_rule_aware_character_bias_overrides_pair_rule_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pair_rules_bytes = json.dumps(
                {
                    "labels": ["A", "B"],
                    "best_checkpoint": {
                        "validation_accuracy": 93.7,
                        "letter_validation_accuracy": 92.9,
                    },
                }
            ).encode("utf-8")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_labels.json").write_text(json.dumps(["A", "B"]))
            (root / "character_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"validation_accuracy": 91.0, "letter_validation_accuracy": 90.0}})
            )
            (root / "character_pair_rules.json").write_bytes(pair_rules_bytes)
            torch.save(
                {
                    "labels": ["A", "B"],
                    "includes_pair_rules": True,
                    "pair_rules_sha256": hashlib.sha256(pair_rules_bytes).hexdigest(),
                    "best_checkpoint": {
                        "validation_accuracy": 94.2,
                        "letter_validation_accuracy": 93.4,
                    },
                },
                root / "character_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_exact"]["value"], 94.2)
        self.assertEqual(by_name["character_letter_exact"]["value"], 93.4)

    def test_ignores_stale_character_pair_rule_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_cnn.pt").write_bytes(b"current character checkpoint")
            (root / "character_labels.json").write_text(json.dumps(["A", "B"]))
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "letter_validation_accuracy": 90.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            (root / "character_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": ["A", "B"],
                        "checkpoint_sha256": "not-the-current-checkpoint",
                        "best_checkpoint": {
                            "validation_accuracy": 99.0,
                            "letter_validation_accuracy": 99.0,
                        },
                    }
                )
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_exact"]["value"], 91.0)
        self.assertEqual(by_name["character_letter_exact"]["value"], 90.0)

    def test_summarizes_matching_mixedcase_calibration_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "best_checkpoint": {
                        "test_accuracy": 87.2,
                        "case_or_ambiguity_aware_test_accuracy": 97.6,
                        "digit_test_accuracy": 94.0,
                        "upper_test_accuracy": 84.0,
                        "lower_test_accuracy": 73.0,
                    },
                },
                root / "mixedcase_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 87.2)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 97.6)
        self.assertEqual(by_name["mixedcase_digit_exact"]["value"], 94.0)
        self.assertEqual(by_name["mixedcase_upper_exact"]["value"], 84.0)
        self.assertEqual(by_name["mixedcase_lower_exact"]["value"], 73.0)

    def test_summarizes_matching_mixedcase_pair_rule_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            (root / "mixedcase_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": list(MIXEDCASE_LABELS),
                        "best_checkpoint": {
                            "test_accuracy": 87.5,
                            "case_or_ambiguity_aware_test_accuracy": 97.8,
                            "digit_test_accuracy": 95.1,
                            "upper_test_accuracy": 84.1,
                            "lower_test_accuracy": 72.7,
                        },
                    }
                )
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 87.5)
        self.assertEqual(by_name["mixedcase_digit_exact"]["value"], 95.1)

    def test_summarizes_mixedcase_bias_that_includes_current_pair_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixed_weights = root / "mixedcase_cnn.pt"
            mixed_weights.write_bytes(b"current mixedcase checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            pair_rules_bytes = json.dumps(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                    "best_checkpoint": {
                        "test_accuracy": 87.5,
                        "case_or_ambiguity_aware_test_accuracy": 97.8,
                        "digit_test_accuracy": 95.1,
                        "upper_test_accuracy": 84.1,
                        "lower_test_accuracy": 72.7,
                    },
                }
            ).encode("utf-8")
            (root / "mixedcase_pair_rules.json").write_bytes(pair_rules_bytes)
            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                    "includes_pair_rules": True,
                    "pair_rules_sha256": hashlib.sha256(pair_rules_bytes).hexdigest(),
                    "best_checkpoint": {
                        "test_accuracy": 88.2,
                        "case_or_ambiguity_aware_test_accuracy": 98.1,
                        "digit_test_accuracy": 95.3,
                        "upper_test_accuracy": 84.8,
                        "lower_test_accuracy": 73.2,
                    },
                },
                root / "mixedcase_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 88.2)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 98.1)
        self.assertEqual(by_name["mixedcase_digit_exact"]["value"], 95.3)

    def test_stale_mixedcase_combined_bias_uses_pair_rule_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mixedcase_cnn.pt").write_bytes(b"current mixedcase checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            (root / "mixedcase_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": list(MIXEDCASE_LABELS),
                        "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                        "best_checkpoint": {
                            "test_accuracy": 87.5,
                            "case_or_ambiguity_aware_test_accuracy": 97.8,
                        },
                    }
                )
            )
            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                    "includes_pair_rules": True,
                    "pair_rules_sha256": "stale",
                    "best_checkpoint": {
                        "test_accuracy": 88.2,
                        "case_or_ambiguity_aware_test_accuracy": 98.1,
                    },
                },
                root / "mixedcase_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 87.5)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 97.8)

    def test_mixedcase_combined_bias_rejected_when_pair_rule_checkpoint_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mixedcase_cnn.pt").write_bytes(b"current mixedcase checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "test_accuracy": 80.0,
                            "case_or_ambiguity_aware_test_accuracy": 97.0,
                        }
                    }
                )
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            pair_rules_bytes = json.dumps(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": "stale",
                    "best_checkpoint": {
                        "test_accuracy": 87.5,
                        "case_or_ambiguity_aware_test_accuracy": 97.8,
                    },
                }
            ).encode("utf-8")
            (root / "mixedcase_pair_rules.json").write_bytes(pair_rules_bytes)
            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                    "includes_pair_rules": True,
                    "pair_rules_sha256": hashlib.sha256(pair_rules_bytes).hexdigest(),
                    "best_checkpoint": {
                        "test_accuracy": 88.2,
                        "case_or_ambiguity_aware_test_accuracy": 98.1,
                    },
                },
                root / "mixedcase_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 80.0)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 97.0)

    def test_mixedcase_combined_bias_requires_pair_rule_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mixedcase_cnn.pt").write_bytes(b"current mixedcase checkpoint")
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps({"best_checkpoint": {"test_accuracy": 80.0, "case_or_ambiguity_aware_test_accuracy": 97.0}})
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            (root / "mixedcase_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": list(MIXEDCASE_LABELS),
                        "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                        "best_checkpoint": {
                            "test_accuracy": 87.5,
                            "case_or_ambiguity_aware_test_accuracy": 97.8,
                        },
                    }
                )
            )
            torch.save(
                {
                    "labels": list(MIXEDCASE_LABELS),
                    "checkpoint_sha256": hashlib.sha256(b"current mixedcase checkpoint").hexdigest(),
                    "includes_pair_rules": True,
                    "best_checkpoint": {
                        "test_accuracy": 88.2,
                        "case_or_ambiguity_aware_test_accuracy": 98.1,
                    },
                },
                root / "mixedcase_logit_bias.pt",
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 87.5)
        self.assertEqual(by_name["mixedcase_case_or_visual"]["value"], 97.8)

    def test_ignores_stale_mixedcase_pair_rule_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 99.0}}))
            (root / "alnum_training_metrics.json").write_text(json.dumps({"best_checkpoint": {"test_accuracy": 96.0}}))
            (root / "mixedcase_cnn.pt").write_bytes(b"current mixedcase checkpoint")
            (root / "mixedcase_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "test_accuracy": 80.0,
                            "case_or_ambiguity_aware_test_accuracy": 97.0,
                            "digit_test_accuracy": 91.0,
                            "upper_test_accuracy": 70.0,
                            "lower_test_accuracy": 60.0,
                        }
                    }
                )
            )
            (root / "character_training_metrics.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": {
                            "validation_accuracy": 91.0,
                            "ambiguity_aware_validation_accuracy": 98.0,
                            "punctuation_validation_accuracy": 95.0,
                            "punctuation_ambiguity_aware_validation_accuracy": 99.0,
                        }
                    }
                )
            )
            from alnum_model import MIXEDCASE_LABELS

            (root / "mixedcase_pair_rules.json").write_text(
                json.dumps(
                    {
                        "labels": list(MIXEDCASE_LABELS),
                        "checkpoint_sha256": "not-the-current-checkpoint",
                        "best_checkpoint": {
                            "test_accuracy": 99.0,
                            "case_or_ambiguity_aware_test_accuracy": 99.0,
                            "digit_test_accuracy": 99.0,
                            "upper_test_accuracy": 99.0,
                            "lower_test_accuracy": 99.0,
                        },
                    }
                )
            )

            report = summarize_saved_metrics(root, target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["mixedcase_exact"]["value"], 80.0)
        self.assertEqual(by_name["mixedcase_upper_exact"]["value"], 70.0)
        self.assertEqual(by_name["mixedcase_lower_exact"]["value"], 60.0)

    def test_summarizes_app_hardcase_gates_on_demand(self) -> None:
        with patch(
            "scripts.evaluate_hardcases.evaluate_cases",
            return_value={
                "exact_accuracy": 100.0,
                "exact_correct": 176,
                "ambiguity_aware_accuracy": 100.0,
                "ambiguity_aware_correct": 176,
                "total": 176,
            },
        ) as evaluate:
            report = summarize_app_hardcases(target=95.0, all_fonts=False)

        evaluate.assert_called_once_with(all_fonts=False, script_cases=False)
        by_name = {str(item["name"]): item for item in report}
        self.assertTrue(by_name["app_hardcase_exact"]["passed"])
        self.assertTrue(by_name["app_hardcase_ambiguity"]["passed"])
        self.assertEqual(by_name["app_hardcase_exact"]["correct"], 176)
        self.assertEqual(by_name["app_hardcase_exact"]["total"], 176)

    def test_summarizes_script_hardcase_gates_on_demand(self) -> None:
        with patch(
            "scripts.evaluate_hardcases.evaluate_cases",
            return_value={
                "exact_accuracy": 50.0,
                "exact_correct": 1,
                "ambiguity_aware_accuracy": 100.0,
                "ambiguity_aware_correct": 2,
                "total": 2,
            },
        ) as evaluate:
            report = summarize_app_hardcases(target=95.0, all_fonts=False, script_cases=True)

        evaluate.assert_called_once_with(all_fonts=False, script_cases=True)
        by_name = {str(item["name"]): item for item in report}
        self.assertFalse(by_name["app_script_hardcase_exact"]["passed"])
        self.assertTrue(by_name["app_script_hardcase_ambiguity"]["passed"])
        self.assertEqual(by_name["app_script_hardcase_exact"]["correct"], 1)

    def test_summarizes_uploaded_hardcase_gates_on_demand(self) -> None:
        with patch(
            "scripts.evaluate_hardcases.evaluate_uploaded_fixtures",
            return_value={
                "exact_accuracy": 100.0,
                "exact_correct": 1,
                "ambiguity_aware_accuracy": 100.0,
                "ambiguity_aware_correct": 1,
                "raw_exact_accuracy": 0.0,
                "raw_exact_correct": 0,
                "raw_ambiguity_aware_accuracy": 0.0,
                "raw_ambiguity_aware_correct": 0,
                "total": 1,
            },
        ) as evaluate:
            report = summarize_uploaded_hardcases(target=95.0)

        evaluate.assert_called_once_with()
        by_name = {str(item["name"]): item for item in report}
        self.assertTrue(by_name["uploaded_hardcase_exact"]["passed"])
        self.assertTrue(by_name["uploaded_hardcase_ambiguity"]["passed"])
        self.assertEqual(by_name["uploaded_hardcase_exact"]["correct"], 1)
        self.assertEqual(by_name["uploaded_hardcase_exact"]["total"], 1)
        self.assertFalse(by_name["uploaded_hardcase_raw_exact"]["passed"])
        self.assertFalse(by_name["uploaded_hardcase_raw_ambiguity"]["passed"])
        self.assertEqual(by_name["uploaded_hardcase_raw_exact"]["correct"], 0)

    def test_summarizes_correction_memory_priority_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "character_labels.json").write_text(json.dumps(["s", "O", "V"]), encoding="utf-8")

            with patch("main.CHARACTER_PRACTICE_PRIORITY_LABELS", ["s", "O"]):
                with patch("main.PRACTICE_TARGET_PER_LABEL", 2):
                    with patch(
                        "character_model.load_correction_memory_exemplars",
                        return_value=(torch.zeros((3, 1, 32, 32)), torch.tensor([0, 0, 1])),
                    ):
                        report = summarize_correction_memory(target=95.0, project_dir=root)

        by_name = {str(item["name"]): item for item in report}
        self.assertEqual(by_name["character_correction_memory_samples"]["correct"], 3)
        self.assertEqual(by_name["character_correction_memory_samples"]["total"], 4)
        self.assertAlmostEqual(by_name["character_correction_memory_samples"]["value"], 75.0)
        self.assertEqual(by_name["character_correction_memory_samples"]["by_label"], {"s": 2, "O": 1})
        self.assertEqual(by_name["character_correction_memory_samples"]["not_ready_label_list"], ["O"])
        self.assertEqual(by_name["character_correction_memory_ready_labels"]["correct"], 1)
        self.assertEqual(by_name["character_correction_memory_ready_labels"]["total"], 2)
        self.assertFalse(by_name["character_correction_memory_ready_labels"]["passed"])

    def test_correction_memory_missing_labels_keeps_stable_row_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = summarize_correction_memory(target=95.0, project_dir=Path(temp_dir))

        self.assertEqual(
            [item["name"] for item in report],
            ["character_correction_memory_samples", "character_correction_memory_ready_labels"],
        )

    def test_summarizes_correction_training_priority_coverage(self) -> None:
        folded_targets = torch.tensor([0, 0, 1], dtype=torch.long)
        mixed_targets = torch.tensor([0, 2], dtype=torch.long)

        def fake_load_correction_cache(labels):
            if labels == ["A", "B"]:
                return torch.zeros((3, 1, 32, 32)), folded_targets
            if labels == ["s", "O", "V"]:
                return torch.zeros((2, 1, 32, 32)), mixed_targets
            return None

        with patch("scripts.train_from_corrections.LABELS", ["A", "B"]):
            with patch("scripts.train_from_corrections.MIXEDCASE_LABELS", "sOV"):
                with patch("scripts.train_from_corrections.DEFAULT_PRIORITY_LABELS", "ab!"):
                    with patch("scripts.train_from_corrections.DEFAULT_MIXEDCASE_PRIORITY_LABELS", "sO!V"):
                        with patch("main.PRACTICE_TARGET_PER_LABEL", 2):
                            with patch(
                                "scripts.train_from_corrections.load_correction_cache",
                                side_effect=fake_load_correction_cache,
                            ):
                                report = summarize_correction_training(target=95.0)

        by_name = {str(item["name"]): item for item in report}
        self.assertIn("folded_alnum_correction_training_samples", by_name)
        self.assertIn("mixedcase_correction_training_samples", by_name)
        self.assertNotIn("character_correction_memory_samples", by_name)
        self.assertEqual(by_name["folded_alnum_correction_training_samples"]["priority_labels"], ["A", "B"])
        self.assertEqual(by_name["mixedcase_correction_training_samples"]["priority_labels"], ["s", "O", "V"])
        self.assertEqual(by_name["mixedcase_correction_training_samples"]["by_label"], {"s": 1, "V": 1})
        self.assertEqual(by_name["mixedcase_correction_training_samples"]["not_ready_label_list"], ["O", "s", "V"])
        self.assertEqual(by_name["mixedcase_correction_training_samples"]["correct"], 2)
        self.assertEqual(by_name["mixedcase_correction_training_samples"]["total"], 6)
        self.assertEqual(by_name["mixedcase_correction_training_ready_labels"]["correct"], 0)
        self.assertEqual(by_name["mixedcase_correction_training_ready_labels"]["total"], 3)


if __name__ == "__main__":
    unittest.main()
