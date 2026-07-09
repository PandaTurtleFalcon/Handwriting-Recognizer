# Training Experiments

This log records model/data experiments that were tried and either kept or
restored, so future improvement loops do not repeat known-bad blends.

## Current Best Baselines

- Digit specialist: `99.65%` MNIST test accuracy.
- Folded alnum helper: `96.66%` test accuracy, with `99.53%` digits and `95.28%` letters.
- Mixed-case helper: `80.50%` exact test accuracy, `96.42%` ambiguity-aware.
- Character model: deployed checkpoint is `90.89%` on the combined-extra validation split, with `94.82%` exact punctuation and `98.30%` ambiguity-aware punctuation.
- App hard-case evaluator: `12/12` exact after `fix: resolve visual twin hardcases`.

## Restored Experiments

- Character model with HASY + all UJI character data:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --augment --extra-root data/extra_hasyv2/character_ascii --extra-root data/uji_pen_v2/character_ascii ...`
  - Result: peaked below the deployed `90.89%` checkpoint, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Character model with HASY + UJI punctuation-only data:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --augment --extra-root data/extra_hasyv2/character_ascii --extra-root data/uji_pen_v2/punctuation_ascii ...`
  - Result: overall validation reached about `91.41%` on that run's split, but punctuation fell to about `90.64%`, below the current `94.82%` punctuation side-eval, so it was restored.

- Mixed-case helper with NIST + UJI + corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --mixedcase-extra-root data/uji_pen_v2/character_ascii --include-corrections ...`
  - Result: best exact test accuracy stayed around `79.40%`, below the current `80.50%`, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with NIST + Chars74K + corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --include-chars74k --include-corrections ...`
  - Result: best exact test accuracy stayed around `79.14%`, below the current `80.50%`, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with corrections only:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-corrections --samples-per-class 3500 --learning-rate 0.00004 --epochs 4 ...`
  - Result: epoch 4 reached `78.88%` exact, with `99.22%` digits, `70.48%` uppercase, and `85.85%` lowercase. This did not beat the current `80.50%` exact checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

## Next Higher-Value Directions

- Add more real user-labeled correction uploads for exact visual twins, then use `scripts/train_from_corrections.py`.
- Try training changes that alter objective/architecture for exact mixed case, not just adding broad extra datasets.
- Keep using `python3 scripts/evaluate_hardcases.py --json` after app-level changes; it catches failures that aggregate model metrics miss.
