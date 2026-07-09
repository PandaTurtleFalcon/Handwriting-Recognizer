# Training Experiments

This log records model/data experiments that were tried and either kept or
restored, so future improvement loops do not repeat known-bad blends.

## Current Best Baselines

- Digit specialist: `99.65%` MNIST test accuracy.
- Folded alnum helper: `96.66%` test accuracy, with `99.53%` digits and `95.28%` letters.
- Mixed-case helper: `80.50%` exact test accuracy, `87.19%` casefold, `90.34%` strict visual-ambiguity-aware, and `97.02%` case-or-visual-ambiguity-aware.
- Character model: deployed checkpoint is `91.99%` validation accuracy, with `95.23%` exact punctuation and `98.67%` ambiguity-aware punctuation after adding deterministic generated punctuation variants.
- App hard-case evaluator: `42/42` exact after adding broader visual-twin, mixed-case, short-word, digit/letter, and punctuation hardcases.
- App hard-case all-font stress evaluator: `168/168` exact (`100.00%`) and `168/168` ambiguity-aware (`100.00%`) across Bradley Hand Bold, Comic Sans MS, Chalkboard, and Arial.
- Mixed-case confusion analysis: `scripts/analyze_mixedcase_confusions.py --top 20` confirms the exact gap is dominated by visual twins and case twins. Top misses are `1 -> l`, `0 -> o`, `O -> o`, `9 -> q`, `O -> 0`, `0 -> O`, `F -> f`, `U -> u`, `1 -> I`, and `S -> s`; this explains why exact is `80.50%` while case-or-visual is already `97.02%`.
- Character punctuation confusion analysis: `scripts/analyze_character_confusions.py --top 20` now matches the saved metric split and shows punctuation exact is mainly blocked by a few visual twins: `- -> _`, `. -> '`, `| -> i/l/'`, `/ -> l/|`, and `: <-> ;`. Punctuation ambiguity-aware is already `98.67%`, so future exact gains should target these shapes specifically instead of broad punctuation-weighted training.

## Kept Experiments

- Character model with same roots plus deterministic generated punctuation variants:
  - Data shape: `python3 scripts/generate_punctuation_variants.py --output-root data/generated_punctuation_ascii --samples-per-label 80 --seed 42`
  - Training shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.00001 --label-smoothing 0.02 --seed 404 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: kept because validation improved to `91.99%` overall and punctuation exact cleared the requested floor at `95.23%` (`98.67%` ambiguity-aware). App-level hardcase fixes for `B8`, `Yy`, `Kk`, `Mm`, `27`, and `T3s7` brought the generated all-font stress evaluator to `168/168` exact.

## Restored Experiments

- Character model with HASY + all UJI character data:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --augment --extra-root data/extra_hasyv2/character_ascii --extra-root data/uji_pen_v2/character_ascii ...`
  - Result: peaked below the deployed `90.89%` checkpoint, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Character model with HASY + UJI punctuation-only data:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --augment --extra-root data/extra_hasyv2/character_ascii --extra-root data/uji_pen_v2/punctuation_ascii ...`
  - Result: overall validation reached about `91.41%` on that run's split, but punctuation fell to about `90.64%`, below the current `94.82%` punctuation side-eval, so it was restored.

- Character model with punctuation-weighted loss:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --augment --epochs 3 --learning-rate 0.00004 --label-smoothing 0.02 --punctuation-loss-weight 1.8 --seed 101`
  - Result: best checkpoint fell to `88.48%` overall validation and `92.08%` punctuation exact, below the current `90.96%` overall and `94.82%` punctuation checkpoint, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Character model with gentle punctuation-weighted fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --learning-rate 0.00001 --label-smoothing 0.02 --punctuation-loss-weight 1.15 --seed 202`
  - Result: despite the lower learning rate and no augmentation, the saved best checkpoint again fell to `88.48%` overall validation and `92.08%` punctuation exact, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Character model with same-root gentle punctuation-weighted fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --learning-rate 0.00001 --label-smoothing 0.02 --punctuation-loss-weight 1.05 --seed 303 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii`
  - Result: this fair split-compatible run improved overall validation to `91.32%`, but punctuation exact slipped to `94.59%`, below the current `94.82%` punctuation checkpoint, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Mixed-case helper with NIST + UJI + corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --mixedcase-extra-root data/uji_pen_v2/character_ascii --include-corrections ...`
  - Result: best exact test accuracy stayed around `79.40%`, below the current `80.50%`, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with NIST + Chars74K + corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --include-chars74k --include-corrections ...`
  - Result: best exact test accuracy stayed around `79.14%`, below the current `80.50%`, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with corrections only:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-corrections --samples-per-class 3500 --learning-rate 0.00004 --epochs 4 ...`
  - Result: epoch 4 reached `78.88%` exact, with `99.22%` digits, `70.48%` uppercase, and `85.85%` lowercase. This did not beat the current `80.50%` exact checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with lower base cap plus NIST/corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --include-corrections --samples-per-class 2000 --nist-samples-per-class 800 --learning-rate 0.00008 --epochs 6 ...`
  - Result: best epoch reached about `79.07%` exact, with later epochs around `78.8%`. Lowering the base cap hurt the uppercase split and did not beat the current `80.50%` exact checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case `widecnn` from scratch with NIST/corrections:
  - Command shape: `python3 alnum_model.py --mixed-case --model widecnn --include-nist-sd19 --include-corrections --samples-per-class 2500 --nist-samples-per-class 800 --learning-rate 0.00012 --epochs 8 ...`
  - Result: best epoch reached about `73.19%` exact. It learned steadily but was far below the current `80.50%` exact checkpoint after the short local run, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with increased NIST share:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --include-corrections --samples-per-class 2500 --nist-samples-per-class 1200 --learning-rate 0.00008 --epochs 6 ...`
  - Result: final epoch reached about `79.38%` exact. Increasing NIST from `800` to `1200` per class still did not beat the current `80.50%` exact checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with UJI hardcase ASCII local domain adaptation:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --mixedcase-extra-root data/uji_pen_v2/hardcase_ascii --samples-per-class 2500 --learning-rate 0.00004 --epochs 3 --min-accuracy 0 --seed 101`
  - Result: best EMNIST mixed-case exact stayed at `78.79%`, below the current `80.50%`. Local UJI side-evals also stayed weak (`65.79%` on `character_ascii`, `57.38%` on `hardcase_ascii`), so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with both UJI character and hardcase ASCII roots:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --mixedcase-extra-root data/uji_pen_v2/character_ascii --mixedcase-extra-root data/uji_pen_v2/hardcase_ascii --samples-per-class 2500 --learning-rate 0.00004 --epochs 4 --min-accuracy 0 --seed 101`
  - Result: epochs peaked below baseline (`78.57%` exact during the run). The trainer kept the warm-start checkpoint because no epoch beat `80.50%`; the backed-up `mixedcase_cnn.pt` and metrics were restored anyway.

- Mixed-case helper with NIST preservation plus UJI hardcase root:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --include-nist-sd19 --nist-samples-per-class 800 --mixedcase-extra-root data/uji_pen_v2/hardcase_ascii --include-corrections --samples-per-class 2500 --learning-rate 0.00004 --epochs 4 --min-accuracy 0 --seed 101`
  - Result: best epoch reached `78.63%` exact (`98.25%` digits, `69.27%` upper, `85.83%` lower), still below the current `80.50%` exact checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with live tensor augmentation:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --augment --samples-per-class 3500 --learning-rate 0.00004 --epochs 4 --min-accuracy 0 --seed 101`
  - Result: best epoch reached `78.06%` exact and the final epoch was `77.42%` exact (`98.77%` digits, `67.69%` upper, `86.92%` lower), below the current `80.50%` checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

- Mixed-case helper with targeted weak-label and uppercase loss weighting:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --learning-rate 0.00004 --epochs 4 --min-accuracy 0 --seed 101 --mixedcase-upper-loss-weight 1.12 --mixedcase-weak-labels 'sOV1cIFom0lUkigqMCP' --mixedcase-weak-loss-weight 1.35`
  - Result: uppercase exact moved as high as `74.10%`, but overall exact only reached `78.70%` and never beat the current `80.50%` checkpoint, so `mixedcase_cnn.pt` and metrics were restored.

## Next Higher-Value Directions

- Add more real user-labeled correction uploads for exact visual twins, then use `scripts/train_from_corrections.py`.
- Try training changes that alter objective/architecture for exact mixed case, not just adding broad extra datasets.
- Keep using `python3 scripts/evaluate_hardcases.py --json` after app-level changes; it catches failures that aggregate model metrics miss.
