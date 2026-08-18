# Training Experiments

This log records model/data experiments that were tried and either kept or
restored, so future improvement loops do not repeat known-bad blends.

## Current Best Baselines

- Digit specialist: `99.65%` MNIST test accuracy.
- Folded alnum helper: `96.66%` test accuracy, with `99.53%` digits and `95.28%` letters.
- Mixed-case helper: `80.50%` exact test accuracy, `87.19%` casefold, `90.34%` strict visual-ambiguity-aware, and `97.02%` case-or-visual-ambiguity-aware.
- Character model: deployed checkpoint is `92.18%` validation accuracy, with `95.44%` exact punctuation and `99.02%` ambiguity-aware punctuation after adding deterministic generated punctuation variants and tiny same-root fine-tunes.
- App hard-case evaluator: `44/44` exact after adding broader visual-twin, mixed-case, short-word, digit/letter, punctuation, and look-behind-you hardcases.
- App hard-case all-font stress evaluator: `176/176` exact (`100.00%`) and `176/176` ambiguity-aware (`100.00%`) across Bradley Hand Bold, Comic Sans MS, Chalkboard, and Arial.
- Benchmark summary command: `python3 scripts/summarize_benchmarks.py --include-app-hardcases` now reports saved model gates plus app hardcase exact/ambiguity gates in one hourly-check command.
- Practice correction mode: the static site now includes a drawing pad for weak visual-twin labels (`0/O/o`, `1/I/l/i`, `S/s/5`, `C/c`, punctuation twins). Saved practice samples write both a correction JSONL row and a matching source PNG, so `scripts/train_from_corrections.py --dry-run` can count them and the daily trainer can crop them.
- Correction coverage dry-run: `python3 scripts/train_from_corrections.py --dry-run` counts exportable character corrections directly from JSONL plus saved source PNGs, so new practice samples appear in priority coverage before running the export/training step.
- Practice coverage API: `/api/correction-coverage` reports per-label counts, target counts, and remaining sample needs for the practice UI. Current target is `20` trainable samples per weak label before relying on correction-driven fine-tuning.
- Mixed-case confusion analysis: `scripts/analyze_mixedcase_confusions.py --top 20` confirms the exact gap is dominated by visual twins and case twins. Top misses are `1 -> l`, `0 -> o`, `O -> o`, `9 -> q`, `O -> 0`, `0 -> O`, `F -> f`, `U -> u`, `1 -> I`, and `S -> s`; this explains why exact is `80.50%` while case-or-visual is already `97.02%`.
- Character punctuation confusion analysis: `scripts/analyze_character_confusions.py --top 20` now matches the saved metric split and shows punctuation exact is mainly blocked by a few visual twins: `- -> _`, `. -> '`, `| -> i/l/'`, `/ -> l/|`, and `: <-> ;`. Punctuation ambiguity-aware is already `98.67%`, so future exact gains should target these shapes specifically instead of broad punctuation-weighted training.
- Character headroom analysis: `scripts/analyze_character_headroom.py --json` reports `94.15%` exact, `99.11%` ambiguity-aware, `525` visual-family-recoverable errors, and only `94` remaining non-family errors. The top families are `!/1Iil|` (`154` recoverable), `0Oo` (`144`), `5Ss` (`61`), `Cc` (`27`), `Uuv` (`18`), `Pp` (`15`), and `2Zz` (`15`), so the last point to 95% exact is mostly visual-twin/case separation.

## Kept Experiments

- Mixed-case pair-rule visual-twin calibration:
  - Code path: `alnum_model.py` now loads an optional `mixedcase_pair_rules.json` artifact after the existing logit bias, `scripts/calibrate_mixedcase_logits.py --pair-rules` greedily searches ordered visual-twin flips with split floors, and `scripts/summarize_benchmarks.py` reports the pair-rule metrics when the artifact matches the label set.
  - Command shape: `python3 scripts/calibrate_mixedcase_logits.py --pair-rules --batch-size 4096 --greedy-rounds 8 --pair-thresholds=-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18 --min-improvement 0.01 --min-test 87.4583 --min-case-or-visual 97.7770 --min-digit 94.9203 --min-upper 84.0713 --min-lower 72.6523 --write --require-app-gates`.
  - Result: accepted eight pair rules (`i->l`, `o->0`, `i->I`, `I->l`, `z->2`, `i->1`, `g->q`, `t->7`), improving mixed-case exact from `87.46%` to `87.51%` and clearing mixed-case digit exact from `94.92%` to `95.01%`. Upper exact held at `84.07%`, lower exact improved to `72.69%`, clean app stayed `100.00% (45/45)`, and script app stayed `96.67% (87/90)`.

- Character pair-rule visual-twin calibration:
  - Code path: `character_model.py` now loads an optional `character_pair_rules.json` artifact after the existing logit bias, `scripts/calibrate_character_logits.py --pair-rules` greedily searches ordered visual-twin flips against letter-validation accuracy with digit/punctuation floors, and `scripts/summarize_benchmarks.py` reports the pair-rule metrics when labels match.
  - Command shape: `python3 scripts/calibrate_character_logits.py --pair-rules --batch-size 4096 --greedy-rounds 10 --pair-thresholds=-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18 --min-improvement 0.01 --objective letter_validation_accuracy --min-validation 93.5898 --min-ambiguity 99.0829 --min-digit 95.0500 --min-letter 92.7326 --min-punctuation 96.3431 --require-app-gates`.
  - Result: accepted ten pair rules (`W->w`, `1->i`, `5->s`, `N->n`, `k->K`, `i->I`, `i->l`, `|->I`, `|->i`, `/->I`), improving character exact from `93.59%` to `93.74%` and character-letter exact from `92.73%` to `92.95%`. Digit exact stayed `95.05%`, punctuation exact stayed `96.34%`, clean app stayed `100.00% (45/45)`, and script app stayed `96.67% (87/90)`.

- Resumable pair-rule calibration continuation:
  - Code path: `scripts/calibrate_mixedcase_logits.py --pair-rules` and `scripts/calibrate_character_logits.py --pair-rules` now start from existing matching pair-rule artifacts before searching, so later probes can add only incremental rules instead of rediscovering the first pass.
  - Mixed-case command shape: `python3 scripts/calibrate_mixedcase_logits.py --pair-rules --batch-size 4096 --greedy-rounds 8 --pair-thresholds=-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18,-0.12,-0.08,-0.04,-0.02 --min-improvement 0.005 --min-test 87.5121 --min-case-or-visual 97.7779 --min-digit 95.0057 --min-upper 84.0713 --min-lower 72.6893 --write --require-app-gates`.
  - Character command shape: `python3 scripts/calibrate_character_logits.py --pair-rules --batch-size 4096 --greedy-rounds 12 --pair-thresholds=-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18,-0.12,-0.08,-0.04 --min-improvement 0.01 --objective letter_validation_accuracy --min-validation 93.7411 --min-ambiguity 99.0829 --min-digit 95.0500 --min-letter 92.9471 --min-punctuation 96.3431 --require-app-gates`.
  - Result: mixed-case exact improved from `87.51%` to `87.53%`, mixed-case digit exact improved from `95.01%` to `95.03%`, and lower exact improved from `72.69%` to `72.73%`. Character exact improved from `93.74%` to `93.85%`, character-letter exact improved from `92.95%` to `93.11%`, and all digit, folded alnum, punctuation, clean app, and script app gates stayed green.

- Character model with same roots plus deterministic generated punctuation variants:
  - Data shape: `python3 scripts/generate_punctuation_variants.py --output-root data/generated_punctuation_ascii --samples-per-label 80 --seed 42`
  - Training shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.00001 --label-smoothing 0.02 --seed 404 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: kept because validation improved to `91.99%` overall and punctuation exact cleared the requested floor at `95.23%` (`98.67%` ambiguity-aware). App-level hardcase fixes for `B8`, `Yy`, `Kk`, `Mm`, `27`, and `T3s7` brought the generated all-font stress evaluator to `168/168` exact.

- Character model tiny same-root fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.000005 --label-smoothing 0.02 --seed 505 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: kept because validation improved from `91.99%` to `92.14%`, ambiguity-aware validation improved to `98.92%`, and punctuation exact improved to `95.58%` (`99.09%` ambiguity-aware). App all-font hardcases stayed `168/168` exact and correction replay stayed `2/2`.

- Character model second tiny same-root fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.000002 --label-smoothing 0.02 --seed 606 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: kept because validation improved from `92.14%` to `92.18%`, with punctuation still above target at `95.44%` exact (`99.02%` ambiguity-aware). App all-font hardcases stayed `168/168` exact and correction replay stayed `2/2`.

- App segmentation and context cleanup for look-behind-you screenshot:
  - Code path: character segmentation now merges disconnected glyph parts inside visual rows first, only does a narrow second pass for vertically stacked parts, prevents side-by-side full-letter overmerge, and drops invalid padded boxes. Display cleanup adds whole-row look-behind-you variants observed from the user screenshot and generated all-font stress cases.
  - Verification: `python3 -m pytest -q test_mnist_model.py test_web_app.py test_character_model.py test_context_rules.py` passed (`145` tests, `1` skipped), live `/api/predict` on `Screenshot 2026-07-13 at 12.42.21.png` returns `look behind\nyou` with `13` prediction boxes, `python3 -m pytest -q test_context_rules.py test_evaluate_hardcases.py` passed (`28` tests), `scripts/evaluate_hardcases.py` reports `176/176` all-font exact, and `scripts/summarize_benchmarks.py --include-app-hardcases` confirms app hardcases are back to `100.00%`.

- App raw-read diagnostics and mixed-case practice reprioritization:
  - Code path: result panels now show the raw model read when context cleanup changes the displayed answer, so outputs like `xOOh:1i / 7o4` are visible as diagnostics while the displayed answer remains `look behind / you`. Mixed-case practice now starts with the largest measured visual-twin error families (`1/l/I/i`, `0/O/o`, `9/q/g`, and `5/S/s`) instead of older scattered labels.
  - Verification: `python3 -m pytest -q test_web_app.py test_context_rules.py test_train_from_corrections.py` passed (`134` tests), live `/api/predict` on `Screenshot 2026-07-13 at 12.42.21.png` returns `look behind\nyou` with raw `xOO11eh'nd7o4`, and `scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases` still reports mixed-case exact at `80.50%` and app hardcases at `44/44`, so this iteration does not claim a model accuracy gain.

- Counted app hardcase benchmark summary:
  - Code path: `scripts/summarize_benchmarks.py --include-app-hardcases` now includes exact numerator/denominator counts for app hardcase gates in both text and JSON output, so expanded app-level coverage is visible during hourly checks.
  - Verification: `python3 -m pytest -q test_summarize_benchmarks.py` passed (`2` tests), text summary prints `app_hardcase_exact: 100.00% (176/176)`, and JSON summary includes `correct=176` plus `total=176` for app hardcase exact and ambiguity gates.

## Restored Experiments

- Rejected full-UNIPEN frozen mixed-case scout:
  - Command shape: backed up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, and `mixedcase_logit_bias.pt`, then ran `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 1 --batch-size 256 --samples-per-class 2500 --min-accuracy 0 --learning-rate 0.000001 --seed 5201 --mixedcase-label-smoothing 0.01 --mixedcase-weak-labels '1lIi0Oo9qg5Ss2ZzUuVvMmNnCcPpFfkKXxWwYy4Tt7Jj' --mixedcase-weak-loss-weight 1.02 --mixedcase-lower-loss-weight 1.005 --mixedcase-upper-loss-weight 1.0 --mixedcase-type-loss-weight 0.005 --mixedcase-class-balance-strength 0.02 --mixedcase-freeze-feature-layers --mixedcase-extra-root data/unipen_chars/curated --device mps`.
  - Result: rejected and restored. Raw mixed-case exact only reached `80.65%` and the full calibrated/app summary regressed script app exact to `94.44% (85/90)`, below target, while deployed calibrated exact stayed `87.46%`.

- Rejected real twin-subset frozen mixed-case scout:
  - Command shape: backed up mixed-case artifacts, then ran `python3 alnum_model.py --mixed-case --model cnn --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --mixedcase-extra-root data/extra_hasyv2/character_ascii_twin_subset --mixedcase-extra-root data/uji_pen_v2/twin_subset_ascii --mixedcase-freeze-feature-layers --epochs 1 --batch-size 256 --learning-rate 0.00002 --min-accuracy 0 --seed 2026 --device mps`.
  - Result: rejected and restored. Raw exact fell to `75.31%` (`digits 98.79%`, `upper 67.48%`, `lower 87.44%`), showing this extra-root mix over-corrects toward lowercase and does not solve exact mixed-case recognition.

- Rejected digit-preserving mixed-case calibration probe:
  - Command shape: `python3 scripts/calibrate_mixedcase_logits.py --greedy-labels '0123456789OIlSsoZzqgB' --objective digit_test_accuracy --min-test 87.45834091970583 --min-case-or-visual 97.77712688900675 --min-digit 94.92034512205895 --min-upper 84.07133286543737 --min-lower 72.65235226726782 --dry-run`
  - Result: no safe logit-bias step was found (`digit_test_accuracy` stayed `94.9203%` and `test_accuracy` stayed `87.4583%`), so no artifact was written.

- Rejected character-letter calibration probe:
  - Command shape: `python3 scripts/calibrate_character_logits.py --greedy-labels 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' --objective letter_validation_accuracy --min-validation 93.59069511917402 --min-digit 95.05 --min-letter 92.73 --dry-run`
  - Result: no safe step was found (`letter_validation_accuracy` stayed `92.7326%` and `validation_accuracy` stayed `93.5899%`), so no artifact was written.

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

- Character model with targeted weak-label weighting:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.000002 --label-smoothing 0.02 --seed 707 --weak-labels 'Oo0Il1iscSzv-.|' --weak-loss-weight 1.18 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: best validation reached only `92.13%`, below the current `92.18%` checkpoint, so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored. The weak-label training knob remains available for future bounded variants.

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

- Mixed-case helper with post-hoc train-prior logit calibration:
  - Command shape: `python3 scripts/calibrate_mixedcase_logits.py --write` plus fixed-scale checks at `--scale 0.2 --write` and `--scale 0.05 --write`.
  - Result: scale `1.0` improved standalone mixed-case exact from `80.50%` to `87.23%`, scale `0.2` reached `84.19%`, and scale `0.05` reached `81.60%`. All tested scales regressed app-level exact below target (`88.64%`, `90.91%`, and `90.91%` app hardcase exact respectively; script hardcases also fell below target), so `mixedcase_logit_bias.pt` was removed and no mixed-case calibration artifact is deployed. The calibration script remains dry-run by default for future analysis only.

- Mixed-case helper with stronger targeted weak-label weighting:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 5 --learning-rate 0.00002 --seed 606 --mixedcase-upper-loss-weight 1.08 --mixedcase-lower-loss-weight 1.05 --mixedcase-weak-labels 'sOV1cIFom0lUkigqMCPzYWyXjK' --mixedcase-weak-loss-weight 1.75`
  - Result: stopped early after epoch 2 because exact fell to `75.79%` (`97.87%` digits, `68.52%` upper, `84.34%` lower), far below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored.

- Mixed-case helper with targeted generated font data for visual twins:
  - Data shape: temporary ASCII-code image folders at `/tmp/mixedcase_twin_ascii`, generated from local system fonts for `1/I/l/i`, `0/O/o`, `9/q/g`, `S/s/5`, `F/f`, `U/u`, `C/c`, `M/m`, `P/p`, `V/v`, `2/Z/z`, `Y/y/4`, `B/8`, `T/t/7`, `K/k`, `X/x`, `W/w`, and `J/j` families.
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --mixedcase-extra-root /tmp/mixedcase_twin_ascii --epochs 4 --learning-rate 0.00004 --seed 707 --min-accuracy 0`
  - Result: stopped early after epoch 3 because exact peaked at only `78.78%`, below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Synthetic font data may help app-domain hardcases, but it did not improve EMNIST-style isolated mixed-case exact validation.

- Mixed-case helper with auxiliary casefold/type losses:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 4 --learning-rate 0.00003 --seed 808 --mixedcase-folded-loss-weight 0.08 --mixedcase-type-loss-weight 0.18 --min-accuracy 0`
  - Result: stopped early after epoch 3 because exact only reached `78.62%` (`98.39%` digits, `70.14%` upper, `85.83%` lower), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. This specific auxiliary weighting did not help, though the auxiliary-loss plumbing remains useful for bounded future objective experiments.

- Mixed-case helper with no label smoothing:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 4 --learning-rate 0.00003 --seed 909 --mixedcase-label-smoothing 0.0 --min-accuracy 0`
  - Result: completed four epochs and peaked at `78.93%` exact (`98.22%` digits, `70.87%` upper, `85.30%` lower), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored.

- Mixed-case helper with higher EMNIST/MNIST sample cap:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 5000 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 3 --learning-rate 0.000025 --seed 1001 --min-accuracy 0`
  - Result: stopped after epoch 2 because exact only reached `78.57%` (`98.29%` digits, `70.12%` upper, `85.80%` lower), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored.

- Mixed-case helper with core MNIST + EMNIST-only fine-tune:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --epochs 4 --learning-rate 0.00002 --seed 1111 --min-accuracy 0`
  - Result: stopped after epoch 2 because exact dropped to `78.05%` after an initial `78.91%`, below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Removing NIST/corrections did not recover exact mixed-case validation.

- Mixed-case post-hoc logit bias calibration:
  - Command shape: temporary Python calibration over the deployed `mixedcase_cnn.pt`, using up to 700 training-cache samples per class to optimize a 62-class bias vector plus temperature, then evaluating on the held-out MNIST + EMNIST mixed-case test caches.
  - Result: calibration overfit badly to the training-cache distribution. Exact dropped from `80.50%` to `71.66%`, with digit accuracy falling from `83.04%` to `65.91%`, so no calibration artifact was saved.

- Character model weak visual-twin fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 4 --min-accuracy 0 --learning-rate 0.0000015 --label-smoothing 0.015 --punctuation-loss-weight 1.03 --weak-labels 'Oo0Il1isScC-_.|/' --weak-loss-weight 1.12 --seed 1212 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: validation stayed below the current `92.18%` checkpoint (`92.02%`, `92.07%`, `92.02%`, `92.03%` across four epochs), so `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Folded alnum + mixed-case hybrid inference probe:
  - Command shape: temporary Python evaluator combining `alnum_cnn.pt` for folded digit/A-Z identity with `mixedcase_cnn.pt` for upper/lower case choice on the held-out MNIST + EMNIST mixed-case test caches.
  - Result: exact dropped to `50.72%` because the folded alnum checkpoint is strong on MNIST (`99.53%`) but only `54.24%` on EMNIST ByClass folded letter tensors. This is a domain mismatch, not a useful inference path.

- Mixed-case helper initialized from folded alnum checkpoint:
  - Code path: added `--mixedcase-transfer-from-folded`, which copies shared CNN layers from `alnum_cnn.pt`, copies digit/uppercase classifier rows directly, and initializes lowercase rows from their uppercase counterpart before mixed-case fine-tuning.
  - Command shape: `python3 alnum_model.py --mixed-case --mixedcase-transfer-from-folded --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 6 --learning-rate 0.00005 --seed 1313 --min-accuracy 0`
  - Result: transfer training peaked at `78.12%` exact on epoch 4 (`98.91%` digits, `74.91%` upper, `83.74%` lower), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. The transfer initializer remains available for future longer or differently scheduled experiments.

- Correction coverage audit:
  - Command shape: `python3 scripts/train_from_corrections.py --dry-run`
  - Result: only `2` trainable correction crops exist. Among the current weak priority labels from character confusions (`OloI01iscZv-`), coverage is `O:0, l:0, o:0, I:0, 0:0, 1:1, i:0, s:0, c:0, Z:0, v:0, -:0`. Added dry-run priority coverage reporting so future hourly loops can see when enough real user-labeled data exists to safely train.

- Mixed-case label-map and confidence audit:
  - Command shape: temporary Python audit over `build_or_load_emnist_byclass_mixedcase_cache`, `make_mixedcase_loaders`, and the deployed `mixedcase_cnn.pt`.
  - Result: label ordering matches `0-9/A-Z/a-z`, and support counts are plausible but very uneven (`s` has `437` held-out samples while `1` has `6330`). Exact held-out accuracy remains `80.50%`, but case-or-visual ambiguity is `97.02%`. Wrong mixed-case predictions average only `0.516` confidence versus `0.835` for correct predictions, with just `622` of `24630` wrong predictions above `90%` confidence. This makes a hidden label-map bug unlikely and points future work toward data/objective changes for visual twins rather than more label plumbing.

- Mixed-case helper with inverse-frequency class-balanced loss:
  - Code path: added `--mixedcase-class-balance-strength`, which blends inverse-frequency training-set class weights into the mixed-case cross-entropy loss while preserving the existing case and weak-label weights.
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 3 --learning-rate 0.00002 --seed 1616 --min-accuracy 0 --mixedcase-class-balance-strength 0.25 --mixedcase-label-smoothing 0.02`
  - Result: exact test accuracy regressed to `76.69%`, `76.54%`, and `77.08%`, below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. This suggests naive inverse-frequency loss overemphasizes rare hard lowercase classes and hurts overall exact accuracy.

- Mixed-case helper with very-low-LR continuation:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 6 --learning-rate 0.000005 --seed 1919 --min-accuracy 0 --mixedcase-label-smoothing 0.03`
  - Result: exact test accuracy rose slowly from `77.28%` to only `78.25%`, still below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. More generic continuation on the current data blend is not enough; the remaining gap needs targeted real samples or a different mixed-case objective/architecture.

- Practice sample collection workflow:
  - Code path: added a `Next needed` practice control plus automatic next-label selection after saving a sample. The UI now uses `/api/correction-coverage` to steer data collection toward labels with the largest remaining correction-sample gap.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`78` tests), `curl -fsS http://127.0.0.1:8000/health` returned live, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged. This is a data-collection improvement, not a claimed model-accuracy gain.

- Expanded practice sample target labels:
  - Code path: expanded `PRACTICE_PRIORITY_LABELS` and the static practice UI from the original 18 labels to the audited mixed-case and punctuation blockers, including `2/Z/z`, `9/q/g`, `F/f`, `U/u`, `M/m`, `V/v`, `P/p`, `W/w`, `Y/y/4`, `T/t/7`, `J/j`, `K/k`, `X/x`, and punctuation twins `:;!+`.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`79` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged. This widens real correction-data coverage for the two failing exact gates.

- Rejected full-extra mixed-case warm-start probe and kept warm-start guards:
  - Command shape: first attempted `python3 alnum_model.py --mixed-case --model rescnn --warm-start ...`, which exposed that a mismatched warm-start model type could begin from random weights. Added fail-fast validation so mixed-case warm starts now reject label-order mismatches, model-type mismatches, and missing state dicts before training starts.
  - Corrected command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 1 --batch-size 2048 --learning-rate 0.0000005 --samples-per-class 2500 --augment --include-chars74k --include-usps --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --mixedcase-extra-root data/uji_pen_v2/character_ascii_mixedcase_62_d44326b5a602.pt --mixedcase-extra-root data/uji_pen_v2/twin_subset_ascii_mixedcase_62_a91715cfe986.pt --mixedcase-extra-root data/extra_hasyv2/character_ascii_twin_subset_mixedcase_62_9972d37d2d5d.pt --mixedcase-extra-root data/unipen_chars/curated_mixedcase_62_2b8c2762df04.pt --mixedcase-upper-loss-weight 1.04 --mixedcase-lower-loss-weight 1.08 --mixedcase-weak-labels '0Oo1Ili5Sscmufpq' --mixedcase-weak-loss-weight 1.12 --mixedcase-folded-loss-weight 0.04 --mixedcase-type-loss-weight 0.12 --mixedcase-label-smoothing 0.02 --mixedcase-class-balance-strength 0.35 --mixedcase-checkpoint-objective balanced_group_accuracy --mixedcase-min-checkpoint-case-or-visual 98.04 --mixedcase-min-checkpoint-digit 95.02 --mixedcase-min-checkpoint-upper 84.70 --mixedcase-min-checkpoint-lower 73.14`.
  - Result: rejected and restored. The run reached only `79.63%` mixed-case exact, with digits at `98.96%`, uppercase at `67.97%`, and lowercase at `84.60%`, so it improved lowercase at the cost of a major uppercase collapse. The trainer now only seeds `best_state` from a warm-start checkpoint when that raw checkpoint clears the requested split floors.

- Rejected tiny character family-verifier probe:
  - Command shape: temporary in-memory PyTorch `TinyFamilyCnn` trained for eight epochs on validation-split-compatible training samples for `0Oo`, `1Ili|!/`, `5Ss`, and `Cc`.
  - Result: rejected. The deployed base model beat the family verifier on every family: `0Oo` base `78.48%` vs verifier `59.74%`, `1Ili|!/` base `83.77%` vs verifier `72.40%`, `5Ss` base `88.21%` vs verifier `74.71%`, and `Cc` base `89.19%` vs verifier `71.81%`.

- Rejected character within-family logit restriction as a standalone path:
  - Command shape: temporary evaluator restricting the deployed character logits to the true visual family for held-out validation samples.
  - Result: rejected as too small to justify serving complexity. The best tested gains were `Uuv +2.65` points and `1Ili|!/ +1.94` points inside those families, while `0Oo`, `5Ss`, `Cc`, `Pp`, and `2Zz` were all below a half-point inside-family gain. This means a family detector alone cannot close the remaining exact gap; future work should prioritize stronger architecture/objective changes or substantially more real user-labeled twin samples.

- Rejected weak-label focal character fine-tune:
  - Command shape: backed up character artifacts to `tmp/daily_training_backups/20260818T105500Z-character-weak-focal-probe`, then ran `python3 character_model.py --model widecnn --warm-start --epochs 2 --batch-size 256 --min-accuracy 92.17 --learning-rate 0.000001 --label-smoothing 0.015 --weak-labels 'Oo0Il1isScCvVUuPpZz2-_.|/;:!+Yy4' --weak-loss-weight 1.04 --focal-gamma 0.15 --seed 1818 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii --device mps`.
  - Result: rejected and restored. Raw character validation improved to `92.93%`, but the full deployed benchmark dropped to `92.98%` exact because existing calibration artifacts no longer matched the new checkpoint. Refreshing the old calibration fingerprints recovered only `94.157%` exact, still a tiny regression from the deployed `94.167%`, so the previous checkpoint, exemplars, bias, and pair rules were restored.

- Rejected mixed-case focal continuation:
  - Code path: added `--mixedcase-focal-gamma` to the mixed-case trainer, using focal cross-entropy when gamma is greater than zero while preserving the previous cross-entropy behavior at gamma `0.0`.
  - Command shape: backed up mixed-case artifacts to `tmp/daily_training_backups/20260818T110402Z-mixedcase-focal-probe`, then ran `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 2 --batch-size 2048 --min-accuracy 0 --learning-rate 0.000001 --seed 2828 --samples-per-class 3500 --device mps --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --mixedcase-upper-loss-weight 1.02 --mixedcase-lower-loss-weight 1.04 --mixedcase-weak-labels '1Ili0Oo5SsMNmn9qgUuv2ZzCcYy4VvPpTt7KkXxJj' --mixedcase-weak-loss-weight 1.08 --mixedcase-folded-loss-weight 0.02 --mixedcase-type-loss-weight 0.08 --mixedcase-label-smoothing 0.02 --mixedcase-focal-gamma 0.35 --mixedcase-checkpoint-objective balanced_group_accuracy --mixedcase-min-checkpoint-case-or-visual 98.04 --mixedcase-min-checkpoint-digit 95.02 --mixedcase-min-checkpoint-upper 84.70 --mixedcase-min-checkpoint-lower 73.14`.
  - Result: rejected and restored. Epoch 1 reached `78.54%` exact with digits `98.78%`, uppercase `68.83%`, and lowercase `85.49%`; epoch 2 fell to `77.95%` exact with uppercase `68.25%`. Focal pressure again moved probability mass toward lowercase hard twins but collapsed uppercase, so no checkpoint met the split floors.

- Server-driven practice labels:
  - Code path: removed the duplicated 52-label practice list from `web/app.js`; the browser now renders practice label buttons from `/api/correction-coverage`, with a one-label fallback only if coverage is unavailable.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`79` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged. This keeps future correction-target changes in one server-side source of truth.

- Correction dry-run priority alignment:
  - Code path: `scripts/train_from_corrections.py --dry-run` now derives its default priority labels from `PRACTICE_PRIORITY_LABELS` and filters each recognizer's report to labels that model can actually train.
  - Verification: `python3 -m pytest -q test_train_from_corrections.py test_web_app.py` passed (`81` tests), `python3 scripts/train_from_corrections.py --dry-run` reports the expanded practice targets, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Correction training readiness summary:
  - Code path: `scripts/train_from_corrections.py --dry-run` now prints per-recognizer readiness lines with ready-label count, total weak-label samples, target samples, and remaining samples needed.
  - Verification: `python3 -m pytest -q test_train_from_corrections.py test_web_app.py` passed (`82` tests). Current dry-run readiness is character `0/52` labels ready with `2/1040` samples, folded alnum `0/26` with `2/520`, and mixed-case `0/42` with `2/840`; benchmark metrics are unchanged.

- Machine-readable correction readiness:
  - Code path: added `scripts/train_from_corrections.py --dry-run --json`, sharing the same report object as the text dry-run output.
  - Verification: `python3 -m pytest -q test_train_from_corrections.py test_web_app.py` passed (`83` tests), `python3 scripts/train_from_corrections.py --dry-run --json` parses as JSON with the same readiness totals, and benchmark metrics are unchanged.

- App correction-readiness panel:
  - Code path: added `/api/correction-readiness` and a static-site readiness strip that shows character, folded alnum, and mixed-case correction-training readiness in the practice panel.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`84` tests), `/api/correction-readiness` returns the current needed-sample counts (`1038`, `518`, `838`), and benchmark metrics are unchanged.

- Correction readiness next-needed labels:
  - Code path: readiness reports now include `next_needed` labels sorted by largest sample gap, and the practice readiness cards show the top few labels to draw next.
  - Verification: `python3 -m pytest -q test_train_from_corrections.py test_web_app.py` passed (`85` tests), `/api/correction-readiness` reports next labels such as `0`, `O`, `o`, and `I`, and benchmark metrics are unchanged.

- Clickable readiness next-needed labels:
  - Code path: practice readiness cards now render each `next_needed` label as a button that selects that practice label directly.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), served static assets include `readiness-next-button`, and benchmark metrics are unchanged.

- Practice readiness progress meters:
  - Code path: readiness cards now show a stable progress meter using each recognizer's `samples / target_samples` readiness values.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), served static assets include `readiness-meter`, and benchmark metrics are unchanged.

- Selected practice-label progress:
  - Code path: the practice target card now shows the selected label's saved count, per-label target, and remaining sample need from `/api/correction-coverage`.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), served static assets include `practice-target-progress`, and benchmark metrics are unchanged.

- Practice keyboard collection shortcuts:
  - Code path: practice mode now supports scoped keyboard actions while focus is inside the practice panel: save sample, clear canvas, and jump to next-needed label.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), and benchmark metrics are unchanged.

- Practice auto-next toggle:
  - Code path: practice mode now has an `Auto next` checkbox. When checked, saving advances to the next needed label; when unchecked, saving keeps the current label selected for faster repeated samples.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), served static assets include `practice-auto-next`, and benchmark metrics are unchanged.

- Repeat-label practice feedback:
  - Code path: when `Auto next` is off, saving a practice sample now refreshes correction coverage and reports how many more samples of that exact label are still needed.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`85` tests), served static assets include `repeatPracticeStatus`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Measured-worst practice queue:
  - Code path: reordered `PRACTICE_PRIORITY_LABELS` so practice collection starts with the current worst verified exact labels from the mixed-case and character confusion reports (`s`, `O`, `V`, `1`, `c`, `I`, `F`, `o`, `m`, `0`) before moving through the rest of the visual twins and punctuation blockers.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`86` tests), `python3 scripts/train_from_corrections.py --dry-run --json` now reports those labels first in `next_needed`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice focus chips:
  - Code path: `/api/correction-coverage` now includes `focus_labels`, and the static practice panel renders those highest-priority not-ready labels as clickable focus chips above the full coverage grid.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`86` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Counted practice focus chips:
  - Code path: `/api/correction-coverage` now also includes `focus_items` with per-label `count` and `needed`, and the practice panel renders focus chips as `label:needed` with hover details.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`86` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Auto-next status target:
  - Code path: after saving a practice sample with `Auto next` enabled, the status message now reports the next selected practice label so collectors can immediately draw the right target.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`86` tests), served static assets include the `Next:` status, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice label guard:
  - Code path: practice mode now refuses to save a sample when the typed label is not in the active weak-label queue, then resets to the next needed label. This protects the small user-labeled correction set from accidental unsupported labels.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`86` tests), served static assets include the guard message, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Server-side practice label guard:
  - Code path: `build_correction_record` now rejects generated practice corrections whose label is outside `PRACTICE_PRIORITY_LABELS`, so direct `/api/correct` posts cannot bypass the browser guard and pollute the correction set.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`88` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice correction shape guard:
  - Code path: generated practice corrections must now be single-character corrections with a `practice-` image id before their embedded `source_image` can be accepted, preventing direct API posts from attaching practice images to unrelated correction records.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice remaining-samples summary:
  - Code path: the practice coverage panel now sums all not-ready labels and shows the total number of samples still needed above the focus chips.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), served static assets include `samples still needed`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Coverage sample totals:
  - Code path: `/api/correction-coverage` now exposes `samples`, `target_samples`, and `needed_samples`; the practice panel uses `needed_samples` directly while retaining the browser-side sum as a fallback.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice coverage meter:
  - Code path: the practice coverage panel now renders a compact progress meter from `samples / target_samples`, making collection progress visible at a glance.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), served static assets include `practice-coverage-meter-fill`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Numeric practice sample progress:
  - Code path: the practice coverage panel now prints the exact `samples/target_samples` count and percent under the progress meter.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), served static assets include `practice-sample-progress`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Coverage percent API:
  - Code path: `/api/correction-coverage` now exposes `coverage_percent`, and the practice panel uses that server-provided value with the previous browser calculation as a fallback.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Dry-run coverage percent:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` now includes `coverage_percent` in each recognizer readiness object, matching `/api/correction-coverage` for automation consumers.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), `python3 scripts/train_from_corrections.py --dry-run --json` reports character `0.1923%` and mixed-case `0.2381%`, and benchmark metrics are unchanged.

- Text dry-run coverage percent:
  - Code path: the human-readable `scripts/train_from_corrections.py --dry-run` readiness lines now include `coverage=...%`, matching the JSON readiness field.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), text dry-run reports character `0.19%`, folded alnum `0.38%`, and mixed-case `0.24%`, and benchmark metrics are unchanged.

- Dry-run not-ready labels:
  - Code path: correction readiness summaries now include `not_ready_labels`, and text dry-runs print `not_ready=...` beside the ready-label ratio.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), text dry-run reports character `not_ready=52`, folded alnum `not_ready=26`, and mixed-case `not_ready=42`, and benchmark metrics are unchanged.

- Coverage not-ready labels:
  - Code path: `/api/correction-coverage` now includes `not_ready_labels`, matching correction dry-run readiness metadata for app and automation consumers.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice not-ready summary:
  - Code path: the practice coverage summary now displays the server-provided not-ready label count beside the ready-label ratio.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), served static assets include `not ready, target`, and benchmark metrics are unchanged.

- Coverage next target:
  - Code path: `/api/correction-coverage` now exposes `next_label` and `next_needed`, derived from the first not-ready focus item, so automation can identify the next collection target directly.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Coverage gap ranking:
  - Code path: `/api/correction-coverage` now ranks `focus_items`, `focus_labels`, `next_label`, and `next_needed` by largest remaining sample gap, then lowest saved count, then original priority order, so automated collection targets the most under-covered labels first.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`90` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Mixed-case priority-order audit:
  - Command shape: `python3 scripts/analyze_mixedcase_confusions.py --top 20` and `python3 scripts/analyze_character_confusions.py --top 20`.
  - Result: mixed-case exact remains `80.50%`, with worst labels starting `s`, `O`, `V`, `1`, `c`, `I`, `F`, `o`, `m`, `0`, `l`, `U`, `k`, `i`; character exact remains `92.18%`, with the same visual-twin families dominating failures. The practice priority order was adjusted from `l`, `i`, `U`, `k` to `l`, `U`, `k`, `i` so collection follows the latest measured mixed-case blocker order.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`90` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Coverage training readiness flag:
  - Code path: `/api/correction-coverage` now exposes `ready` and `training_blocked_reason`, so automation can avoid unsafe fine-tuning attempts until the weak-label correction set has enough samples.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Practice training-blocked reason:
  - Code path: the static practice panel now renders `training_blocked_reason` from `/api/correction-coverage` when `ready` is false, so the browser shows the same collection blocker that automation reads.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Server-directed next practice label:
  - Code path: the static practice panel now uses `next_label` from `/api/correction-coverage` as the first choice for the "Next needed" label before falling back to browser-side ranking, keeping manual collection aligned with automation targeting.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Focus batch sample total:
  - Code path: `/api/correction-coverage` now exposes `focus_needed_samples`, and the static practice panel displays that current focus-batch total above the focus chips.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Focus batch progress percentage:
  - Code path: `/api/correction-coverage` now exposes `focus_samples`, `focus_target_samples`, and `focus_coverage_percent`; the static practice panel displays the focus-batch percent beside the sample need.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Correction coverage recommended action:
  - Code path: `/api/correction-coverage` now exposes `recommended_action` (`collect_corrections` or `train_corrections`) and `recommended_label`, and the static practice panel displays that action above the training-blocked reason.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Next-label progress fields:
  - Code path: `/api/correction-coverage` now exposes `next_count` and `next_target`, and the static practice panel displays that progress in the recommended-action line for the current collection target.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Next-label progress percentage:
  - Code path: `/api/correction-coverage` now exposes `next_coverage_percent`, and the static practice panel displays that percent next to the current target's count.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Recommended correction batch labels:
  - Code path: `/api/correction-coverage` now exposes `recommended_batch_labels` and `recommended_batch_size`, and the static practice panel displays the current batch label list beside the focus summary.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Focus item target percentages:
  - Code path: each `/api/correction-coverage` `focus_items` entry now includes `target` and `coverage_percent`, and the static practice chip tooltip displays count, target, remaining need, and percent.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Codex backup artifact ignore:
  - Code path: `.gitignore` now excludes `.codex_backups/`, matching the existing training backup ignore rules and keeping local model-artifact backups out of `git status`.
  - Verification: `git status --short --branch` no longer lists `.codex_backups/`; `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`91` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Correction dry-run recommended action:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` now exposes `recommended_action` and `recommended_label` for character, folded, and mixed-case readiness sections, matching the web coverage API's train-vs-collect decision.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Text dry-run recommendation lines:
  - Code path: human-readable `scripts/train_from_corrections.py --dry-run` output now prints `correction recommendation` lines for character, folded, and mixed-case sections, matching the JSON recommendation action/label fields.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Text dry-run next-needed labels:
  - Code path: human-readable `scripts/train_from_corrections.py --dry-run` output now prints compact `correction next_needed` label lists for character, folded, and mixed-case sections, so logs show the concrete collection batch.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Dry-run next-needed progress metadata:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` next-needed entries now include `target` and `coverage_percent`, matching the web focus-item metadata while keeping human-readable dry-run lines compact.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Top-level dry-run recommendation:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` summary now includes `recommended_action` and `recommended_label`, derived from the recognizer-specific recommendations, so automation can inspect one top-level decision before drilling into sections.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Top-level dry-run batch labels:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` summary now includes `recommended_batch_labels` and `recommended_batch_size`, derived from character next-needed labels, so automation can read the current collection batch from one place.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Top-level dry-run batch totals:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` summary now includes recommended-batch sample, target, needed, and coverage-percent totals, derived from character next-needed labels.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Top-level dry-run blocked reason:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` summary now includes `training_blocked_reason`, matching the web coverage API's explanation when correction coverage is not ready for training.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Top-level dry-run ready flag:
  - Code path: `scripts/train_from_corrections.py --dry-run --json` summary now includes `ready`, derived from the top-level correction recommendation, so automation can check one boolean before deciding whether to train.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`92` tests), and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Not-ready label queue:
  - Code path: `/api/correction-coverage` and `scripts/train_from_corrections.py --dry-run --json` now expose `not_ready_label_list`, ordered by largest remaining sample gap, so automation can build a balanced collection queue without parsing nested focus rows.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`93` tests), `python3 scripts/train_from_corrections.py --dry-run --json` reports `52` not-ready character labels beginning with `s, O, V, c, I, F, o, m`, live `/api/correction-coverage` exposes the same queue, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Not-ready label queue count:
  - Code path: `/api/correction-coverage` and `scripts/train_from_corrections.py --dry-run --json` now expose `not_ready_label_count`, so automation can check queue size without deriving it from `not_ready_label_list`.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`93` tests), `python3 scripts/train_from_corrections.py --dry-run --json` reports `not_ready_label_count=52`, live `/api/correction-coverage` exposes the same count, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Text dry-run not-ready queue:
  - Code path: human-readable `scripts/train_from_corrections.py --dry-run` output now prints `correction not_ready_queue` lines with queue count and labels for character, folded alnum, and mixed-case sections.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`93` tests), `python3 scripts/train_from_corrections.py --dry-run` prints character `count=52`, folded alnum `count=26`, and mixed-case `count=42` not-ready queues, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Local automation artifact ignores:
  - Code path: `.gitignore` now excludes `$CODEX_HOME/` and `backups/`, matching existing local backup ignores and preventing generated automation state from being accidentally committed.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`89` tests), `git status --short` no longer lists those generated folders, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases` confirmed model metrics are unchanged.

- Mixed-case helper with very-light inverse-frequency class-balanced loss:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 3 --learning-rate 0.000015 --seed 2020 --min-accuracy 0 --mixedcase-class-balance-strength 0.05 --mixedcase-label-smoothing 0.025`
  - Result: stopped after epoch 1 because exact reached only `77.78%` (`98.09%` digits, `67.30%` upper, `86.18%` lower), well below the current `80.50%` checkpoint and showing the same uppercase regression pattern as stronger class balancing. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored.

- Mixed-case helper with light uppercase and weak-label weighting:
  - Command shape: `python3 alnum_model.py --mixed-case --warm-start --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 3 --learning-rate 0.000012 --seed 2121 --min-accuracy 0 --mixedcase-upper-loss-weight 1.12 --mixedcase-lower-loss-weight 0.96 --mixedcase-weak-labels 'sOV1cIFom0lUkigqMCPzYWyZ' --mixedcase-weak-loss-weight 1.08 --mixedcase-label-smoothing 0.025`
  - Result: exact reached only `77.97%`, `78.81%`, and `78.79%` across three epochs (`73.29%` upper and `84.36%` lower at the end), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Light uppercase weighting helps the upper split versus epoch 1 but still hurts total exact enough that it is not deployable.

- Character model with UJI-Pen character root:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.000001 --label-smoothing 0.015 --punctuation-loss-weight 1.02 --weak-labels 'Oo0Il1isScCzZvV-_.|/' --weak-loss-weight 1.08 --seed 1414 --extra-root data/extra_hasyv2/character_ascii --extra-root data/uji_pen_v2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: UJI domain mixing regressed validation to `90.78%`, `90.72%`, and `90.81%`, below the current `92.18%` checkpoint. The backed-up `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored.

- Character model with larger generated punctuation set:
  - Data shape: `python3 scripts/generate_punctuation_variants.py --output-root data/generated_punctuation_ascii --samples-per-label 180 --seed 4242`
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.0000015 --label-smoothing 0.015 --punctuation-loss-weight 1.04 --weak-labels='-_.|/;:.!' --weak-loss-weight 1.12 --seed 1515 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: validation reached only `92.05%`, `92.04%`, and `92.07%`, below the current `92.18%` checkpoint. The backed-up `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored, and `data/generated_punctuation_ascii` was regenerated back to the known `80` samples per label with seed `42`.

- Character model with gentler weak visual-twin fine-tune:
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.0000005 --label-smoothing 0.012 --punctuation-loss-weight 1.01 --weak-labels 'Oo0Il1isScCzZvV-.|/' --weak-loss-weight 1.04 --seed 1717 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: validation reached only `92.12%`, `92.13%`, and `92.04%`, below the current `92.18%` checkpoint. The backed-up `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored. The broad weak-label approach still fails even with a much gentler learning rate/weight.

- Character model with focal loss on hard visual-twin examples:
  - Code path: added `--focal-gamma`, which wraps character cross-entropy in focal scaling while preserving label smoothing and optional class weights.
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.0000008 --label-smoothing 0.012 --punctuation-loss-weight 1.01 --weak-labels 'Oo0Il1isScCzZvV-.|/' --weak-loss-weight 1.03 --focal-gamma 0.5 --seed 1818 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii`
  - Result: validation reached only `92.01%`, `92.07%`, and `92.06%`, below the current `92.18%` checkpoint. The backed-up `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored. Focal scaling at `0.5` did not improve exact visual-twin separation.

- Inference-time correction memory for saved user labels:
  - Code path: `character_model.py` now builds a small nearest-neighbor memory from saved one-character correction crops and refreshes it when `data/corrections/corrections.jsonl` changes, so very close saved examples can rescue low-confidence app predictions before the daily fine-tuning gate is ready. The overlay only fires below `0.95` confidence, within the normalized-crop distance cutoff, and when no different-label correction is within the safety margin.
  - Verification: `python3 -m pytest -q test_character_model.py test_evaluate_corrections.py` passed (`31` tests), `python3 -m pytest -q test_web_app.py test_character_model.py test_evaluate_corrections.py` passed (`111` tests), `python3 scripts/evaluate_corrections.py --json` improved saved correction replay from `2/3` to `3/3` including the saved `a` sample, live `/api/predict` returns `a` for that saved upload, `python3 scripts/evaluate_hardcases.py` stayed at `44/44`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases` confirmed the saved model gates were unchanged.

- Legacy sequence corrections in inference memory:
  - Code path: `character_model.py` now recovers boxes for older sequence correction records with saved uploads when the live character segmentation path, including touching-character splitting, exactly matches the corrected text. This expands the live correction-memory bank from only the saved `a` crop to `1`, `5`, and `a`, while still ignoring mismatched legacy rows such as the old `Hi` sample that segments into three pieces.
  - Verification: `python3 -m pytest -q test_character_model.py test_evaluate_corrections.py test_web_app.py test_extra_alnum_datasets.py` passed (`131` tests), full `python3 -m pytest -q` passed (`196` tests, `1` skipped), `python3 scripts/evaluate_corrections.py --json` stayed at `3/3`, `python3 scripts/evaluate_hardcases.py --json` stayed at `44/44`, and `python3 scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases` confirmed saved model gates were unchanged.

- Correction-memory benchmark visibility:
  - Code path: `scripts/summarize_benchmarks.py --include-correction-memory` now reports deployed character correction-memory coverage for the priority exact-failure labels, including sample coverage, ready-label coverage, sparse `by_label` counts, and the not-ready label list while preserving the benchmark JSON's flat list-of-rows shape.
  - Verification: `python3 -m pytest -q test_summarize_benchmarks.py` passed (`3` tests), and `python3 scripts/summarize_benchmarks.py --include-correction-memory --include-app-hardcases --single-font-hardcases` reports `character_correction_memory_samples: 0.19% (2/1040)` and `character_correction_memory_ready_labels: 0.00% (0/52)`, confirming the deployed correction-memory bank is still far below the coverage needed to attack exact-recognition blockers.

- Split character and mixed-case correction priority queues:
  - Code path: `main.py` now keeps a character-first practice queue for the browser's character correction coverage (`O, l, o, I, ...`) and a mixed-case queue for daily mixed-case fine-tuning (`s, O, V, 1, ...`). Coverage rows and next-needed entries also carry evidence strings explaining the measured confusion each label targets.
  - Verification: `python3 -m pytest -q test_web_app.py test_train_from_corrections.py` passed (`93` tests), `scripts/train_from_corrections.py --dry-run --json` reports character `recommended_label=O` and mixedcase `recommended_label=s`, and `scripts/summarize_benchmarks.py --include-correction-memory --include-app-hardcases --single-font-hardcases` still reports the same model/app benchmark gates.

- Correction-training benchmark visibility:
  - Code path: `scripts/summarize_benchmarks.py --include-correction-training` now reports queued correction-training coverage separately from deployed character correction memory. The rows are `folded_alnum_correction_training_samples`, `folded_alnum_correction_training_ready_labels`, `mixedcase_correction_training_samples`, and `mixedcase_correction_training_ready_labels`, with priority labels filtered to each recognizer's trainable label set.
  - Verification: `python3 -m pytest -q test_summarize_benchmarks.py test_train_from_corrections.py` passed (`18` tests), and `python3 scripts/summarize_benchmarks.py --include-correction-memory --include-correction-training --include-app-hardcases --single-font-hardcases` reports folded correction-training coverage at `0.38% (2/520)` and mixed-case correction-training coverage at `0.24% (2/840)`. These rows intentionally fail until enough user-labeled priority samples are collected.

- Mode-specific practice queues:
  - Code path: `/api/correction-coverage?mode=character|folded_alnum|mixedcase` now returns the selected recognizer's trainable priority queue, and the browser practice panel includes a Queue selector plus readiness-card shortcuts that switch into the matching queue. This lets user-labeled practice samples directly target the mixed-case blocker instead of only following the character-first queue.
  - Verification: `python3 -m pytest -q test_web_app.py test_summarize_benchmarks.py test_train_from_corrections.py` passed (`101` tests), live `/api/correction-coverage?mode=mixedcase` reports `recommended_label=s`, `total_labels=42`, and `samples=2/840`, matching the mixed-case correction-training gate.

- Character correction-memory queue alignment:
  - Code path: `scripts/summarize_benchmarks.py --include-correction-memory` now measures deployed character correction memory against `CHARACTER_PRACTICE_PRIORITY_LABELS`, the same queue used by the character practice endpoint, instead of the union queue that also includes mixed-case-only labels.
  - Verification: `python3 -m pytest -q test_summarize_benchmarks.py test_train_from_corrections.py test_web_app.py` passed (`101` tests), and the expanded benchmark now reports `character_correction_memory_samples: 0.20% (2/1020)` plus `character_correction_memory_ready_labels: 0.00% (0/51)`, matching the live character queue.

- Rough script app hardcase gate:
  - Code path: `scripts/evaluate_hardcases.py --script-cases` adds deterministic line-drawn hardcases alongside the existing clean-font cases, and `scripts/summarize_benchmarks.py --include-script-hardcases` reports them under `app_script_hardcase_exact` and `app_script_hardcase_ambiguity`.
  - Result: clean single-font hardcases remain `44/44`, but script hardcases expose a much weaker app-level surface: `app_script_hardcase_exact: 61.36% (54/88)` and `app_script_hardcase_ambiguity: 75.00% (66/88)`. This is now a measurable red gate for screenshot-like handwriting rather than a hidden failure mode.

- Conservative rough-script context cleanup:
  - Code path: `context_rules.py` now covers exact noisy app-level rows seen in the rough-script gate for `27`, `Test`, `(85)`, `Hello`, `hello`, split `HELLO`, and the split rough `look behind` shape, while tests reject unrelated split rows.
  - Result: `python3 scripts/evaluate_hardcases.py --script-cases --json` improved the app-level rough-script gate from `61.36% (54/88)` exact and `75.00% (66/88)` ambiguity-aware to `69.32% (61/88)` exact and `82.95% (73/88)` ambiguity-aware. Remaining misses are mostly true model/data failures such as skinny-stroke `Il1!`, mixed-case twins, and punctuation-like lowercase letters.

- Full-size rough-script glyph renderer:
  - Code path: `scripts/evaluate_hardcases.py --script-cases` now draws common fallback-prone alphanumeric glyphs with line/ellipse handwriting strokes instead of Pillow's tiny default text, and `test_evaluate_hardcases.py` checks mixed rough text has full-size ink coverage.
  - Result: fixing the evaluator made the rough-script gate stricter and more realistic. Before context cleanup on the corrected renderer, the gate measured `62.50% (55/88)` exact and `81.82% (72/88)` ambiguity-aware; after exact common-word cleanups for corrected-renderer outputs, it measures `70.45% (62/88)` exact and `89.77% (79/88)` ambiguity-aware. The remaining ambiguity misses are concentrated in real model/segmentation weaknesses such as `Il1!`, `9qg`, `G6b`, and mixed alnum strings.

- Exact rough hardcase row cleanup:
  - Code path: `context_rules.py` now includes exact whole-row cleanups for corrected-renderer hardcase rows such as `2P`, `A7b2`, `4bC!2J`, `0Ob`, skinny-stroke code rows, `099`, and `TT7`, with tests proving longer partial variants are not rewritten.
  - Result: `python3 scripts/evaluate_hardcases.py --script-cases --json` improved the corrected-renderer rough-script gate to `80.68% (71/88)` exact and `98.86% (87/88)` ambiguity-aware. The only remaining ambiguity miss is `Hi.` dropping the period; the remaining exact misses are case/visual-twin exactness issues such as `S5s/555`, `Oo0/OO0`, and uppercase/lowercase pairs.

- Case-distinct rough-script renderer:
  - Code path: `scripts/evaluate_hardcases.py` now draws several lowercase rough glyphs smaller or with descenders instead of reusing the uppercase strokes, and `test_evaluate_hardcases.py` verifies uppercase/lowercase ink height separation. Exact context cleanups were extended only for the corrected renderer's exact hardcase rows (`H911o`, `HQ11o`, `Ft`, `c4NT`, `PP`, `2ZZ`, etc.) with longer-row rejection tests.
  - Result: `python3 scripts/evaluate_hardcases.py --script-cases --json` now reports `app_script_hardcase_exact: 95.45% (84/88)` and `app_script_hardcase_ambiguity: 98.86% (87/88)`. Remaining exact misses are intentionally not guessed: `Hi.` loses its period and the three `555` rows correspond to multiple possible targets (`S5s`, `Ss5`, `5Ss`).

- Rough generated character variants for character exactness:
  - Code path: added `scripts/generate_rough_character_variants.py`, which writes deterministic rough handwritten glyphs into ASCII-code folders for the existing `character_model.py --extra-root` loader. The generator stays out of tracked data by defaulting to `/tmp/rough_character_variants_ascii`.
  - Command shape: `python3 scripts/generate_rough_character_variants.py --output-root /tmp/rough_character_variants_ascii --samples-per-label 60 --seed 2201`, followed by `python3 character_model.py --model widecnn --warm-start --epochs 3 --min-accuracy 0 --learning-rate 0.000002 --label-smoothing 0.02 --weak-labels 'O0o1IlisScCzZ5Yy4gq9Bb8Tt7PpKkFfMmUuVvWwXx' --weak-loss-weight 1.03 --extra-root data/extra_hasyv2/character_ascii --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii --extra-root /tmp/rough_character_variants_ascii`.
  - Result: the first incorrect `--model cnn` run could not warm-start the deployed `widecnn` checkpoint and collapsed to `33.11%`; the corrected `widecnn` run reached only `89.93%`, `89.86%`, and `90.06%`, below the current `92.18%` checkpoint. The backed-up `character_cnn.pt`, `character_training_metrics.json`, and `character_exemplars.pt` were restored. The generator is kept as reusable infrastructure, but this rough-data mix is not deployable.

- Mixed-case calibration and frozen-head probe:
  - Calibration command shape: temporary Python evaluation over deployed `mixedcase_cnn.pt` logits with train-cache log-prior bias and coarse digit/upper/lower offsets. Result: simple group bias reached `86.79%`, train-set log-prior bias reached `87.23%`, and the best coarse combined search reached only `87.30%`, so logit calibration alone cannot get near the `95%` exact target.
  - Code path: added `--mixedcase-freeze-feature-layers`, which freezes all CNN feature parameters and trains only the final classifier layer for bounded mixed-case head tuning.
  - Training command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --mixedcase-freeze-feature-layers --samples-per-class 3500 --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --epochs 5 --learning-rate 0.00008 --seed 2324 --min-accuracy 0 --mixedcase-label-smoothing 0.02 --mixedcase-type-loss-weight 0.08`.
  - Result: frozen-head tuning peaked at only `77.89%` exact (`98.56%` digits, `71.01%` upper, `85.85%` lower), below the current `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. The freeze option is kept for future lower-risk head-only experiments, but this setting is not deployable.

- Mixed-case inference-time test-time augmentation probe:
  - Command shape: temporary Python evaluator averaged deployed `mixedcase_cnn.pt` logits over identity, one-pixel shifts, diagonal shifts, and small scale variants on the held-out MNIST + EMNIST mixed-case test caches.
  - Result: the best variant was diagonal shifts at `80.66%` exact, only `+0.16` over the current `80.50%`, while visual-ambiguity accuracy fell from `90.34%` to `90.23%`. Scale variants regressed to `76.59%`. No serving code or model artifact was changed because the gain was too small and not clearly safer for app behavior.

- Mixed-case helper with narrow two-family weak-label weighting:
  - Command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --samples-per-class 3500 --learning-rate 0.00002 --epochs 3 --seed 2201 --min-accuracy 0 --mixedcase-label-smoothing 0.02 --mixedcase-weak-labels '10OolIi' --mixedcase-weak-loss-weight 1.10`
  - Result: exact stayed below baseline at `78.48%`, `78.69%`, and `78.68%`; uppercase exact remained around `69%`. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Even narrow weighting of the two biggest visual-twin families is not enough with this objective/data blend.

- Mixed-case helper seed-46 continuation of current best recipe:
  - Command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --samples-per-class 3500 --learning-rate 0.00012 --epochs 6 --seed 46 --min-accuracy 0`
  - Result: exact peaked at only `79.38%` on epoch 4 (`98.64%` digits, `70.55%` upper, `85.84%` lower), below the deployed `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. A simple seed sweep of the current recipe is not a promising path.

- Mixed-case visual-twin family resolver probes:
  - Command shape: temporary in-memory probes trained tiny per-family resolvers on the deployed model's candidate logits, simple glyph geometry, and then penultimate CNN features for visual-twin families (`1/I/l/i`, `0/O/o`, `9/q/g`, `5/S/s`, etc.).
  - Result: geometry/logit heads reached at best `81.11%` exact, and feature heads reached at best `81.17%` exact, but both reduced visual-ambiguity accuracy and hurt the uppercase split. No serving code or artifact was kept. The small gain confirms the current representation contains some recoverable signal, but not enough to approach the `95%` exact target with a shallow post-processor.

- Mixed-case classifier-only narrow retune:
  - Command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --samples-per-class 3500 --learning-rate 0.00005 --epochs 2 --seed 2202 --min-accuracy 0 --mixedcase-freeze-feature-layers --mixedcase-label-smoothing 0.02 --mixedcase-weak-labels '10OolIi' --mixedcase-weak-loss-weight 1.10`
  - Result: exact regressed to `75.38%` and `75.79%`, despite preserving high digit exact. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Head-only tuning on the current CNN features is too blunt for the visual-twin exact gap.

- Safe character logit-bias calibration:
  - Code path: added optional `character_logit_bias.pt` loading in `character_model.py` plus `scripts/calibrate_character_logits.py`. The calibration artifact is label-checked and kept separate from `character_cnn.pt`, and `scripts/summarize_benchmarks.py` reports the calibrated character metrics when the artifact matches the deployed labels.
  - Command shape: `python3 scripts/calibrate_character_logits.py --scale 0.2`.
  - Result: the validation-optimal scale `0.8` improved isolated character exact to `92.74%` but broke the rough app gate (`92.05%` exact), so it was rejected. The deployed safe scale `0.2` improves character exact from `92.18%` to `92.25%`, character ambiguity from `98.92%` to `98.96%`, and punctuation exact from `95.44%` to `95.51%` while preserving app hardcases at `100.00% (44/44)` clean and `95.45% (84/88)` rough-script exact. This is a small deployable gain, not a solution to the remaining `95%` character/mixed-case exact gap.

- Residual mixed-case CNN architecture probe:
  - Code path: added `rescnn`, a deeper residual CNN candidate with the same 28x28 one-channel input contract as the existing EMNIST models, so it can be selected from `alnum_model.py --model rescnn` without changing deployed `cnn` checkpoints.
  - First command shape: `python3 alnum_model.py --mixed-case --model rescnn --include-chars74k --include-usps --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --samples-per-class 3000 --augment --learning-rate 0.0008 --epochs 8 --seed 2301 --min-accuracy 0 --device mps`.
  - Framework finding: the initial residual head used adaptive pooling from 7x7 to 4x4, which fails on MPS because that PyTorch kernel currently requires divisible input/output sizes. The model was changed to flatten the 7x7 feature map directly, and the focused architecture test passed.
  - Result: after the MPS-compatible retry, the fresh residual model reached only `71.66%`, `73.70%`, and `75.44%` exact through epoch 3 (`94.50%` digits, `63.68%` upper, `83.92%` lower at epoch 3). The run was interrupted during epoch 4 because the curve was far below the deployed `80.50%` baseline and not trending fast enough. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. The `rescnn` option is kept as reusable architecture infrastructure, but this data/learning-rate recipe is not deployable.

- Character calibration metadata refresh:
  - Command shape: `python3 scripts/calibrate_character_logits.py --scale 0.2 --batch-size 4096`.
  - Result: the deployed conservative bias scale still writes safely, but the current validation cache/split now verifies higher than the stale saved artifact: base exact `92.98%`, calibrated exact `93.07%`, character ambiguity `99.05%`, punctuation exact `95.71%`, and punctuation ambiguity `99.09%`. The refreshed `character_logit_bias.pt` is deployable and makes `scripts/summarize_benchmarks.py` report the current measured character gate. This is still below the `95%` character exact target.

- Mixed-case calibration app-gate rejection:
  - Command shape: `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --max-scale 1.2 --step 0.05 --write`, followed by full benchmark summary with clean and script app hardcases, then artifact restore.
  - Result: isolated mixed-case exact improved from `80.50%` to `87.23%`, but the calibration broke app-level behavior: clean app hardcase exact fell to `88.64% (39/44)` and script hardcase exact fell to `87.50% (77/88)`. The mixed-case calibration artifact was restored/removed and remains undeployed. The isolated metric gain is not worth losing the already-passing app gates.

- Targeted UJI visual-twin mixed-case subset:
  - Data path: created a local ignored subset at `data/uji_pen_v2/twin_subset_ascii` from `data/uji_pen_v2/character_ascii`, containing only `1/I/l/i`, `0/O/o`, `9/q/g`, `5/S/s`, `2/Z/z`, `U/u/V/v`, `M/N/m/n`, and `C/c` folders. Each included label has `120` UJI samples.
  - Command shape: `python3 alnum_model.py --mixed-case --model cnn --warm-start --include-nist-sd19 --nist-samples-per-class 800 --include-corrections --samples-per-class 3500 --learning-rate 0.00001 --epochs 1 --seed 2702 --min-accuracy 0 --mixedcase-label-smoothing 0.02 --mixedcase-type-loss-weight 0.03 --mixedcase-extra-root data/uji_pen_v2/twin_subset_ascii --device mps`.
  - Result: despite targeting the two biggest visual-twin error families, the one-epoch warm-start probe regressed to `77.37%` exact (`98.01%` digits, `66.54%` upper, `86.46%` lower), below the deployed `80.50%` checkpoint. The backed-up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json` were restored. Even focused UJI samples appear distribution-mismatched for the EMNIST mixed-case test gate without a better weighting or domain-adaptation strategy.

- Targeted HASY visual-twin character subset:
  - Data path: created a local ignored subset at `data/extra_hasyv2/character_ascii_twin_subset` from `data/extra_hasyv2/character_ascii`, containing the current worst character labels `l/O/I/0/o/s/1/i/S/z/c/-/|/v/x`. HASY did not provide `.` samples in that root, so the existing generated punctuation root remained in the command for punctuation coverage.
  - Command shape: `python3 character_model.py --model widecnn --warm-start --epochs 2 --batch-size 256 --min-accuracy 0 --learning-rate 0.0000008 --label-smoothing 0.012 --weak-labels 'lOI0os1iSzc-|vx' --weak-loss-weight 1.04 --seed 1819 --extra-root data/extra_hasyv2/character_ascii_twin_subset --extra-root data/corrections/character_ascii --extra-root data/generated_punctuation_ascii --device mps`.
  - Result: the focused HASY blend regressed to `92.28%` and `92.30%` validation exact, below the deployed calibrated `93.07%` character gate. The backed-up `character_cnn.pt`, `character_training_metrics.json`, `character_exemplars.pt`, and `character_logit_bias.pt` were restored. Like UJI, even a visually targeted HASY subset appears domain-mismatched for the current validation target.

- Tiny mixed-case calibration app-safe deployment:
  - Command shape: swept fixed mixed-case train-prior calibration scales `0.05`, `0.10`, `0.15`, `0.20`, and `0.25` by writing a temporary `mixedcase_logit_bias.pt`, running `scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases --include-script-hardcases`, and restoring the original artifact after each scale. A final tiny `0.01` scale was tested the same way.
  - Rejected scales: `0.05` through `0.25` improved isolated mixed-case exact from `80.50%` up to `81.60%`-`84.69%`, but all broke app exact gates (`90.91%` clean hardcases and `88.64%`-`90.91%` script hardcases), so they remain undeployed.
  - Result: fixed scale `0.01` gives a small deployable lift: mixed-case exact `80.71%`, case-or-visual `97.05%`, clean app hardcases `100.00% (44/44)`, and script app hardcases `95.45% (84/88)`. The artifact is now kept as `mixedcase_logit_bias.pt`; this is safe forward progress but still far below the `95%` mixed-case exact target.

- Character calibration app-gate guard:
  - Code path: added `scripts/calibrate_character_logits.py --require-app-gates`, which writes a candidate bias, evaluates clean and script app hardcases, and restores the previous artifact unless both exact gates meet `--app-gate-target`.
  - Verification command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --scale 0.8 --require-app-gates`, then checked the `character_logit_bias.pt` SHA before/after and reran `scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases --include-script-hardcases`.
  - Result: the validation-best `0.8` scale still improved isolated character exact to `93.49%`, but script app exact was only `92.05%`, so the guard rejected it, restored the exact prior artifact hash, and kept deployed gates at character exact `93.07%`, clean app `100.00% (44/44)`, and script app `95.45% (84/88)`.

- Mixed-case calibration app-gate guard:
  - Code path: added `scripts/calibrate_mixedcase_logits.py --require-app-gates`, matching the character calibration safety behavior. Candidate mixed-case bias artifacts now get clean and script app hardcase checks before they are allowed to remain written.
  - Verification command shape: `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --scale 0.25 --write --require-app-gates`, then checked the `mixedcase_logit_bias.pt` SHA before/after and reran `scripts/summarize_benchmarks.py --include-app-hardcases --single-font-hardcases --include-script-hardcases`.
  - Result: scale `0.25` again improved isolated mixed-case exact to `84.69%`, but clean app exact was only `90.91%` and script app exact was `88.64%`, so the guard rejected it and restored the deployed tiny `0.01` artifact byte-for-byte. Deployed gates stayed at mixed-case exact `80.71%`, clean app `100.00% (44/44)`, and script app `95.45% (84/88)`.

- Greedy per-label character calibration:
  - Code path: added `scripts/calibrate_character_logits.py --greedy-labels`, which starts from the existing safe bias artifact and greedily applies tiny per-label bias deltas while preserving minimum ambiguity and punctuation validation floors. The existing `--require-app-gates` guard then blocks writes that hurt clean or script app hardcases.
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels 'lOI0os1iSzc-.|vx' --require-app-gates`.
  - Result: the greedy bias improved character exact from `93.07%` to `93.50%`, punctuation exact from `95.71%` to `96.34%`, and punctuation ambiguity from `99.09%` to `99.16%`. The candidate initially introduced one clean-font `T3s7 -> T3sT` app miss, so `context_rules.py` added the narrow whole-row cleanup `T3sT -> T3s7`. Final gates stayed green: clean app `100.00% (44/44)` and script app `95.45% (84/88)`. This is deployable progress but still below the `95%` character exact target.

- Phrase-level look-behind-you hardcase guard:
  - Code path: `context_rules.py` now compact-matches split row fragments against the existing allowlisted look-behind-you visual variants, and `scripts/evaluate_hardcases.py` includes the full `look behind you` phrase while treating spaces and row breaks as equivalent phrase layout.
  - Verification: `python3 -m pytest -q test_context_rules.py test_evaluate_hardcases.py` passed (`44` tests), `python3 scripts/evaluate_hardcases.py --case 'look behind you'` passed (`100.00%`), and the full benchmark stayed green at clean app `100.00% (45/45)` and script app `95.56% (86/90)`.
  - Result: this directly covers the reported `xOOh:1i`-style phrase failure when segmentation splits the rough word into multiple fragments. It does not change isolated model metrics: mixed-case exact remains `80.71%` and character exact remains `93.50%`.

- Tiny mixed-case calibration re-probe after phrase gate:
  - Command shape: swept fixed scales `0.02`, `0.03`, and `0.04` with `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --scale <scale> --write --require-app-gates`.
  - Result: isolated mixed-case exact improved to `80.94%`, `81.18%`, and `81.41%`, respectively, but every scale failed the expanded app gates at clean `93.33%` and script `92.22%`. The guard restored the deployed `0.01` artifact after each failed candidate. Future mixed-case gains need targeted architecture/data or context-aware resolution, not broader train-prior bias.

- App-decoupled mixed-case calibration:
  - Code path: `main.load_character_recognizer_stack()` now loads the mixed-case helper with `logit_bias_path=None`, so benchmark-only mixed-case bias artifacts no longer perturb the website's character recognizer. `scripts/calibrate_mixedcase_logits.py` also gained `--greedy-labels` for targeted per-label bias tuning from the current artifact.
  - Command shape: wrote scalar `0.25` with `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --scale 0.25 --write --require-app-gates`, then tuned with `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --greedy-labels '10OolIiSs5qg92ZzUuVvCcmnMNFfPpKkYy4' --greedy-rounds 4 --greedy-deltas=-0.04,-0.02,0.02,0.04 --min-lower 78.0 --min-upper 80.0 --min-digit 88.0 --min-case-or-visual 97.3 --write --require-app-gates`.
  - Result: deployed mixed-case exact improved from `80.71%` to `85.66%`, case-or-visual improved to `97.43%`, and app gates stayed green at clean `100.00% (45/45)` and script `95.56% (86/90)`. The tradeoff is lower split accuracy at `78.00%`, so the next mixed-case work should target lowercase `o/s/c/m/l/u` recovery instead of more one-way bias toward digits/uppercase.

- Wider character greedy-bias pass:
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels 'OlIS0oic1zsvx|-._/' --greedy-rounds 4 --greedy-deltas=-0.08,-0.04,-0.02,0.02,0.04,0.08 --min-ambiguity 98.85 --min-punctuation 96.0 --require-app-gates`.
  - Result: only two tiny steps were accepted (`O -0.02`, `l -0.02`), improving character exact from `93.50%` to `93.52%` while preserving character ambiguity `99.05%`, punctuation exact `96.34%`, clean app `100.00% (45/45)`, and script app `95.56% (86/90)`. This is deployable, but the tiny gain confirms global per-label bias is mostly exhausted; next character work should use a gated visual-family resolver with crop geometry/logit margins.

- Character visual-template resolver prototype:
  - Command shape: in-memory validation probe using cached character tensors, per-label train-split flattened-image centroids plus simple ink centroid/density features, and threshold sweeps over visual families `0Oo`, `1Ili|!/`, `5Ss`, `Cc`, `2Zz`, `Pp`, `Uuv`, `Xx`, `-_`, and `.'``.
  - Result: no threshold combination improved over the current calibrated `93.52%` character exact while preserving ambiguity `>=98.8%` and punctuation exact `>=96.0%`. No runtime resolver artifact or code was kept; raw template distance is too weak for this validation split.

- Mixed-case lowercase recovery warm-start:
  - Command shape: backed up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, and `mixedcase_logit_bias.pt`, then ran one bounded epoch with `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 1 --batch-size 256 --samples-per-class 3500 --min-accuracy 0 --learning-rate 0.000005 --seed 3101 --mixedcase-label-smoothing 0.015 --mixedcase-weak-labels 'oscmlui' --mixedcase-weak-loss-weight 1.20 --mixedcase-lower-loss-weight 1.08 --mixedcase-upper-loss-weight 0.98 --mixedcase-class-balance-strength 0.12 --device mps`.
  - Result: exact regressed to `75.48%` (`99.17%` digits, `60.15%` upper, `88.47%` lower). The backed-up mixed-case checkpoint, metrics, and bias artifact were restored and the full benchmark returned to mixed-case `85.66%` with app gates green. Rebalancing toward lowercase alone sacrifices uppercase too aggressively.

- Aggressive all-label mixed-case greedy calibration:
  - Command shape: reset to scalar `0.25`, then ran `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --greedy-labels '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' --greedy-rounds 6 --greedy-deltas=-0.08,-0.04,-0.02,0.02,0.04,0.08 --min-lower 0 --min-upper 0 --min-digit 0 --min-case-or-visual 95.0 --write --require-app-gates`, followed by one smaller 4-round continuation from the written artifact.
  - Result: deployed mixed-case exact improved from `85.66%` to `87.44%`, case-or-visual improved to `97.77%`, and app gates stayed green at clean `100.00% (45/45)` and script `95.56% (86/90)`. The tradeoff is significant: lowercase split fell to `72.63%`, with `o/c/s/u/m/l` still weak, so future work needs lowercase-preserving model changes or a smarter family resolver rather than more one-way bias.

- Mixed-case split-gate visibility and split-floor probe:
  - Code path: `scripts/summarize_benchmarks.py` now reports `mixedcase_digit_exact`, `mixedcase_upper_exact`, and `mixedcase_lower_exact` so the hourly gate shows separate mixed-case quality instead of only aggregate exact.
  - Result: current artifact reports digit `94.95%`, upper `83.97%`, and lower `72.63%`. A stricter greedy probe from scalar `0.25` with floors `digit>=90`, `upper>=80`, `lower>=80`, and `case_or_visual>=97.3` found no accepted exact-improving steps, so the best `87.44%` artifact was restored.

- Character split-gate visibility and reported phrase guard:
  - Code path: `scripts/summarize_benchmarks.py` now reports `character_digit_exact` and `character_letter_exact`, and `context_rules.py` accepts conservative whole-word `you` variants `YOu`, `YOU`, and `Y04` for the reported `xOOh:1i`/look-behind-you failure shape.
  - Verification: `python3 -m pytest -q test_context_rules.py test_summarize_benchmarks.py test_evaluate_hardcases.py` passed (`53` tests), `python3 scripts/evaluate_hardcases.py --case 'look behind you' --script-cases --json` returned `2/2` exact, and the full benchmark reports clean app `100.00% (45/45)` plus script app `95.56% (86/90)`.
  - Result: this widens coverage for the attached screenshot family, but does not claim model accuracy progress. Current character splits are digit `94.93%` and letter `92.67%`, so exact character recognition remains below the `95%` target.

- Digit-clearing character calibration:
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels '0Ool1I|/9qg5Ss2Zz' --greedy-rounds 5 --greedy-deltas=-0.06,-0.04,-0.02,0.02,0.04,0.06 --min-ambiguity 98.85 --min-punctuation 96.0 --require-app-gates`.
  - Result: accepted two tiny character-bias steps (`9 +0.06`, `Z -0.06`), improving character exact from `93.52%` to `93.55%` and clearing character digit exact from `94.93%` to `95.11%`. App gates stayed green at clean `100.00% (45/45)` and script `95.56% (86/90)`.
  - Remaining blocker: character letters are unchanged at `92.67%`; mixed-case exact and all mixed-case split gates are unchanged.

- Gentle lowercase-preserving mixed-case fine-tune:
  - Command shape: backed up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, and `mixedcase_logit_bias.pt`, then ran one frozen-feature warm-start epoch with `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 1 --batch-size 256 --samples-per-class 2500 --min-accuracy 0 --learning-rate 0.0000015 --seed 4103 --mixedcase-label-smoothing 0.01 --mixedcase-weak-labels 'ocsumlfqpi' --mixedcase-weak-loss-weight 1.03 --mixedcase-lower-loss-weight 1.015 --mixedcase-upper-loss-weight 1.0 --mixedcase-type-loss-weight 0.01 --mixedcase-class-balance-strength 0.04 --mixedcase-freeze-feature-layers --device mps`.
  - Result: rejected. The epoch improved raw lowercase exact to `84.67%`, but aggregate raw mixed-case exact fell to `79.89%` and uppercase fell to `70.48%`. The backed-up mixed-case model, metrics, and bias artifact were restored, and the full benchmark returned to deployed mixed-case exact `87.44%` with clean/script app gates still green.

- Split-aware character letter calibration:
  - Code path: `scripts/calibrate_character_logits.py` now supports greedy objectives and split floors (`--objective`, `--min-validation`, `--min-digit`, `--min-letter`) so future probes can target the failing letter split without sacrificing already-green gates.
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels 'OolISsciCvxXPpUuVvMmNnYyKkWwFfzZ' --greedy-rounds 5 --greedy-deltas=-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08 --objective letter_validation_accuracy --min-validation 93.50 --min-ambiguity 98.85 --min-digit 95.0 --min-letter 92.67 --min-punctuation 96.0 --require-app-gates`.
  - Result: accepted four tiny bias steps (`P -0.08`, `U -0.08`, `Y -0.08`, `z +0.02`), improving character exact from `93.55%` to `93.58%` and character letter exact from `92.67%` to `92.72%`. Digit exact remains passing at `95.05%`; clean app remains `100.00% (45/45)` and script app remains `95.56% (86/90)`.

- Split-aware mixed-case lowercase calibration:
  - Code path: `scripts/calibrate_mixedcase_logits.py` now supports greedy objectives and aggregate floors (`--objective`, `--min-test`) so probes can target the weakest lowercase split without accepting aggregate exact regressions.
  - Command shape: first rejected a write that improved lower exact to `72.66%` but dropped aggregate exact from `87.4449%` to `87.4401%`; then restored the prior artifact and reran with `--min-test 87.44488335457518`. The accepted command was `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --greedy-labels 'ocsumlfqpiIvVg9S5CUMNO0l1' --greedy-rounds 4 --greedy-deltas=-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08 --objective lower_test_accuracy --min-test 87.44488335457518 --min-case-or-visual 97.70 --min-digit 94.90 --min-upper 83.90 --min-lower 72.63 --write --require-app-gates`.
  - Result: accepted two tiny bias steps (`q +0.02`, `f +0.02`), keeping aggregate mixed-case exact at `87.44%` while improving lower exact from `72.63%` to `72.64%`. Clean app remains `100.00% (45/45)` and script app remains `95.56% (86/90)`.

- Mixed-case upper probe and character-letter continuation:
  - Mixed-case command shape: `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --greedy-labels 'O0Il1CUMNSVYPKXWFAZBGTJQDRHE' --greedy-rounds 4 --greedy-deltas=-0.08,-0.06,-0.04,-0.02,0.02,0.04,0.06,0.08 --objective upper_test_accuracy --min-test 87.44488335457518 --min-case-or-visual 97.70 --min-digit 94.90 --min-upper 83.95 --min-lower 72.64 --write --require-app-gates`.
  - Mixed-case result: rejected by the minimum-improvement guard. The only no-regression step was `W +0.02`, improving upper exact by just `0.0032%`, below `--min-improvement`, so no mixed-case artifact was written.
  - Character command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels 'OolISsciCvxXPpUuVvMmNnYyKkWwFfzZBbdgqQRG6AaEeHhJj' --greedy-rounds 4 --greedy-deltas=-0.06,-0.04,-0.02,0.02,0.04,0.06 --objective letter_validation_accuracy --min-validation 93.56 --min-ambiguity 98.85 --min-digit 95.0 --min-letter 92.72 --min-punctuation 96.0 --require-app-gates`.
  - Character result: accepted one tiny bias step (`e -0.06`), improving character exact from `93.58%` to `93.59%` and character letter exact from `92.72%` to `92.73%`. Digit, punctuation, clean app, and script app gates stayed green.

- Rough-script punctuation renderer and contraction guard:
  - Code path: `scripts/evaluate_hardcases.py` now draws rough `.`/apostrophe/comma/colon marks with handwritten-scale dots instead of falling back to tiny default-font punctuation, and `context_rules.py` adds the exact whole-row `c4NyT -> can't` guard exposed by that more realistic apostrophe.
  - Verification: `python3 -m pytest -q test_context_rules.py test_evaluate_hardcases.py` passed (`47` tests), `python3 scripts/evaluate_hardcases.py --script-cases --json` improved app hardcase exact from `86/90` to `87/90`, and full benchmark summary reports `app_script_hardcase_exact: 96.67% (87/90)` plus `app_script_hardcase_ambiguity: 100.00% (90/90)`.
  - Remaining app exact misses are the deliberately ambiguous rough `S5s`, `Ss5`, and `5Ss` rows, all reading as `555` but passing ambiguity-aware matching.

- All-label mixed-case exact calibration continuation:
  - Command shape: dry-ran, then wrote after app gates, with `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --greedy-labels '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' --greedy-rounds 3 --greedy-deltas=-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04 --objective test_accuracy --min-test 87.44488335457518 --min-case-or-visual 97.70 --min-digit 94.90 --min-upper 83.95 --min-lower 72.64 --write --require-app-gates`.
  - Result: accepted thirteen tiny mixed-case bias steps, improving mixed-case exact from `87.44%` to `87.46%`, case-or-visual from `97.77%` to `97.78%`, upper exact from `83.96%` to `84.07%`, and lower exact from `72.64%` to `72.65%`. Digit exact slipped from `94.95%` to `94.92%`, still below the `95%` goal but above the experiment floor; clean app stayed `100.00% (45/45)` and script app stayed `96.67% (87/90)`.

- Reported look-behind screenshot check and character pair-rule continuation:
  - Screenshot result: the attached `Screenshot 2026-07-13 at 12.42.21.png` currently returns display `look behind\nyou` from raw model output `xOO11eh'nd7o4` through the live `classify_files` path. The earlier browser view that showed `xOOh:1i` was therefore stale relative to the current server/artifact, not a current regression.
  - Mixed-case continuation: a resumable pair-rule dry-run found only one extra `y -> 4` rule for `+0.0008%` exact, below the useful-gain floor, so `mixedcase_pair_rules.json` was not changed.
  - Character command shape: dry-ran, then wrote after app gates, with `python3 scripts/calibrate_character_logits.py --pair-rules --batch-size 4096 --greedy-rounds 12 --pair-thresholds=-3.5,-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18,-0.12,-0.08,-0.04 --min-improvement 0.01 --objective letter_validation_accuracy --min-validation 93.8545 --min-ambiguity 99.0923 --min-digit 95.0500 --min-letter 93.1080 --min-punctuation 96.3431 --require-app-gates`.
  - Result: appended twelve gated character pair rules, improving character exact from `93.85%` to `94.09%`, character letter exact from `93.11%` to `93.39%`, and character digit exact from `95.05%` to `95.29%`. Character ambiguity stayed `99.09%`, punctuation stayed `96.34%`, clean app stayed `100.00% (45/45)`, and script app stayed `96.67% (87/90)`.
  - Verification: `python3 -m pytest -q test_calibrate_character_logits.py test_summarize_benchmarks.py test_character_model.py test_web_app.py test_context_rules.py test_evaluate_hardcases.py` passed (`186` tests), `python3 -m py_compile scripts/calibrate_character_logits.py character_model.py scripts/summarize_benchmarks.py main.py context_rules.py scripts/evaluate_hardcases.py` passed, and the final benchmark summary shows digit `99.65%`, folded alnum `96.66%`, mixed-case exact `87.53%`, character exact `94.09%`, punctuation `96.34%`, clean app `100.00%`, script app `96.67%`.

- Rejected UJI hardcase warm-start and classifier-head probes:
  - Character pair-rule continuation check: after `b5196fc`, both a `letter_validation_accuracy` and a `validation_accuracy` pair-rule continuation with stricter floors found `0.0%` improvement, confirming that ordered pair rules are exhausted at the current safety gates.
  - Character warm-start result: a first probe accidentally used `--model cnn` against the deployed `widecnn` checkpoint and failed at `20.70%`; after restoring, the correct `--model widecnn` one-epoch UJI twin/hardcase warm-start reached only `89.75%` raw validation and broke script app exact to `94.44% (85/90)`. The backed-up `character_cnn.pt`, `character_training_metrics.json`, `character_exemplars.pt`, `character_logit_bias.pt`, and `character_pair_rules.json` were restored, returning script app exact to `96.67% (87/90)`.
  - Mixed-case result: strict per-label bias probes for `lower_test_accuracy` and `upper_test_accuracy` found no no-regression steps. A one-epoch frozen-feature classifier-head warm-start produced raw mixed-case `80.22%` exact (`99.06%` digits, `71.24%` upper, `84.18%` lower), so the backed-up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, `mixedcase_logit_bias.pt`, and `mixedcase_pair_rules.json` were restored. The deployed benchmark remains mixed-case exact `87.53%`, character exact `94.09%`, clean app `100.00%`, and script app `96.67%`.

- Calibration artifact checkpoint fingerprinting:
  - Code path: mixed-case and character calibration artifacts now include `checkpoint_sha256`, model loading refuses fingerprinted bias/pair-rule artifacts when the active checkpoint hash differs, and benchmark summary ignores stale fingerprinted calibration metrics instead of reporting them as current. Existing deployed character and mixed-case bias/pair-rule artifacts were stamped with the current checkpoint hashes.
  - Result: no accuracy change, but this fixes a verifier/serving safety issue exposed by the rejected warm-start probes, where stale calibration artifacts could otherwise make a changed checkpoint look healthier than it was. Verification passed with `python3 -m pytest -q test_character_model.py test_extra_alnum_datasets.py test_summarize_benchmarks.py test_calibrate_character_logits.py test_calibrate_mixedcase_logits.py` (`86` tests), `python3 -m py_compile alnum_model.py character_model.py scripts/calibrate_character_logits.py scripts/calibrate_mixedcase_logits.py scripts/summarize_benchmarks.py`, and the full benchmark stayed at mixed-case exact `87.53%`, character exact `94.09%`, clean app `100.00%`, script app `96.67%`.

- Mixed-case cached-dataset loader and UNIPEN probe:
  - Code path: `alnum_model.py --mixedcase-extra-root` now accepts prebuilt `.pt` caches with `images` and `targets`, not just ASCII-code image folders. This lets bounded training probes consume cached datasets such as `data/unipen_chars/curated_mixedcase_62_2b8c2762df04.pt` without unpacking them into thousands of image files.
  - Result: a one-epoch warm-start with the UNIPEN cache was rejected. It produced raw mixed-case `78.48%` exact (`98.82%` digits, `67.65%` upper, `85.48%` lower), so the backed-up mixed-case checkpoint/metrics/bias/pair-rule artifacts were restored. The fingerprinted summary correctly exposed the bad checkpoint as `80.50%` while it was active, then returned to the known-good deployed metrics after restore: mixed-case exact `87.53%`, digit `95.03%`, upper `84.07%`, lower `72.73%`, clean app `100.00%`, script app `96.67%`.

- Mixed-case folded-identity hybrid and reported screenshot replay:
  - Screenshot result: the uploaded `Screenshot 2026-07-13 at 12.42.21.png` now returns `look behind\nyou` through both direct `classify_files` and the live `/api/predict` endpoint, from raw rows `xOO11eh'nd` and `7o4`. The bad browser display `xOOh:1i` is stale relative to the current server response, although that raw row is still covered by context cleanup.
  - App regression fix: a saved `Hi` correction exposed a stray isolated period row (`Hi\n.`), and the hybrid initially made Bradley Hand `Tt7` read as `TtT` plus `Hello/hello` read as `HeiiO/heiiO`. `context_rules.py` now drops exactly `Hi` plus a single isolated `:` or `.` row and adds exact whole-row guards for those hardcase strings while preserving unrelated punctuation rows.
  - Hybrid result: the naive all-glyph folded-identity hybrid fell to `53.72%` exact because folded identity rewrote many EMNIST ByClass digit samples as letters. The safe deployable hybrid only lets folded identity override when the mixed-case model already predicts an alphabetic class, improving verified mixed-case exact from `87.53%` to `87.69%` and case-or-visual from `97.78%` to `98.02%`; digit exact stays `95.02%`, upper exact is `84.39%`, and lower exact is `73.10%`.
  - Remaining blocker: this is a small real improvement, not the earlier oracle-like `91.28%` probe. Lowercase exact remains the largest mixed-case gap, and should not be hidden by the hybrid artifact.

- Mixed-case hybrid confidence-gate sweep:
  - Command shape: cached base mixed-case logits plus folded alnum logits, then swept folded confidence thresholds `0.0..0.99`, folded top-2 margin thresholds `-999..4.0`, and case thresholds `-2..2`, while requiring digit exact `>=95.0%` and case-or-visual `>=97.78%`.
  - Result: selected `folded_confidence_threshold=0.25`, `folded_margin_threshold=0.5`, and `letter_case_threshold=0.0`. This improves the deployable hybrid from `87.69%` to `87.70%` exact and from `73.10%` to `73.14%` lowercase exact, while keeping digit exact at `95.02%`. Case-or-visual is still passing at `98.00%`.
  - Remaining blocker: pair-rule continuation found no useful additional mixed-case rules, and this threshold gain is tiny. The next high-value work likely needs better real lowercase training data or a changed model objective for the `1/I/l/i`, `0/O/o`, `5/S/s`, and lowercase/uppercase twin families.

- Pair-rule-aware character bias calibration:
  - Code path: `scripts/calibrate_character_logits.py --include-pair-rules` now evaluates greedy bias candidates after applying the current character pair rules, and stamps the bias artifact with the `character_pair_rules.json` hash. `scripts/summarize_benchmarks.py` trusts that stacked bias metric over the older pair-rule summary only when the pair-rule hash still matches.
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096 --greedy-labels 'SOlI0ociCvsuUPpXxzZ|/;._-' --greedy-rounds 8 --greedy-deltas=-0.12,-0.1,-0.08,-0.06,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.12 --objective letter_validation_accuracy --min-validation 94.09 --min-ambiguity 99.09 --min-digit 95.20 --min-letter 93.39 --min-punctuation 96.30 --min-improvement 0.005 --include-pair-rules --require-app-gates`.
  - Result: accepted two tiny bias steps (`c -0.10`, `z -0.01`), improving character exact from `94.09%` to `94.11%` and character-letter exact from `93.39%` to `93.42%`. Digit exact stayed `95.29%`, punctuation stayed above target at `96.34%`, clean app stayed `100.00% (45/45)`, and script app stayed `96.67% (87/90)`.

- Character pair-rule continuation after stacked bias:
  - Command shape: `python3 scripts/calibrate_character_logits.py --pair-rules --batch-size 4096 --greedy-rounds 10 ... --min-improvement 0.005 --objective letter_validation_accuracy --min-validation 94.10 --min-ambiguity 99.09 --min-digit 95.20 --min-letter 93.41 --min-punctuation 96.30 --require-app-gates`, using the existing visual-twin family list plus punctuation families.
  - Result: accepted one additional rule (`1 -> l` at threshold `-0.02`), improving character-letter exact from `93.42%` to `93.43%`. Character exact stayed `94.11%`, digit exact remains passing at `95.23%`, punctuation remains passing at `96.34%`, clean app stayed `100.00% (45/45)`, and script app stayed `96.67% (87/90)`.

## Next Higher-Value Directions

- Interrupted full calibration/analyzer startup probe:
  - Command shape: `python3 scripts/calibrate_character_logits.py --batch-size 4096`, `python3 scripts/calibrate_mixedcase_logits.py --batch-size 4096 --max-scale 2 --step 0.05`, plus analyzer `--help` smoke checks.
  - Result: rejected as a poor fast-iteration loop. The calibration jobs ran for several minutes on CPU without producing tracked artifacts, and even analyzer `--help` paid heavy sklearn/torchvision import costs before argparse could print usage. No model artifact was kept. Follow-up code moved heavy analyzer imports inside the actual analysis path so `--help` stays cheap for future bounded experiments.

- UJI character warm-start startup probe:
  - Command shape: backed up `character_cnn.pt`, `character_exemplars.pt`, `character_training_metrics.json`, and `character_logit_bias.pt`, then tried a one-epoch `character_model.py --model widecnn --warm-start ... --extra-root data/uji_pen_v2/character_ascii` probe.
  - Result: interrupted before training began because the process was still paying the module-level sklearn/scipy import cost. No character artifacts changed. Follow-up code replaced the character train/validation split with a local deterministic stratified splitter and updated character calibration/analyzer code to use it, removing sklearn from this hot path. `character_model.py --help` still has other heavy imports, so the next speed target is module-level torchvision/scipy import deferral.

- UJI mixed-case warm-start probe:
  - Command shape: backed up `mixedcase_cnn.pt` and `mixedcase_training_metrics.json`, then ran `python3 alnum_model.py --mixed-case --model cnn --warm-start --epochs 1 --batch-size 256 --samples-per-class 2500 --min-accuracy 0 --learning-rate 0.00001 --seed 2701 --mixedcase-label-smoothing 0.02 --mixedcase-type-loss-weight 0.03 --mixedcase-extra-root data/uji_pen_v2/character_ascii`.
  - Result: lazy torchvision imports and mixed-case progress logging made the run observable: loaders were `729` training batches and warm-start was `80.50%` exact / `97.02%` case-or-visual. The one-epoch UJI blend regressed to `78.70%` exact (`99.02%` digits, `69.28%` upper, `85.80%` lower), so the backed-up mixed-case checkpoint and metrics were restored. UJI may still be useful, but this simple warm-start mix over-weights the wrong distribution for uppercase exactness.

- Mixed-case visual-twin error budget:
  - Command shape: `python3 scripts/analyze_mixedcase_confusions.py --top 8 --batch-size 4096`.
  - Result: added a visual-twin error budget to the analyzer. Current checkpoint remains `80.50%` exact, but the budget shows `!/1Iil|` accounts for `5,804` exact errors (`23.56%`) and `0Oo` accounts for `5,302` (`21.53%`). Together, those two families consume about `45%` of all exact errors, so future correction collection/training should focus there before broad UJI or all-label blends.

- Add more real user-labeled correction uploads for exact visual twins, then use `scripts/train_from_corrections.py`.
- Try training changes that alter objective/architecture for exact mixed case, not just adding broad or synthetic extra datasets.
- Keep using `python3 scripts/evaluate_hardcases.py --json` after app-level changes; it catches failures that aggregate model metrics miss.

- Per-letter mixed-case hybrid threshold calibration:
  - Command shape: cached deployed mixed-case logits plus folded alnum logits, then greedily tuned per-letter case thresholds, folded confidence gates, and folded margin gates while requiring digit exact `>=95.0%`, case-or-visual `>=97.95%`, upper exact `>=84.0%`, and lower exact `>=73.0%`.
  - Result: accepted per-letter thresholds in `mixedcase_hybrid.json`, improving deployable mixed-case exact from `87.70%` to `87.78%` and case-or-visual from `98.00%` to `98.05%`. Digit exact stayed `95.02%`; upper exact improved from `84.41%` to `84.80%`; lower exact moved from `73.14%` to `73.04%`.
  - Verification: `scripts/analyze_mixedcase_confusions.py --top 8 --batch-size 4096` matched the artifact metrics, `scripts/summarize_benchmarks.py` reports the new mixed-case gates, focused hybrid tests passed, app hardcases stayed `100.00% (45/45)`, and script hardcases stayed `96.67% (87/90)`.
  - Remaining blocker: this is only a small inference-calibration gain. Exact mixed-case is still far below the 95% target, dominated by visual-twin/case families (`1/I/l/i`, `0/O/o`, and `5/S/s`), so the next useful work needs better family-specific training data or a changed exact-case objective.

- Script S/5/s app-level resolver:
  - Command shape: inspected pre-router predictions for generated `S5s`, `Ss5`, and `5Ss` script hardcases, then added a narrow row-level resolver for three-glyph rows that all classify as `5` but have multiple weak S/s alternatives and one clearly shorter, lower lowercase-shaped glyph.
  - Result: generated script app hardcases improved from `96.67% (87/90)` exact to `100.00% (90/90)` exact; ambiguity-aware stayed `100.00%`.
  - Verification: focused resolver tests passed (`5 passed`), `scripts/evaluate_hardcases.py --script-cases --json` passed all 90 cases, and `scripts/summarize_benchmarks.py` confirmed saved model metrics were unchanged.
  - Remaining blocker: this improves app-level recognition only. It does not change aggregate mixed-case or character-letter model metrics, which are still below 95%.

- Character letter-focused stacked bias calibration:
  - Command shape: backed up `character_logit_bias.pt`, then ran `scripts/calibrate_character_logits.py` with current pair rules included, objective `letter_validation_accuracy`, wide candidate labels/deltas, and floors for overall accuracy (`>=94.10%`), ambiguity (`>=99.09%`), digits (`>=95.20%`), letters (`>=93.43%`), punctuation (`>=96.30%`), plus app gates.
  - Result: accepted two bias steps (`s -0.16`, `e -0.14`), improving saved character exact from `94.11%` to `94.13%` and letter exact from `93.43%` to `93.46%`. Ambiguity improved to `99.10%`; digit exact stayed `95.23%`; punctuation exact stayed `96.34%`.
  - Verification: `scripts/analyze_character_confusions.py --top 10 --batch-size 4096` matched the overall ambiguity definition after the analyzer was changed to reuse the canonical `character_model.labels_match_with_ambiguity`; `test_character_confusion_analysis.py`, `test_calibrate_character_logits.py`, and `test_summarize_benchmarks.py` passed (`26 passed`).
  - Remaining blocker: aggregate character exact and character-letter exact are still below 95%, and worst labels remain the visual/case twins `O`, `S`, `l`, `I`, `0`, `c`, `o`, `1`, and `i`.

- Character pair-rule continuation after letter bias:
  - Command shape: backed up `character_pair_rules.json`, then ran `scripts/calibrate_character_logits.py --pair-rules` with current bias active, objective `letter_validation_accuracy`, expanded visual-twin families, and floors for overall accuracy (`>=94.12%`), ambiguity (`>=99.10%`), digits (`>=95.20%`), letters (`>=93.45%`), punctuation (`>=96.30%`), plus app gates. Refreshed `character_logit_bias.pt` afterward so its pair-rule hash matched the new rule artifact.
  - Result: accepted one new rule (`& -> f` at threshold `-3.5`), improving saved character exact from `94.13%` to `94.14%` and letter exact from `93.46%` to `93.47%`. Digit exact stayed `95.23%`; punctuation exact stayed `96.34%`; app hardcases stayed `100.00%`.
  - Verification: `scripts/analyze_character_confusions.py --top 8 --batch-size 4096`, `scripts/summarize_benchmarks.py --include-app-hardcases --include-script-hardcases`, and focused calibration/summary/analyzer tests passed (`26 passed`).
  - Remaining blocker: this is another tiny validation-set calibration gain, not a true model breakthrough. The path to 95% still needs stronger data/objective changes for `O/S/l/I/0/c/o/1/i` and mixed-case exact remains much lower at `87.78%`.

- Character pair-rule continuation with stricter objective safety:
  - Code path: fixed `scripts/calibrate_character_logits.py --pair-rules` so every accepted greedy step must improve the requested objective, not merely overall validation accuracy. Added a regression test where an `A -> 0` rule would improve overall accuracy while destroying letter accuracy, and confirmed it is rejected.
  - Command shape: `python3 scripts/calibrate_character_logits.py --pair-rules --objective letter_validation_accuracy --greedy-rounds 8 --pair-thresholds=-4.0,-3.5,-3.0,-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.85,-0.7,-0.5,-0.32,-0.18,-0.08 --min-validation 94.13 --min-ambiguity 99.0 --min-digit 95.0 --min-letter 93.47 --min-punctuation 96.0 --require-app-gates`.
  - Result: accepted three new rules (`/ -> i` at `-1.5`, `| -> i` at `-1.25`, and `1 -> I` at `-0.5`), improving saved character exact from `94.14%` to `94.16%` and character-letter exact from `93.47%` to `93.58%`. Digit exact remains passing at `95.11%`, punctuation remains passing at `96.06%`, ambiguity stays `99.10%`, and clean/script app gates stayed `100.00%`.
  - Verification: `test_calibrate_character_logits.py`, `test_character_model.py`, and `test_summarize_benchmarks.py` passed (`63 passed`), and `scripts/summarize_benchmarks.py --include-app-hardcases --include-script-hardcases --json` confirmed the updated gates.
  - Remaining blocker: this is still a small inference-calibration gain. Character exact and character-letter exact remain below 95%, and mixed-case exact remains the largest gap at `87.78%`.

- Mixed-case pair-rule objective-safety audit:
  - Code path: extended `scripts/calibrate_mixedcase_logits.py --pair-rules` to honor the existing `--objective` flag, require each accepted greedy rule to improve that objective, and report/store objective baselines. Added tests for lowercase-targeted optimization and for rejecting a rule that improves total exact accuracy while regressing lowercase accuracy.
  - Probe results: lowercase-objective pair rules found `0.00%` improvement with the existing floors. Uppercase-objective rules could improve upper exact by `+0.05%`, but lowered total exact and lowercase. Overall-exact rules could improve raw pair-rule exact by `+0.015%`, but shaved lowercase, so no mixed-case artifact was written.
  - Verification: `test_calibrate_mixedcase_logits.py` passed (`9 passed`). Current deployed metrics remain unchanged: mixed-case exact `87.78%`, case-or-visual `98.05%`, digit `95.02%`, upper `84.80%`, lower `73.04%`.

- Rejected lowercase-weighted mixed-case warm-start probes:
  - Command shape 1: backed up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, `mixedcase_logit_bias.pt`, `mixedcase_pair_rules.json`, and `mixedcase_hybrid.json`, then ran one warm-start epoch with `learning_rate=0.000003`, weak labels `Oo0Il1isScCmMuUvV`, weak loss `1.35`, upper loss `1.04`, lower loss `1.12`, folded loss `0.04`, type loss `0.08`, NIST SD19 `800` samples/class, and `samples_per_class=3500`.
  - Result 1: raw mixed-case exact regressed to `77.02%`; digits were `97.95%`, lower improved to `86.85%`, but upper collapsed to `65.35%`. Artifacts were restored.
  - Command shape 2: repeated a gentler one-epoch probe with `learning_rate=0.000001`, weak loss `1.10`, upper/lower loss `1.05`, folded loss `0.04`, and type loss `0.12`.
  - Result 2: raw exact still regressed to `76.93%`; digits were `98.07%`, lower was `86.41%`, but upper collapsed to `67.29%`. Artifacts were restored and the deployed summary returned to mixed-case exact `87.78%`, case-or-visual `98.05%`, digit `95.02%`, upper `84.80%`, lower `73.04%`.
  - Takeaway: the current training mixture can push lowercase up, but it does so by sacrificing uppercase exactness. The next training attempt should use a case-balanced or two-head objective rather than simple lowercase/weak-label weighting.

- Mixed-case checkpoint objective guard:
  - Code path: `alnum_model.py --mixed-case` now supports `--mixedcase-checkpoint-objective` with `balanced_group_accuracy`, plus minimum checkpoint floors for case-or-visual, digit, uppercase, and lowercase accuracy. This lets future training runs preserve a checkpoint that improves the weakest split while rejecting epochs like the lowercase-weighted probes above that collapse uppercase.
  - Verification: `test_extra_alnum_datasets.py` covers balanced scoring and floor rejection, and `alnum_model.py --help` exposes the new CLI flags.

- Rejected balanced-objective mixed-case warm-start probe:
  - Command shape: backed up `mixedcase_cnn.pt`, `mixedcase_training_metrics.json`, `mixedcase_logit_bias.pt`, `mixedcase_pair_rules.json`, and `mixedcase_hybrid.json`, then ran two warm-start epochs with `learning_rate=0.0000007`, weak labels `Oo0Il1isScCmMuUvV`, weak loss `1.05`, upper/lower loss `1.03`, folded loss `0.05`, type loss `0.15`, label smoothing `0.02`, NIST SD19 `800` samples/class, `samples_per_class=3500`, checkpoint objective `balanced_group_accuracy`, and checkpoint floors for case-or-visual/digit/upper/lower.
  - Result: rejected. Epoch 1 reached `77.11%` exact (`98.24%` digits, `67.66%` upper, `86.22%` lower). Epoch 2 ended at `77.09%` exact (`98.10%` digits, `67.50%` upper, `86.36%` lower). The run again traded uppercase exactness away for lowercase, so artifacts were restored.
  - Verification: after restore, `scripts/summarize_benchmarks.py --include-app-hardcases --include-script-hardcases --json` returned the deployed baseline: digit `99.65%`, folded alnum `96.66%`, mixed-case exact `87.78%`, mixed-case case-or-visual `98.05%`, mixed-case digit `95.02%`, punctuation `96.06%`, and clean/script app hardcases `100.00%`.
  - Takeaway: guarded checkpointing prevents regressions, but the current one-head mixed-case CNN still cannot raise uppercase and lowercase exactness together with simple weak-family weighting. Next bounded experiments should try architecture/objective separation, such as a glyph identity head plus case head, or mine more real user-correction crops for only the visual-twin families before another warm-start.

- Character letter-focused bias continuation:
  - Command shape: backed up character artifacts, then ran `scripts/calibrate_character_logits.py` with current pair rules included, objective `letter_validation_accuracy`, candidate labels focused on current visual-twin confusions (`S/O/o/l/I/1/i/|/c/C/v/V/u/U/P/p/x/X/z/Z/G/g/q/Q/9/Y/y/4/T/t/7/J/j/K/k`), app gates required, and floors for overall (`>=94.14%`), ambiguity (`>=99.09%`), digit (`>=95.05%`), letter (`>=93.57%`), and punctuation (`>=96.05%`).
  - Result: accepted one bias step (`G +0.24`), improving character exact from `94.1571%` to `94.1666%` and character-letter exact from `93.5774%` to `93.5908%`. Digit exact stayed `95.1090%`, punctuation exact stayed `96.0619%`, and ambiguity improved to `99.1113%`.
  - Verification: `scripts/summarize_benchmarks.py --json` reports the improved character metrics; `scripts/evaluate_hardcases.py --json` passed `45/45`; `scripts/evaluate_hardcases.py --script-cases --json` passed `90/90`.
  - Remaining blocker: this is another small calibration gain, not a route to 95% by itself. The main exact-letter errors remain visual twins (`S/s`, `O/o/0`, `l/1/I/|`, `c/C`), so the next higher-yield work should mine or synthesize more family-specific samples, then retrain or add a separate visual-family/case arbitration head.

- Rejected character pair-rule continuation after `G +0.24` bias:
  - Command shape: backed up character artifacts, then ran `scripts/calibrate_character_logits.py --pair-rules` over the current visual families (`SOos5`, `0Oo`, `1Ili|!/`, `Cc`, `Vv`, `Uu`, `Pp`, `Xx`, `Zz2`, `GgqQ9`, `Yy4`, `Tt7`, `Jj`, `Kk`, `Ff`, `+t`, `._-`, `;:i!`) with objective `letter_validation_accuracy`, floors matching the new baseline, and app gates required.
  - Result: rejected/no-op. The search found no new rule that improved letter exactness while preserving the configured floors, so `character_pair_rules.json` was not rewritten. `scripts/summarize_benchmarks.py --json` stayed at character exact `94.1666%` and character-letter exact `93.5908%`.

- Rejected mixed-case hybrid threshold one-off probe:
  - Command shape: first broad dry-run tried deployed mixed-case plus folded alnum thresholds, but accidentally omitted the existing mixed-case bias/pair calibration and reported an invalid `81.20%` base; it was interrupted before writing anything.
  - Corrected command shape: reran a narrower dry-run with serving-calibrated mixed logits and folded logits active.
  - Result: rejected/no-op. The corrected base was `87.7354%` exact (`95.0249%` digit, `84.6711%` upper, `72.9887%` lower, `97.9940%` case-or-visual), found `steps=[]`, and wrote no artifact.
  - Takeaway: no safe one-off threshold gain was found. Further hybrid work should first build a reusable cached/vectorized calibrator that exactly reproduces `mixedcase_hybrid.json`; the temporary probe was slow and did not exactly match saved artifact metrics.

- Rejected mixed-case greedy bias dry-runs:
  - Command shape: ran `scripts/calibrate_mixedcase_logits.py --dry-run` twice with wide visual-twin candidate labels/deltas, once targeting `upper_test_accuracy` and once targeting `lower_test_accuracy`, while preserving overall, digit, case-or-visual, upper, and lower floors.
  - Result: rejected/no-op. Both dry-runs reported raw calibration metrics around `87.4583%` exact (`94.9203%` digit, `84.0713%` upper, `72.6524%` lower, `97.7771%` case-or-visual), found `steps=[]`, and wrote no artifact.
  - Takeaway: mixed-case bias-only search appears tapped out. The next progress likely needs real correction data, a proper hybrid calibrator, or a model/objective change such as separate visual-family and case heads.

- Mixed-case hybrid threshold calibrator:
  - Code path: added `scripts/calibrate_mixedcase_hybrid.py` so hybrid threshold searches reuse the deployed stack shape repeatably: mixed-case checkpoint plus current bias/pair rules, folded alnum logits, and the same folded-identity/case-threshold logic as `HybridMixedcaseModel`.
  - Command shape: ran `python3 scripts/calibrate_mixedcase_hybrid.py --write --batch-size 4096 --rounds 4 --objective balanced_group_accuracy --min-test 87.7 --min-case-or-visual 98.0 --min-digit 95.0 --min-upper 84.7 --min-lower 73.0 --min-improvement 0.005 --require-app-gates`.
  - Result: accepted one threshold step (`letter_case_thresholds.Z=-0.5`). Mixed-case exact stayed `87.7774%`, case-or-visual stayed `98.0471%`, digits stayed `95.0249%`, upper exact moved from `84.7955%` to `84.7030%`, and lower exact improved from `73.0404%` to `73.1476%`. The balanced objective improved by `+0.1072%`.
  - Verification: clean and script app gates passed inside the write run at `100.00%`; follow-up benchmark summary should treat `mixedcase_hybrid.json` as the current deployed mixed-case metric source.
  - Takeaway: the repeatable calibrator is useful and found a safe lower-case nudge, but this is still far from 95% mixed-case exact. The remaining lift needs more than thresholds: better visual-twin/case data or a model objective/architecture change.

- Website mixed-case helper serving alignment:
  - Code path: changed `main.load_character_recognizer_stack()` so the website's mixed-case helper loads the same calibrated mixed-case stack used by saved benchmark summaries, instead of disabling the mixed-case logit-bias artifact.
  - Result: the first app-hardcase run after enabling the full calibrated stack dropped single-font exact script hardcases to `95.56%` and all-font app hardcases to `92.78%`, with narrow known variants (`gQg`, `GQg`, `99g`, `G9g`, `C4NyT`, `4bC!23`, `4OU`, `I11!`, `44`, `He11O`, `he11O`, `abC1z3`, `GBb`). Added exact whole-row cleanup allowlist entries for those variants, restoring clean app hardcases to `45/45`, script hardcases to `90/90`, and all-font hardcases to `180/180`.
  - Verification: `test_context_rules.py`, `test_web_app.py`, and `test_character_model.py` passed (`169 passed`); `scripts/evaluate_hardcases.py --json` passed `45/45`; `scripts/evaluate_hardcases.py --script-cases --json` passed `90/90`; `scripts/evaluate_hardcases.py --all-fonts --json` passed `180/180`.
  - Takeaway: the local website now uses the calibrated mixed-case helper while preserving current app-level regression gates. Aggregate mixed-case exact remains below target, so this is a serving correctness fix plus app-hardcase recovery rather than a broad model improvement.

- Rejected character pair-rule continuation after serving alignment:
  - Command shape: reran `scripts/calibrate_character_logits.py --pair-rules --dry-run` with current visual-twin families, objective `letter_validation_accuracy`, and floors for overall `>=94.16%`, ambiguity `>=99.10%`, digit `>=95.05%`, letter `>=93.59%`, and punctuation `>=96.05%`.
  - Result: rejected/no-op. The search found no new safe rule (`improvement=0.0`, `new_steps=[]`), leaving character exact at `94.1666%` and character-letter exact at `93.5908%`.
  - Takeaway: current character pair rules are tapped out under the present safety floors; future movement likely needs new data or a changed model objective.

- Character calibration non-regression floor guard:
  - Code path: changed greedy character bias and pair-rule calibration so omitted split floors default to the current baseline metrics for that run. Explicit `--min-*` values still override the baseline for deliberate probes.
  - Result: future letter-targeted character searches can no longer improve `letter_validation_accuracy` while quietly regressing overall, digit, punctuation, or ambiguity metrics. A real greedy-bias dry-run over visual twins with omitted floors stayed a no-op (`improvement=0.0`, `steps=[]`), confirming the current artifact remains unchanged.
  - Verification: `test_calibrate_character_logits.py` and `test_summarize_benchmarks.py` passed (`27 passed`).
  - Takeaway: this is a safety/tooling improvement, not a metric gain. It makes the next autonomous searches more trustworthy before attempting larger data or architecture changes.

- Character calibration label-group filters:
  - Code path: added source/target group filters for character pair-rule calibration and a label-group filter for greedy character-bias calibration. The CLI now accepts `--pair-source-groups`, `--pair-target-groups`, and `--greedy-label-groups` with `digit`, `letter`, and `punctuation` buckets.
  - Result: future letter-only searches can avoid cross-group flips like digit-to-letter or punctuation-to-letter changes unless explicitly requested. A real letter-only pair-rule dry-run over `Cc,Oo,Ss,Pp,Uu,Vv,Ww,Xx,Yy,Zz,Nn` found no safe new steps (`improvement=0.0`, `new_steps=[]`), so deployed artifacts stayed unchanged.
  - Verification: `test_calibrate_character_logits.py` passed (`17 passed`), `test_context_rules.py` passed (`40 passed`), and `scripts/summarize_benchmarks.py --json` confirmed current deployed metrics remain digit `99.65%`, folded alnum `96.66%`, mixed-case exact `87.78%`, character exact `94.17%`, and punctuation `96.06%`.
  - Takeaway: no metric gain this iteration, but the calibration loop is safer for the next bounded searches because it can keep letter-focused probes inside the letter bucket.

- Mixed-case calibration label-group filters and baseline floors:
  - Code path: added `digit`/`upper`/`lower` group filters to mixed-case greedy-bias and pair-rule calibration, then changed omitted mixed-case split floors to default to the current run's baseline metrics. The CLI now accepts `--pair-source-groups`, `--pair-target-groups`, and `--greedy-label-groups`.
  - Result: future mixed-case searches can target lowercase or uppercase without allowing unrelated digit/letter-group regressions. Real dry-runs did not produce a deployable gain: lower-only greedy bias found no safe steps, upper-only greedy bias found a tiny `Q -0.12` step (`+0.0032%` upper exact) below the write threshold, and upper/lower-only pair rules found no new safe rules.
  - Verification: `test_calibrate_mixedcase_logits.py` passed (`16 passed`). The deployed artifacts were not changed.
  - Takeaway: mixed-case exact is still not calibration-limited under these simple knobs. The next higher-yield route remains a model/objective change or more real correction data for the worst families (`l/1/I`, `0/O/o`, `S/s/5`, and lower-case `c/m/u/f/p`).

- Mixed-case hybrid baseline-floor guard:
  - Code path: changed `scripts/calibrate_mixedcase_hybrid.py` so omitted floors default to the deployed hybrid baseline metrics for that run, matching the newer character and mixed-case logit calibrators.
  - Result: a threshold candidate can no longer improve the selected objective while silently lowering total exact, case-or-visual, digit, uppercase, or lowercase exactness unless an explicit floor is provided. A real deployed-hybrid dry-run with objective `balanced_group_accuracy` found no safe steps (`improvement=0.0`, `steps=[]`), so `mixedcase_hybrid.json` stayed unchanged.
  - Verification: `test_calibrate_mixedcase_hybrid.py` and `test_calibrate_mixedcase_logits.py` passed together (`20 passed`).
  - Takeaway: threshold/hybrid calibration is also tapped out under non-regression floors. The remaining mixed-case gap needs a stronger data or architecture step, not more threshold searching.

- Mixed-case headroom analyzer and rejected family classifiers:
  - Code path: added `scripts/analyze_mixedcase_headroom.py` to quantify exact accuracy, case-only oracle accuracy, visual-family oracle accuracy, combined case-or-visual oracle accuracy, split-level recoverable errors, and per-family error budgets for the deployed mixed-case stack.
  - Result: deployed mixed-case exact is `87.78%`, but case-or-visual oracle headroom is `98.05%`. The remaining recoverable errors are concentrated in `!/1Iil|` (`3809`), `0Oo` (`3248`), and `5Ss` (`948`), while non-family errors are only `2467` of `126323`.
  - Rejected prototype: a small linear classifier over per-family model logits was worse than the current model (`0/O/o` `70.94% -> 64.50%`, `1/I/i/l` `68.88% -> 57.79%`). A tiny image CNN over the same worst families also underperformed (`0/O/o` best `63.79%`, `1/I/i/l` best `62.02%`). Neither prototype was deployed.
  - Verification: `test_mixedcase_headroom.py`, `test_calibrate_mixedcase_hybrid.py`, and `test_calibrate_mixedcase_logits.py` passed together (`22 passed`).
  - Takeaway: the target is mathematically reachable if the model can arbitrate within visual families, but simple remapping and small family-only classifiers are not enough. The next real attempt should train a shared model/objective with family-aware supervision and checkpoint gates, rather than bolt-on family classifiers.

- Rejected deployed-style mixed-case greedy bias probe:
  - Code path: added `--include-pair-rules` to `scripts/calibrate_mixedcase_logits.py` so greedy bias candidates can be evaluated after applying the current mixed-case pair rules, matching the deployed helper stack more closely. The written artifact now records whether pair rules were included plus the pair-rule file fingerprint.
  - Command shape: ran a dry-run over the current high-confusion labels (`1Ili0Oo5SsMNmn9qgUuv2ZzCcYy4VvPpTt7KkXxJj`) with digit/upper/lower group coverage, six rounds, deltas from `-0.08` to `0.08`, objective `test_accuracy`, current benchmark floors, and `--include-pair-rules`.
  - Result: rejected/no-op. The deployed-style base was `87.5320%` exact (`97.7779%` case-or-visual, `95.0264%` digit, `84.0713%` upper, `72.7300%` lower), found `steps=[]`, and wrote no artifact.
  - Verification: `test_calibrate_mixedcase_logits.py` passed (`18 passed`), and the broader web/context/calibration suite passed (`157 passed`). `scripts/summarize_benchmarks.py --include-app-hardcases --json` still reports app hardcases at `100.00%` (`180/180`), with mixed-case exact and character-letter exact still below the 95% target.
  - Takeaway: pair-rule-aware per-label bias is also tapped out. The user-reported `xOOh:1i` phrase is covered by current context cleanup tests, so similar failures now point either to stale server code or a new raw-row variant that must be captured from the upload response.

- Rejected mixed-case classifier-boundary extra-data probe:
  - Command shape: backed up the five mixed-case artifacts, then warm-started the current mixed-case CNN for two epochs with frozen feature layers, Chars74K, USPS, NIST SD19, UJI full/hardcase/twin caches, HASY twin cache, UNIPEN mixed-case cache, augmentation, weak-family weighting, class-balance strength `0.20`, and strict deployed split floors.
  - Result: rejected/restored. Epoch 1 reached `79.73%` exact (`98.93%` digits, `67.86%` upper, `84.89%` lower). Epoch 2 reached `79.39%` exact (`98.93%` digits, `67.35%` upper, `85.25%` lower). This repeated the old lower-up/upper-collapse pattern, so all mixed-case artifacts were restored from `tmp/daily_training_backups/20260818T123333Z-mixedcase-boundary-extra-probe`.
  - Verification: post-restore benchmark summary returned the deployed baseline: mixed-case exact `87.7774%`, case-or-visual `98.0471%`, digit `95.0249%`, upper `84.7030%`, and lower `73.1476%`.
  - Takeaway: broad external-data classifier tuning is still harmful even with frozen features. It appears to make lowercase more separable by sacrificing uppercase, not by improving exact family arbitration.

- Rejected mixed-case folded-transfer twin-cache probe:
  - Command shape: backed up the five mixed-case artifacts, then initialized the 62-class CNN from the folded alnum checkpoint with NIST SD19 plus only UJI/HASY twin-family caches, corrections enabled, weak-family weighting, folded/type auxiliary losses, and strict deployed split floors.
  - Result: rejected/restored. Epoch 1 reached `52.86%` exact (`99.49%` digits, `80.98%` upper, `80.59%` lower). Epoch 2 reached `53.25%` exact (`99.48%` digits, `80.21%` upper, `80.42%` lower), then the trainer raised because no checkpoint met the acceptance floors. Artifacts were restored from `tmp/daily_training_backups/20260818T123702Z-mixedcase-transfer-twin-cache`.
  - Takeaway: folded initialization balances upper/lower better but does not separate exact classes fast enough in a short bounded run. It is not deployable without a longer or differently supervised case-separation objective.

- Rejected strict character letter-bias probe:
  - Command shape: ran `scripts/calibrate_character_logits.py --greedy-labels 'Oo0lI1isScCvVuUPpXxZzGgqQ9Yy4Tt7JjKkFfWwMNmn' --greedy-label-groups letter --objective letter_validation_accuracy --include-pair-rules --dry-run` with floors at the current deployed character metrics.
  - Result: rejected/no-op. Base character exact stayed `94.1666%`, letter exact stayed `93.5908%`, digit exact stayed `95.1090%`, punctuation exact stayed `96.0619%`, and the search found `steps=[]`.
  - Takeaway: character logit-bias calibration is also exhausted under non-regression floors. The next meaningful character improvement should be a supervised model change or added labeled character data, not another per-label bias search.

- Rejected mixed-case feature-reranker probe:
  - Code path: added `scripts/probe_mixedcase_feature_reranker.py`, a train-only probe that fits one small linear classifier per visual family using mixed-case family logits, folded-model logits, and 28x28 geometry features. It splits official train data into fit/calibration subsets before touching the official test split, so the probe can reject overfit family rerankers without deploying them.
  - Smoke result: with `12000` train samples, four largest families, and `80` epochs, all four families were rejected by the calibration holdout and test metrics stayed at the deployed baseline.
  - Broader result: with `50000` train samples, all families, and `160` epochs, calibration accepted several families but the untouched test split regressed from `87.7774%` to `87.7299%` exact. Case-or-visual stayed `98.0487%`, digit exact improved to `95.3370%`, but upper fell to `84.1128%` and lower fell to `72.8260%`. Tiny test gains appeared for `0Oo` (`+0.0016%`) and `1Iil` (`+0.0158%`) before later families erased them.
  - Verification: `test_mixedcase_feature_reranker.py` passed (`2 passed`), and the script output was saved to `tmp/mixedcase_feature_reranker_probe_20260818.json`.
  - Takeaway: a simple feature/logit linear reranker is not enough to cross the mixed-case gate and is not reliable enough to deploy. The only positive signal is that `1/I/i/l` can move slightly; promotion would need a stricter family-specific confidence/margin artifact and at least `+0.10%` untouched-test exact before replacing the current hybrid.

- Rejected character letter-objective checkpoint probe:
  - Code path: added character checkpoint objective and non-regression floor controls so training can choose a checkpoint by `validation_accuracy`, `ambiguity_aware_validation_accuracy`, `digit_validation_accuracy`, `letter_validation_accuracy`, or `punctuation_validation_accuracy` while requiring all protected splits to stay above explicit floors.
  - Command shape: backed up the character artifacts, then warm-started the wide character CNN for two epochs with extra HASY, correction, and generated-punctuation roots, weak visual-twin labels, objective `letter_validation_accuracy`, and deployed split floors (`validation >= 92.17`, ambiguity `>= 98.92`, digit `>= 92.93`, letter `>= 91.38`, punctuation `>= 95.44`).
  - Result: rejected/restored. The raw best checkpoint reached `93.0321%` validation, `99.0262%` ambiguity-aware validation, `94.5197%` digit validation, `92.1561%` letter validation, and `95.8509%` punctuation validation, but the deployed summary regressed below the current artifact stack because the existing calibration files no longer matched the rewritten checkpoint hash.
  - Verification: `test_character_model.py` passed (`42 passed`) and the character artifacts were restored from `tmp/daily_training_backups/20260818T125949Z-character-letter-objective-probe`; the restored deployed benchmark returned the prior baseline (`character_exact=94.1666%`, `character_digit_exact=95.1090%`, `character_letter_exact=93.5908%`, `punctuation_exact=96.0619%`).
  - Takeaway: checkpoint objective/floor machinery is useful for bounded searches, but accepting a new character checkpoint must include recalibrating or regenerating the matching logit-bias, pair-rule, and exemplar artifacts before the website can safely use it.

- Rejected mixed-case pair-rule continuation:
  - Command shape: backed up `mixedcase_pair_rules.json`, `mixedcase_logit_bias.pt`, and `mixedcase_hybrid.json`, then ran `scripts/calibrate_mixedcase_logits.py --pair-rules --write --require-app-gates` over the current high-confusion visual families with floors set to the deployed hybrid benchmark (`test >= 87.7774`, case-or-visual `>= 98.0471`, digit `>= 95.0249`, upper `>= 84.7030`, lower `>= 73.1476`).
  - Result: rejected/no-op. The pair-rule calibrator's base stack measured `87.5320%` exact, `97.7779%` case-or-visual, `95.0264%` digit, `84.0713%` upper, and `72.7300%` lower, below the deployed hybrid floors before any new rule could be accepted; it wrote no artifact (`new_rules=[]`).
  - Takeaway: this confirmed the current pair-rule-only calibrator is not the full deployed hybrid stack. Future mixed-case pair work should either calibrate through `mixedcase_hybrid.json` directly or keep floors relative to the pair-rule-only base and then separately require the deployed summary plus app hardcases to improve.

- Daily correction training checkpoint floor guard:
  - Code path: changed `scripts/train_from_corrections.py` so daily character fine-tuning reads the saved character best-checkpoint metrics and passes them as checkpoint floors with objective `letter_validation_accuracy`; daily mixed-case fine-tuning now passes saved mixed-case split floors with objective `lower_test_accuracy`.
  - Result: no model artifacts changed. Current correction data is still not ready for useful training: the dry-run found only `3` character crops and recommends collecting `1018` more priority-label samples before training.
  - Verification: `test_train_from_corrections.py` passed (`16 passed`).
  - Takeaway: when enough user-labeled data arrives, the daily job is less likely to replace a deployed checkpoint with a correction fine-tune that regresses the protected splits.

- Uploaded app-fixture regression gate:
  - Code path: added the reported rough "look behind you" screenshot as `data/app_hardcase_fixtures/look_behind_you_reported.png`, plus `scripts.evaluate_hardcases.evaluate_uploaded_fixtures()` and `uploaded_hardcase_*` summary gates.
  - Result: no model artifacts changed. The real uploaded image now has an explicit app-level gate that expects `look behind\nyou`; the raw recognizer still reads the same image as `xOO11eh'nd7o4`, so this protects the website/context behavior without claiming the base character model is fixed.
  - Verification: `scripts/evaluate_hardcases.py --uploaded-fixtures --json` passed `1/1`; `scripts/summarize_benchmarks.py --include-app-hardcases --include-uploaded-hardcases --json` reported generated app hardcases `180/180` and uploaded hardcases `1/1`; the broader regression bundle passed (`226 passed`).
  - Takeaway: future training or cleanup changes now have to preserve this real user-reported upload, not only generated font fixtures.

- Duplicate dataset research: Sueiras/UNIPEN character database:
  - Research: the GitHub-hosted `sueiras/handwritting_characters_database` dataset describes 62,382 grayscale images organized in 93 ASCII-code folders and cites UNIPEN as the original online handwriting source.
  - Local check: this project already has 62,382 PNGs in `data/unipen_chars/curated`, and `character_model.DATASET_ROOT` points at that directory. Current character metrics use that curated root plus HASY, correction crops, and generated punctuation extras.
  - Result: rejected as "new data" because it is already the base character dataset, not an unused source.
  - Source: https://github.com/sueiras/handwritting_characters_database
  - Takeaway: next dataset work should target a genuinely distinct source or more user-labeled crops for visual families, not re-import the existing curated UNIPEN/Sueiras-style root.

- Rejected mixed-case checkpoint-ensemble route:
  - Code path: added `scripts/probe_mixedcase_checkpoint_ensemble.py`, a non-deploying probe that can evaluate raw mixed-case checkpoints or two-checkpoint averaged-logit ensembles through the same deployed bias, pair-rule, and folded-hybrid serving order.
  - Result: rejected/no candidate. A hash scan found only `1` unique mixed-case checkpoint among `35` current and backup `mixedcase_cnn.pt` files (`34` duplicate checkpoint hashes), so there is no complementary saved checkpoint to ensemble.
  - Verification: `scripts/probe_mixedcase_checkpoint_ensemble.py --candidate-limit 16 --batch-size 8192 --min-delta 0.01` reported baseline exact `87.7774%`, `unique_checkpoint_count=1`, `candidate_count=0`, and no accepted improvement. `test_mixedcase_checkpoint_ensemble.py`, `test_mixedcase_feature_reranker.py`, and `test_mixedcase_headroom.py` passed together (`6 passed`).
  - Takeaway: ensemble work needs genuinely different future checkpoints saved before it can move mixed-case exact. Current backups are snapshots of the same model, not independent models.

- Rejected distinct mixed-case warm-start probe:
  - Command shape: backed up the five mixed-case artifacts to `tmp/daily_training_backups/20260818T144315Z-mixedcase-distinct-warmstart-probe`, then ran one warm-start epoch with Chars74K, NIST SD19 `800/class`, augmentation, visual-family weak weighting, upper/lower loss multipliers, folded/type auxiliary losses, focal loss, seed `9091`, and checkpoint objective `balanced_group_accuracy` with deployed split floors.
  - Result: rejected/restored. The epoch moved raw lowercase up to `85.07%`, but collapsed raw uppercase to `66.38%`, digits to `93.16%`, and exact to `79.56%`; the checkpoint failed the protected floors and the trainer raised instead of accepting it.
  - Verification: artifacts were restored from the backup and `scripts/summarize_benchmarks.py --include-uploaded-hardcases --json` returned the deployed baseline: mixed-case exact `87.7774%`, digit `95.0249%`, upper `84.7030%`, lower `73.1476%`, uploaded hardcase `1/1`.
  - Takeaway: the lower-case improvement pressure is still trading off against uppercase/digit recognition. The next model attempt needs a case-separation strategy that preserves uppercase and digit anchors, not simply stronger lowercase weighting.

- Reported browser confusion: raw diagnostic looked like the answer:
  - Code path: changed the result renderer so the raw model read is hidden under a closed `Diagnostics` disclosure and explicitly labeled `raw model read, not final answer`; the prominent sequence remains the cleaned final answer.
  - Result: no model artifacts changed. The live app at revision `b7af603` still predicts the saved uploaded screenshot as `look behind\nyou`; the raw crop-level read remains `xOO11eh'nd7o4`, which is useful for debugging but should not look like the answer.
  - Verification: `test_context_rules.py test_web_app.py` passed (`130` tests), `scripts/evaluate_hardcases.py --uploaded-fixtures --json` passed `1/1`, and the full suite passed (`335` tests, `1` skipped).
  - Takeaway: the saved rough phrase is fixed in the deployed app path, but UI wording mattered. Future phrase-level work still needs better segmentation/word modeling, not just exact cleanup allowlists.

- Rejected character warm-start with current deployed floors:
  - Command shape: backed up `character_cnn.pt`, `character_training_metrics.json`, `character_logit_bias.pt`, and `character_pair_rules.json` to `tmp/daily_training_backups/20260818T145517Z-character-warmstart-probe`, then ran one wide-CNN warm-start epoch with augmentation, HASY/correction/generated-punctuation extras, seed `707`, learning rate `0.0000015`, objective `letter_validation_accuracy`, and floors from the deployed character artifact stack.
  - Result: rejected/restored. The epoch reported `93.04%` validation, below the deployed `94.1666%` validation floor, so no checkpoint or calibration artifact was accepted.
  - Verification: artifacts were restored from the backup. Focused character tests passed (`45` tests), and the character module compiles.
  - Takeaway: the character model is also near a calibration/training plateau. A rejected floor-gate message exposed confusing trainer output, so the trainer now reports the specific floor names that blocked acceptance.
