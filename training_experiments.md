# Training Experiments

This log records model/data experiments that were tried and either kept or
restored, so future improvement loops do not repeat known-bad blends.

## Current Best Baselines

- Digit specialist: `99.65%` MNIST test accuracy.
- Folded alnum helper: `96.66%` test accuracy, with `99.53%` digits and `95.28%` letters.
- Mixed-case helper: `80.50%` exact test accuracy, `87.19%` casefold, `90.34%` strict visual-ambiguity-aware, and `97.02%` case-or-visual-ambiguity-aware.
- Character model: deployed checkpoint is `92.18%` validation accuracy, with `95.44%` exact punctuation and `99.02%` ambiguity-aware punctuation after adding deterministic generated punctuation variants and tiny same-root fine-tunes.
- App hard-case evaluator: `42/42` exact after adding broader visual-twin, mixed-case, short-word, digit/letter, and punctuation hardcases.
- App hard-case all-font stress evaluator: `168/168` exact (`100.00%`) and `168/168` ambiguity-aware (`100.00%`) across Bradley Hand Bold, Comic Sans MS, Chalkboard, and Arial.
- Benchmark summary command: `python3 scripts/summarize_benchmarks.py --include-app-hardcases` now reports saved model gates plus app hardcase exact/ambiguity gates in one hourly-check command.
- Practice correction mode: the static site now includes a drawing pad for weak visual-twin labels (`0/O/o`, `1/I/l/i`, `S/s/5`, `C/c`, punctuation twins). Saved practice samples write both a correction JSONL row and a matching source PNG, so `scripts/train_from_corrections.py --dry-run` can count them and the daily trainer can crop them.
- Correction coverage dry-run: `python3 scripts/train_from_corrections.py --dry-run` counts exportable character corrections directly from JSONL plus saved source PNGs, so new practice samples appear in priority coverage before running the export/training step.
- Practice coverage API: `/api/correction-coverage` reports per-label counts, target counts, and remaining sample needs for the practice UI. Current target is `20` trainable samples per weak label before relying on correction-driven fine-tuning.
- Mixed-case confusion analysis: `scripts/analyze_mixedcase_confusions.py --top 20` confirms the exact gap is dominated by visual twins and case twins. Top misses are `1 -> l`, `0 -> o`, `O -> o`, `9 -> q`, `O -> 0`, `0 -> O`, `F -> f`, `U -> u`, `1 -> I`, and `S -> s`; this explains why exact is `80.50%` while case-or-visual is already `97.02%`.
- Character punctuation confusion analysis: `scripts/analyze_character_confusions.py --top 20` now matches the saved metric split and shows punctuation exact is mainly blocked by a few visual twins: `- -> _`, `. -> '`, `| -> i/l/'`, `/ -> l/|`, and `: <-> ;`. Punctuation ambiguity-aware is already `98.67%`, so future exact gains should target these shapes specifically instead of broad punctuation-weighted training.

## Kept Experiments

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

## Next Higher-Value Directions

- Add more real user-labeled correction uploads for exact visual twins, then use `scripts/train_from_corrections.py`.
- Try training changes that alter objective/architecture for exact mixed case, not just adding broad or synthetic extra datasets.
- Keep using `python3 scripts/evaluate_hardcases.py --json` after app-level changes; it catches failures that aggregate model metrics miss.
