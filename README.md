# Handwriting Recognizer

A small dependency-light website for classifying uploaded handwriting images. It supports the original MNIST digit CNN and an expanded character recognizer trained on a UNIPEN-derived dataset with digits, English letters, and common punctuation.

## Train

```bash
python3 mnist_model.py --epochs 6 --min-accuracy 95
```

The trainer uses the MNIST files already in `~/Downloads/data` when present, saves weights to `mnist_cnn.pt`, and writes epoch metrics to `training_metrics.json`.

For the expanded character recognizer, download/extract the UNIPEN-derived character database into `data/unipen_chars/curated`, then run:

```bash
python3 character_model.py --epochs 20 --batch-size 2048 --min-accuracy 50
```

The expanded model saves `character_cnn.pt`, `character_exemplars.pt`, `character_labels.json`, and `character_training_metrics.json`. The app automatically uses it when present. The current 93-class character model reaches about 68% validation accuracy across digits, uppercase/lowercase letters, and punctuation, then uses a compact nearest-exemplar assist for low-confidence web crops.

Dataset note: EMNIST is a strong official baseline for handwritten letters and digits, but it does not include punctuation. The expanded recognizer uses the UNIPEN-derived 93-class character dataset from `sueiras/handwritting_characters_database`, which includes digits, uppercase/lowercase English letters, and common punctuation.

For a stronger alphabet benchmark, the EMNIST experiment runner trains an alphabet-only CNN:

```bash
python3 emnist_experiment.py --split letters --model cnn --epochs 30 --batch-size 2048 --device mps
```

The checked-in `emnist_experiment.pt` checkpoint reached **95.125% test accuracy** on the EMNIST letters test split.

If torchvision cannot download USPS because of a local certificate error, fetch the archives with verified MD5 hashes first:

```bash
python3 scripts/download_usps.py --insecure-ssl
python3 alnum_model.py --epochs 5 --batch-size 2048 --learning-rate 0.00012 --warm-start --include-usps --device mps
```

## Run

```bash
python3 main.py
```

Open `http://127.0.0.1:8000`, upload one or more PNG/JPG/WEBP images, and the app returns the predicted sequence for each file. A single image may contain multiple separated handwritten characters; the app segments them in reading order and draws numbered boxes on the uploaded image so each prediction maps back to its source character.

## Test

```bash
python3 -m unittest
```
