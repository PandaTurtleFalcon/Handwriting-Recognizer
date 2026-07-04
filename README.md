# MNIST Digit Sorter

A small dependency-light website for classifying uploaded handwritten digit images with a PyTorch MNIST CNN.

## Train

```bash
python3 mnist_model.py --epochs 6 --min-accuracy 95
```

The trainer uses the MNIST files already in `~/Downloads/data` when present, saves weights to `mnist_cnn.pt`, and writes epoch metrics to `training_metrics.json`.

## Run

```bash
python3 main.py
```

Open `http://127.0.0.1:8000`, upload one or more PNG/JPG/WEBP images, and the app returns the predicted digit sequence for each file. A single image may contain multiple separated handwritten digits; the app segments them in reading order and draws numbered boxes on the uploaded image so each prediction maps back to its source digit.

## Test

```bash
python3 -m unittest
```
