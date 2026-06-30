# Trusted Multi-View Classification (TMC)

A PyTorch implementation of **Trusted Multi-View Classification** — an evidential
deep-learning approach that classifies samples described by several different
"views" (feature sets) while also producing a calibrated, **per-sample
uncertainty estimate**.

Each view produces *evidence* for every class, which is interpreted as the
parameters of a Dirichlet distribution. The views are then fused with
**Dempster–Shafer evidence theory**, yielding a single prediction together with
an explicit measure of how much the model trusts it.

> Reference: Han et al., *Trusted Multi-View Classification*, ICLR 2021.
> [arXiv:2102.02051](https://arxiv.org/abs/2102.02051)

---

## Why uncertainty?

Standard softmax classifiers are often over-confident, even when wrong. By
modelling each prediction as a Dirichlet distribution and combining views with
Dempster's rule, TMC can say *"I am confident"* vs. *"I am unsure"* — which is
valuable in safety-critical settings (medical, autonomous systems) and whenever
views disagree or are noisy.

---

## How it works

```
            ┌─────────────┐   evidence₁   α₁ = e₁+1 ┐
  view 1 ──▶│ Classifier₁ │──────────────▶          │
            └─────────────┘                          │
            ┌─────────────┐   evidence₂   α₂ = e₂+1 │  Dempster–Shafer
  view 2 ──▶│ Classifier₂ │──────────────▶          ├─▶  combination  ──▶ αₐ ──▶ prediction
            └─────────────┘                          │   (DS_Combin)         + uncertainty u
              ...                                     │
            ┌─────────────┐   evidenceᵥ   αᵥ = eᵥ+1 ┘
  view V ──▶│ Classifierᵥ │──────────────▶
            └─────────────┘
```

1. **Per-view evidence** — each view is passed through its own feed-forward
   `Classifier` ending in a `Softplus`, producing non-negative evidence per class.
2. **Dirichlet parameters** — `α = evidence + 1` defines a Dirichlet distribution
   over the class probabilities; the total evidence controls confidence.
3. **Fusion** — `DS_Combin` combines the per-view Dirichlet parameters pairwise
   with Dempster's rule, reducing conflict and aggregating uncertainty.
4. **Loss** — an evidential loss (expected cross-entropy under the Dirichlet plus
   a KL regulariser, annealed over training) is applied to each view *and* to the
   fused result.

---

## Project structure

| File                | Responsibility                                                            |
| ------------------- | ------------------------------------------------------------------------- |
| `main.py`           | Entry point: checks CUDA, builds the `Trainer`, runs training + test.     |
| `train.py`          | `Trainer` class — hyperparameters, data loaders, train/test loops.        |
| `model.py`          | `Classifier` (per-view net) and `TMC` (evidence + Dempster–Shafer fusion).|
| `loss_functions.py` | Evidential cross-entropy (`ce_loss`) and Dirichlet `KL` divergence.       |
| `data_loader.py`    | `MultiViewData` — loads and normalises multi-view `.mat` datasets.        |
| `test_cuda.py`      | `check_cuda` helper for GPU detection.                                    |
| `datasets/`         | Multi-view datasets in MATLAB `.mat` format.                              |

---

## Requirements

- Python 3.8+
- A CUDA-capable GPU (the current code calls `.cuda()` and requires CUDA)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The repository ships with **`handwritten_6views`** — handwritten digits
(10 classes) described by 6 complementary feature views:

| View | Description (feature family) | Dimension |
| ---- | ---------------------------- | --------- |
| 1    | Profile correlations         | 240       |
| 2    | Fourier coefficients         | 76        |
| 3    | Karhunen–Loève coefficients  | 216       |
| 4    | Morphological features       | 47        |
| 5    | Zernike moments              | 64        |
| 6    | Pixel averages               | 6         |

Each `.mat` file is expected to contain `x{i}_train` / `x{i}_test` matrices per
view and `gt_train` / `gt_test` labels. Drop additional datasets in `datasets/`
following the same naming convention.

---

## Usage

Run training and evaluation with the default configuration:

```bash
python main.py
```

Hyperparameters can be overridden via command-line flags (parsed in `train.py`):

| Flag              | Default | Description                                   |
| ----------------- | ------- | --------------------------------------------- |
| `--batch-size`    | 200     | Mini-batch size.                              |
| `--epochs`        | 500     | Number of training epochs.                    |
| `--lambda-epochs` | 50      | Annealing horizon for the KL regulariser.     |
| `--lr`            | 0.0003  | Adam learning rate.                           |

Example:

```bash
python main.py --epochs 300 --lr 1e-3 --batch-size 100
```

On completion the script prints the test accuracy and loss.

---

## License

This project reimplements the method from Han et al. (ICLR 2021) for research and
educational purposes. See the original paper for the method details.
