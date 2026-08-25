# MAGI

This repository contains the anonymous implementation of **Memory-Augmented Generative Inference for Source Localization under Dynamic Forwarding Behavior**.

The artifact contains only the complete MAGI method. It does not include alternative experimental entry points, tuning scripts, exploratory experiments, or baseline implementations.

## Method

MAGI combines three components:

1. structural retrieval memory with conditional VAE augmentation;
2. observation-conditioned Source-VAE posterior sampling;
3. stochastic residual forward consistency with posterior reweighting.

The model receives graph topology and one final diffusion snapshot. Intermediate diffusion states, exposure counts, simulator parameters, and the true number of sources are not used at inference.

## Repository Layout

```text
MAGI/
├── main.py
├── pyproject.toml
├── requirements.txt
├── config/
│   └── locked_seed127.json
├── data/
│   ├── manifest.json
│   └── <dataset>_<mechanism>.npz
├── checkpoints/
│   ├── proposal/
│   ├── memory/
│   ├── forward/
│   └── final/
├── results/
│   └── expected_seed127.csv
├── scripts/
│   ├── compare_results.py
│   └── verify_assets.py
└── src/magi/
```

## Environment

The reported experiments used:

- Python 3.10.20;
- PyTorch 2.5.1 with CUDA 12.1;
- FAISS CPU 1.8.0;
- NumPy 1.26.4;
- SciPy 1.14.1;
- scikit-learn 1.7.2;
- NetworkX 3.4.2.

Create the environment with:

```bash
conda create -n magi python=3.10.20 -y
conda activate magi
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

CPU execution is supported for small graphs. CUDA is recommended for the complete experiment matrix.

## Data

The repository contains 100 cascades for each combination of five graphs and three diffusion mechanisms:

- Dolphins;
- Network Science;
- Cora-ML;
- Power Grid;
- Facebook Company;
- Independent Cascade, Linear Threshold, and exposure-dependent USE.

Each NPZ file stores:

- `node_count`: number of graph nodes;
- `edge_index`: one copy of the undirected edge set;
- `sources`: source indicators for all 100 cascades;
- `observations`: corresponding final-snapshot indicators;
- `source_seeds` and `diffusion_seeds`: generation seeds.

The original experiment files stored the same graph repeatedly inside every cascade object. The compact NPZ representation removes that duplication. It preserves every input consumed by MAGI. Intermediate frames are omitted because the method never reads them. `data/manifest.json` records graph statistics, compact-file checksums, and checksums of the original canonical PKL files.

Verify all released data and checkpoints before running:

```bash
python scripts/verify_assets.py
```

## Checkpoints

The released checkpoints correspond to model seed 127, which is the seed reported in the main comparison table. For each scenario, the artifact contains:

- the multi-scale proposal checkpoint;
- the structural-memory encoder checkpoint;
- the stochastic residual forward-consistency checkpoint;
- the final Source-VAE checkpoint;
- the final memory-conditioned proposal cache.

Only checkpoints for the complete method are included.

## Reproduce One Scenario

```bash
PYTHONPATH=src python main.py \
  --mode evaluate \
  --datasets dolphins \
  --mechanisms IC \
  --seeds 127 \
  --device cuda \
  --output-root outputs/dolphins_ic
```

The expected result is approximately:

```text
F1  = 0.7701143791
AUC = 0.9732207792
```

Compare generated metrics with the released values:

```bash
python scripts/compare_results.py outputs/dolphins_ic/metrics.csv
```

## Reproduce the 12 Main Scenarios

The default dataset list contains Dolphins, Network Science, Cora-ML, and Power Grid:

```bash
PYTHONPATH=src python main.py \
  --mode evaluate \
  --seeds 127 \
  --device cuda \
  --output-root outputs/main_12_scenarios

python scripts/compare_results.py outputs/main_12_scenarios/metrics.csv
```

## Reproduce All 15 Scenarios

```bash
PYTHONPATH=src python main.py \
  --mode evaluate \
  --datasets dolphins netscience cora_ml power_grid facebook_company \
  --mechanisms IC LT USE \
  --seeds 127 \
  --device cuda \
  --output-root outputs/all_15_scenarios

python scripts/compare_results.py outputs/all_15_scenarios/metrics.csv
```

Sparse operations can produce AUC differences below `2e-7` across compatible CUDA and SciPy builds. The comparison script uses that tolerance by default.

## Refit Posterior Stages

MAGI is optimized in stages. The following command rebuilds the generative structural-memory augmentation, retrains Source-VAE, and reruns stochastic forward reweighting. It uses the released proposal, structural encoder, and forward-consistency stage checkpoints.

```bash
PYTHONPATH=src python main.py \
  --mode train \
  --datasets dolphins \
  --mechanisms IC \
  --seeds 127 \
  --device cuda \
  --output-root outputs/refit_dolphins_ic
```

All locked values are available in `config/locked_seed127.json` and are also the command-line defaults.

## Outputs

Each run writes:

- `metrics.csv`: validation and test F1/AUC;
- `run_config.json`: resolved run configuration;
- `<scenario>/seed127/test_predictions.npz`: test labels and source scores;
- training histories and regenerated checkpoints when `--mode train` is used.

Existing outputs are resumed by scenario and seed. Use a new output directory for a clean rerun.
