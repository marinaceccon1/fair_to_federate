# Fair to Federate?
### Two-Sided Prediction of Performance and Fairness Outcomes in Healthcare Federated Learning

> Code for the paper *"Fair to Federate? Two-Sided Prediction of Performance and Fairness Outcomes in Healthcare Federated Learning"*, under review at CIKM 2026.

This repository also includes the **supplementary material** for the paper (**supplementary_material_fair_to_federate.pdf**), containing additional experimental results: predicted vs. actual plots and feature importance analyses for the BalAcc and minTPR targets, SHAP beeswarm plots for all prediction targets on both datasets (from both the candidate and existing federation members' perspectives), and R² breakdowns by federation size relative to Fitzpatrick17k.

---

## Overview

This repository implements a **two-stage meta-learning framework** for the *Federation Assessment Problem*: given only high-level metadata about a candidate institution and an existing federation, predict how federation would affect both parties' predictive performance **and** fairness — before any collaboration or data sharing takes place.

The framework is evaluated on two medical imaging domains:

- **NIH ChestX-ray14** — multi-label chest X-ray diagnosis (7 pathologies), sex as the sensitive attribute.
- **Fitzpatrick17k** — skin lesion classification (tumoral vs. non-tumoral), Fitzpatrick skin tone as the sensitive attribute.

**Key findings:**
- Performance and fairness changes can be reliably predicted from metadata alone (R² up to 0.97 on held-out configurations).
- Label bias is among the strongest predictors of fairness outcomes, often rivaling or exceeding demographic composition in importance.
- The candidate and existing federation members can have **opposed interests** (e.g. a label-biased candidate gains by joining a clean federation while harming the existing members), and the framework makes these configurations predictable before any collaboration occurs.

---

## Repository Structure

```
fair_to_federated/
├── NIH/
│   └── repo_nih/
│       ├── src/                        # Shared source code
│       │   ├── models/model.py         # DenseNet-121 model definition
│       │   ├── data/dataset.py         # Dataset class
│       │   ├── data/utils.py           # Data utilities
│       │   ├── client_nih.py           # FL client (standalone/2-client)
│       │   ├── data_setup_nih.py       # Data partitioning with label-flip injection
│       │   └── param_grid.py           # Experiment parameter grid
│       ├── standalone/                 # Stage 1a: standalone model training
│       │   ├── run_experiments.py      # Runner (supports parallel workers)
│       │   ├── train_single_config.py  # Single-config training script
│       │   └── evaluate.py            # Evaluation on held-out test set
│       ├── federated_2clients/         # Stage 1b: 2-client federated experiments
│       │   ├── run_experiments_2clients.py
│       │   ├── server_nih_2clients.py
│       │   ├── client_nih_2clients.py
│       │   ├── experiment_manager_2clients.py
│       │   ├── evaluate_federated_2clients.py
│       │   ├── compute_client0_advantages_2clients.py
│       │   └── param_grid_2clients.py
│       ├── federated/                  # Stage 1c: N-client federated experiments (N=3,5,7)
│       │   ├── run_experiments_cumulative_Nclients.py
│       │   ├── server_nih_Nclients.py
│       │   ├── client_nih_Nclients.py
│       │   ├── experiment_manager_cumulative_Nclients.py
│       │   ├── evaluate_federated_{3,5,7}clients.py
│       │   ├── compute_client0_advantages_{3,5,7}clients.py
│       │   ├── data_setup_nih_Nclients.py
│       │   └── param_grid_cumulative_Nclients.py
│       ├── meta_learning/              # Stage 2: regression model training
│       │   ├── train_meta_regressors.py        # Candidate perspective
│       │   └── train_federation_regressors.py  # Federation perspective
│       └── experiments/               # Auto-created: progress tracking, configs
│
└── Fitzpatrick17k/
    └── repo_fitz/
        ├── src/                        # Shared source code
        │   ├── client_fitz.py
        │   ├── data_setup_fitz.py
        │   └── param_grid_fitz.py
        ├── standalone/                 # Stage 1a: standalone model training
        │   ├── run_standalone_fitz.py
        │   ├── train_standalone_fitz.py
        │   └── evaluate_standalone_fitz.py
        ├── federated/
        │   ├── 2clients/               # Stage 1b: 2-client federated experiments
        │   │   ├── run_experiments_fitz.py
        │   │   ├── server_fitz.py
        │   │   ├── experiment_manager_fitz.py
        │   │   └── evaluate_models_fitz.py
        │   └── Nclients/               # Stage 1c: N-client federated experiments
        │       ├── run_experiments_fitz_Nc.py
        │       ├── server_fitz_Nc.py
        │       ├── evaluate_models_fitz_Nc.py
        │       └── param_grid_fitz_Nc.py
        ├── meta_learning/              # Stage 2: regression model training
        │   ├── train_meta_regressors.py        # Candidate perspective
        │   └── train_federation_regressors.py  # Federation perspective
        ├── data/                       # Place dataset files here (see Data Setup)
        ├── experiments/                # Auto-created: progress tracking
        └── single_models/              # Auto-created: saved standalone models
```

---

## Method

The framework proceeds in two stages.

**Stage 1 — Simulation.** For a broad range of configurations parameterized by dataset size, demographic composition, and label bias, three model types are trained per configuration:
1. A **standalone** model on the candidate client alone — $M(\{c\})$
2. A **federated** model on the candidate and federation jointly — $M(\mathcal{F} \cup \{c\})$
3. A **federated** model on the existing federation alone — $M(\mathcal{F})$

Metric deltas are then computed from these three models for both the candidate perspective ($\Delta^\text{cand}$) and the federation perspective ($\Delta^\text{ext}$).

**Stage 2 — Regression.** The simulation outputs populate two parallel meta-datasets (one per perspective). A suite of regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost) is trained via 5-fold cross-validation with randomized hyperparameter search to predict federation-induced metric changes for unseen configurations.

**Client metadata.** Each client is described by three scalars: `size` (number of training samples), `comp` (fraction of majority-group samples), and `bias` (label bias rate — fraction of minority-group positive samples whose label is flipped to negative). The federation is described by size-weighted aggregates of these same quantities.

---

## Data Setup

### NIH ChestX-ray14

1. Download the dataset from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIH).
2. Place the images and metadata files under `NIH/repo_nih/src/data/`.

**Experiment parameters:**
- Training set fractions (`portion`): 3%, 6%, 12% of the full dataset
- Gender compositions (`comp`): 100% male, 50/50, 100% female
- Label flip fractions (`bias`): 0.0, 0.15, 0.30, 0.45 (applied to female positives; forced to 0.0 when no females present)
- Total unique standalone configurations: 27
- Total 2-client federated configurations: 729 (full enumeration)
- 3-client and 5-client configurations: 800 each (stratified sampling)

### Fitzpatrick17k

1. Download the dataset from the [Fitzpatrick17k repository](https://github.com/mattgroh/fitzpatrick17k).
2. Place the images and CSV metadata under `Fitzpatrick17k/repo_fitz/data/`.

**Experiment parameters:**
- Training set fractions (`portion`): 50%, 60%, 70% of the per-group balanced pool
- Skin tone compositions: 100% light, 75% light / 25% dark, 50% / 50%
- Label flip fractions (`bias`): 0.0, 0.15, 0.30, 0.45 (applied to dark-skin positives; forced to 0.0 for pure-light-skin clients)

---

## Requirements

```bash
pip install torch torchvision flwr scikit-learn xgboost pandas numpy matplotlib seaborn
```

Federated training uses [Flower (flwr)](https://flower.dev/) with gRPC communication. Each worker spawns a server process and client processes that communicate over a configurable port.

**Hardware.** GPU recommended. Workers can be parallelized across multiple GPUs by setting the `WORKER_ID` and `GPU_DEVICE` environment variables.

---

## Running the Experiments

All steps below apply to both domains (`NIH/repo_nih/` and `Fitzpatrick17k/repo_fitz/`). Paths shown are for NIH; substitute the Fitzpatrick equivalents as needed.

### Step 1 — Train standalone models

```bash
cd NIH/
python repo_nih/standalone/run_experiments.py
```

This trains one standalone model per unique client configuration and saves checkpoints to `repo_nih/single_models/`. Progress is tracked in `repo_nih/experiments/unique_configs_completed.json`, so the runner can be safely interrupted and resumed.

### Step 2 — Run 2-client federated experiments

```bash
cd NIH/

# Single worker
python repo_nih/federated_2clients/run_experiments_2clients.py

# Multiple parallel workers (one per terminal)
WORKER_ID=0 python repo_nih/federated_2clients/run_experiments_2clients.py
WORKER_ID=1 python repo_nih/federated_2clients/run_experiments_2clients.py
```

Each worker spawns its own server and client processes. Workers coordinate via file-locking to avoid duplicated experiments. Results are saved per-configuration under `repo_nih/experiments/`.

### Step 3 — Run N-client federated experiments (N = 3, 5)

```bash
cd NIH/

# 3-client federation
export NUM_CLIENTS=3
export MAX_SAMPLES=800
python repo_nih/federated/run_experiments_cumulative_Nclients.py

# 5-client federation
export NUM_CLIENTS=5
export MAX_SAMPLES=800
python repo_nih/federated/run_experiments_cumulative_Nclients.py
```

Configurations are sampled via stratified random sampling (stratified on the candidate client's parameters) to ensure balanced coverage across the candidate space.

### Step 4 — Compute federation advantages

```bash
# 2-client
python repo_nih/federated_2clients/compute_client0_advantages_2clients.py

# 3-client
python repo_nih/federated/compute_client0_advantages_3clients.py

# 5-client
python repo_nih/federated/compute_client0_advantages_5clients.py
```

These scripts compute $\Delta^\text{cand}$ (candidate vs. standalone) and $\Delta^\text{ext}$ (federation with vs. without candidate) for each experiment, producing CSV files consumed by Stage 2.

### Step 5 — Train meta-regressors (Stage 2)

**Candidate perspective:**
```bash
cd NIH/
python repo_nih/meta_learning/train_meta_regressors.py
```

**Federation perspective:**
```bash
cd NIH/
python repo_nih/meta_learning/train_federation_regressors.py
```

Both scripts automatically locate the advantage CSVs produced in Step 4 (paths can be overridden via environment variables `RESULTS_2C`, `RESULTS_3C`, `RESULTS_5C`). They output:
- Predicted vs. actual scatter plots
- Feature importance plots
- A `model_summary.csv` with CV R², RMSE, and MAE for every model–target combination

The same steps apply to the Fitzpatrick17k domain using the corresponding scripts under `Fitzpatrick17k/repo_fitz/`.

---

## Metrics

Performance and fairness are evaluated on a shared held-out test set (15% of each dataset).

| Metric | Description |
|--------|-------------|
| **AUC** | Area under the ROC curve (macro-averaged over pathologies for NIH) |
| **BalAcc** | Balanced accuracy |
| **TPRgap** | Difference in true positive rate between majority and minority group (lower = fairer) |
| **minTPR** | Minimum TPR across demographic groups |

For Fitzpatrick17k, where the dark-skin positive test set is small, a **soft TPR** variant (average predicted probability over positive instances) replaces the standard hard TPR for more stable regression targets.

All delta targets are sign-normalized so that **positive = improvement** across every metric.

---

## Reproducibility Notes

- Randomized hyperparameter search uses `random_state=42` throughout.
- Federated training uses FedAvg for 30 communication rounds with full client participation and 1 local epoch per round.
- Model selection uses lowest weighted validation loss across clients (weights proportional to local validation set sizes).
- The 7-client generalization experiment (Section 5.1 of the paper) uses models trained on 2/3/5-client data evaluated directly on 150 independently generated 7-client configurations, without retraining.

---

## Citation

This work has been accepted for publication at **CIKM 2026** (the 35th ACM International Conference on Information and Knowledge Management), Rome, Italy.

The full citation, DOI, and BibTeX entry will be added here once they are available from ACM.

If you use this code or build on this work in the meantime, please cite:

> M. Ceccon, A. Fabris, O. Irrera, G. Silvello, G. A. Susto. "Fair to Federate? Two-Sided Prediction of Performance and Fairness Outcomes in Healthcare Federated Learning." *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)*, Rome, Italy, 2026.
