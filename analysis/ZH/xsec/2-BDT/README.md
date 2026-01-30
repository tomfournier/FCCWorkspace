# 2-BDT: Multivariate Analysis & Model Training

Boosted Decision Tree (BDT) training and evaluation module for ZH signal/background discrimination in the FCC-ee cross-section analysis.

## Overview

This stage develops machine learning models to separate ZH signal events from background processes using kinematic features. The trained BDT models are subsequently used in the measurement stage ([3-Measurement](../3-Measurement/README.md)) for event selection and physics analysis.

## Workflow

The `2-BDT` module consists of three main scripts executed sequentially:

| Script | Purpose |
|--------|---------|
| **`process_input.py`** | Prepare training data from MVA input histograms |
| **`train_bdt.py`** | Train XGBoost models for each channel and energy combination |
| **`evaluation.py`** | Evaluate BDT performance and generate validation plots |

## Scripts

### 1. `process_input.py` — Training Data Preparation

Processes histograms from MVA input stage and prepares balanced training datasets for BDT training.

**Key operations:**
- Loads MVA input histograms for all processes (signal + backgrounds)
- Extracts kinematic variables: lepton momentum, polar angle, dilepton mass, momentum, acoplanarity, etc.
- Balances event counts between signal and background processes using cross-section and efficiency weighting
- Assigns signal/background labels and luminosity weights
- Saves preprocessed training data as pickle files

**Input:** MVA input histograms
```
output/data/histograms/MVAInputs/{cat}/{ecm}/{sel}/
```

**Output:** Preprocessed training data
```
output/data/MVA/{cat}/{ecm}/{sel}/preprocessed.pkl
```

**Usage:**
```bash
python process_input.py --cat ee --ecm 240
python process_input.py --cat mumu --ecm 365
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)

**Configuration:**
- Processes included:
  - **Signal:** ZH production
  - **Backgrounds:** ZZ, WW, Z+jets, radiative Z (e-γ, γ-e), diphoton (γγ)
- Variable selection: 9 kinematic features from `package.config.input_vars`
- Process dictionary: `FCCee_procDict_winter2023_training_IDEA.json`

For detailed information on utility functions used, see [package/func/bdt.py](../package/func/README.md).

---

### 2. `train_bdt.py` — BDT Model Training

Trains XGBoost models for each channel and selection strategy using prepared training data.

**Key operations:**
- Loads preprocessed training data
- Splits events into 50/50 training/validation sets with luminosity weighting
- Trains XGBoost models with early stopping on validation set
- Saves trained models in TMVA-compatible format for physics analysis
- Generates feature importance mapping

**Input:** Preprocessed training data from `process_input.py`
```
output/data/MVA/{cat}/{ecm}/{sel}/preprocessed.pkl
```

**Output:** Trained BDT models and metadata
```
output/data/MVA/{cat}/{ecm}/{sel}/
  ├── bdt_model.pkl           # Trained XGBoost model
  ├── variables.json          # Input variable names
  └── feature.txt             # Feature map for TMVA evaluation
```

**Usage:**
```bash
python train_bdt.py --cat ee --ecm 240
python train_bdt.py --cat mumu --ecm 365
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)

**XGBoost Configuration:**
```python
n_estimators: 350          # Number of boosting rounds
learning_rate: 0.20        # Step size shrinkage (eta)
max_depth: 3               # Maximum tree depth
subsample: 0.5             # Subsample ratio of training instances
gamma: 3                   # Minimum loss reduction (regularization)
min_child_weight: 10       # Minimum child weight (regularization)
colsample_bytree: 0.5      # Subsample ratio of columns per tree
```

**Early Stopping:** Training stops after 25 rounds without validation set improvement.

**Training Strategy:**
- Uses signal vs combined background classification (binary)
- Applies luminosity weights to training samples
- Validates on separate dataset to prevent overfitting

For detailed information on training functions, see [package/func/bdt.py](../package/func/README.md).

---

### 3. `evaluation.py` — BDT Performance Evaluation

Evaluates trained BDT models and generates diagnostic plots for validation.

**Key operations:**
- Loads trained models and validation datasets
- Computes classification performance metrics (AUC, efficiency, purity)
- Generates BDT score distributions for signal and background
- Creates decision tree visualizations
- Produces input variable distributions in high/low BDT score regions

**Input:** Trained BDT models and MVA input data
```
output/data/MVA/{cat}/{ecm}/{sel}/
output/data/histograms/MVAInputs/{cat}/{ecm}/{sel}/
```

**Output:** Evaluation plots and metrics
```
output/plots/evaluation/{cat}/{ecm}/
  ├── metrics/                    # Classification performance plots
  ├── trees/                      # Decision tree visualizations
  ├── variables/                  # Input variable distributions
  └── high_low/                   # High/low BDT score regions
```

**Usage:**
```bash
python evaluation.py --cat ee --ecm 240                    # Basic evaluation
python evaluation.py --cat ee --ecm 240 --metric           # Skip metric plots
python evaluation.py --cat ee --ecm 240 --tree             # Plot decision trees
python evaluation.py --cat ee --ecm 240 --check            # Plot input variables
python evaluation.py --cat ee --ecm 240 --hl               # Plot high/low regions
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)
- `--metric`: Generate classifier performance metrics plots
- `--tree`: Visualize individual decision trees from the ensemble
- `--check`: Plot distributions of input variables used for training
- `--hl`: Plot variable distributions in high/low BDT score regions

**Metrics Computed:**
- Receiver Operating Characteristic (ROC) curve
- AUC (Area Under Curve) for binary classification
- Signal efficiency and background rejection at various BDT score thresholds
- Confusion matrix

For detailed information on evaluation functions, see [package/func/bdt.py](../package/func/README.md).

---

## Integration with Analysis Pipeline

The BDT models trained in this stage are used in [3-Measurement](../3-Measurement/README.md) for:
- **Signal/background discrimination:** BDT scores define baseline and control regions
- **Event categorization:** High BDT scores (signal-like) vs low BDT scores (background-like) for systematic studies
- **Physics selection:** BDT-based selection improves measurement sensitivity

## Automation

To run all steps automatically across all channel/energy combinations:
```bash
python run/2-run.py
```

For details on the runner script, see [run/README.md](../run/README.md).

## Key Functions

Core BDT functionality is implemented in the `package` module:
- **Data preparation:** [package/func/bdt.py](../package/func/README.md#data-preparation--loading) — `counts_and_effs`, `BDT_input_numbers`, `df_split_data`
- **Model training:** [package/func/bdt.py](../package/func/README.md#model-training--evaluation) — `train_model`, `save_model`, `load_model`
- **Evaluation:** [package/func/bdt.py](../package/func/README.md#model-training--evaluation) — `evaluate_bdt`, `get_metrics`
- **Plotting:** [package/plots/eval.py](../package/plots/README.md) — BDT score and metric visualization

See [package/func/README.md](../package/func/README.md) and [package/config.py](../package/README.md#configpy--physics-configuration) for detailed documentation.

## Output Structure

```
output/data/MVA/{cat}/{ecm}/{sel}/
├── bdt_model.pkl              # Trained XGBoost model (binary format)
├── variables.json             # Input variable names (JSON)
└── feature.txt                # Feature map for TMVA (TSV)

output/plots/evaluation/{cat}/{ecm}/
├── metrics/                   # ROC curves, AUC, efficiency plots
├── trees/                     # Decision tree visualizations (PNG)
├── variables/                 # Input variable distributions
└── high_low/                  # High/low BDT score region distributions
```

Where:
- `{cat}`: Channel (`ee` or `mumu`)
- `{ecm}`: Center-of-mass energy (240 or 365)
- `{sel}`: Selection strategy (`Baseline`, `Baseline_miss`, `Baseline_sep`)
