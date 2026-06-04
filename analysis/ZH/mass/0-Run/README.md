# Analysis Pipeline Scripts

This directory contains wrapper scripts that automate the complete ZH analysis pipeline. Each script orchestrates a specific stage of the analysis across multiple channels and center-of-mass energies, handling batch execution with proper logging and error tracking.

## Overview

The analysis pipeline consists of five main stages (plus two optional study pipelines):

**Main Pipeline:**
1. **1-run.py** – MVA Input Generation
2. **2-run.py** – Boosted Decision Tree (BDT) Training & Evaluation
3. **3-run.py** – Physics Measurement & Cutflow
4. **4-run.py** – Histogram Combination & Datacards
5. **5-run.py** – Statistical Fit & Bias Testing

**Study Pipelines:**
- **a-run.py** – FSR (Final State Radiation) Optimization
- **b-run.py** – BDT Hyperparameter Optimization

## Usage

All scripts must be executed from the **xsec/** root folder to properly locate the `package/` module:

```bash
cd /path/to/xsec/
python 0-Run/<N>-run.py [--cat CHANNEL] [--ecm ENERGY] [--run STAGES] [OPTIONS]
```

### Common Parameters

All runner scripts accept the following parameters:

- `--cat`: Lepton channel(s) (`ee`, `mumu`, or combinations)
  - Single: `--cat ee` (electron channel only)
  - Multiple: `--cat ee-mumu` (runs ee first, then mumu sequentially)
  - Default varies per script

- `--ecm`: Center-of-mass energy(ies) in GeV (`240` or `365`, or combinations)
  - Single: `--ecm 365` (high-energy only)
  - Multiple: `--ecm 240-365` (runs 240 first, then 365 sequentially)
  - Default varies per script

- `--run`: Pipeline stages to execute (dash-separated for multiple)
  - Stage numbers depend on the script
  - Default varies per script
  - Example: `--run 1-2` (first two stages only)

## Script Details

### 1-run.py – MVA Input Generation

Performs event selection and generates histograms with BDT input variables. Uses `fccanalysis` subcommands.

**Stages:**
- Stage 1 = `pre-selection`: Apply pre-selection cuts, compute kinematic variables
- Stage 2 = `final-selection`: Fill histograms with BDT variables  
- Stage 3 = `plots`: Generate distribution plots

**Default:** `--run 2-3` (skips pre-selection by default)

**Examples:**
```bash
# Pre-selection, final selection, and plots for both channels and energies
python 1-run.py --cat ee-mumu --ecm 240-365 --run 1-2-3

# Only final selection and plots for ee channel at 240 GeV
python 1-run.py --cat ee --ecm 240

# Pre-selection only for ee at 365 GeV
python 1-run.py --cat ee --ecm 365 --run 1
```

### 2-run.py – Boosted Decision Tree (BDT) Training & Evaluation

Processes MVA input histograms, trains XGBoost models, and evaluates BDT performance. Supports optional evaluation metrics.

**Stages:**
- Stage 1 = `process_input`: Load histograms, balance samples, prepare for BDT training
- Stage 2 = `train_bdt`: Train XGBoost classifier with early stopping
- Stage 3 = `evaluation`: Evaluate BDT performance, generate plots

**Default:** `--run 1-2-3` (all stages)

**Optional Flags:**
- `--metric`: Generate BDT performance metrics plots (used with stage 3)
- `--tree`: Draw BDT decision trees (used with stage 3)
- `--check`: Check variable distributions (used with stage 3)
- `--hl`: Additional evaluation options (used with stage 3)
- `--sels SELECTIONS`: Specific selection strategies to process

**Examples:**
```bash
# Standard training and evaluation for both channels
python 2-run.py --cat ee-mumu --ecm 240-365

# Train and evaluate with metric plots
python 2-run.py --cat ee --ecm 240 --run 2-3 --metric

# Process and train only (no evaluation)
python 2-run.py --cat mumu --ecm 365 --run 1-2
```

### 3-run.py – Physics Measurement & Cutflow

Applies BDT selection to events, generates measurement plots, and analyzes event yields per cut stage.

**Stages:**
- Stage 1 = `pre-selection`: Apply pre-selection cuts, compute kinematic variables
- Stage 2 = `final-selection`: Apply BDT selection, fill measurement histograms
- Stage 3 = `plots`: Generate distribution plots
- Stage 4 = `cutflow`: Analyze event yields and generate cutflow tables

**Default:** `--run 2-3` (final-selection and plots)

**Optional Flags:**
- `--sels SELECTIONS`: Specific selection strategies
- `--test`: Use test sample for validation
- Plotting options (see script docstring)

**Examples:**
```bash
# Full measurement including cutflow analysis
python 3-run.py --cat ee-mumu --ecm 240-365 --run 1-2-3-4

# Measurement with plots only
python 3-run.py --cat ee --ecm 365

# Pre-selection and measurement without plots
python 3-run.py --cat mumu --ecm 240 --run 1-2
```

### 4-run.py – Histogram Combination & Datacards

Merges histograms across channels and selection strategies, splits into high/low BDT score regions, and creates combine datacards for fitting.

**Stages:**
- Stage 1 = `process_histogram`: Split histograms into high/low BDT regions
- Stage 2 = `combine`: Create combine datacards from processed histograms

**Default:** `--run 1-2` (both stages)

**Selection Strategies:**
By default processes: `Baseline`, `Baseline_miss`, `Baseline_sep`, `test`
Customize with: `--sels SELECTION1-SELECTION2-...`

**Optional Flags:**
- `--sels SELECTIONS`: Specific selection strategies to process
- `--polL` / `--polR`: Polarization left/right (for stage 1)
- `--ILC`: ILC-specific settings (for stage 1)

**Examples:**
```bash
# Process and combine for both channels and energies with default selections
python 4-run.py --cat ee-mumu --ecm 240-365

# Process only (prepare histograms)
python 4-run.py --cat ee --ecm 240 --run 1

# Combine only (requires pre-processed histograms) for specific selections
python 4-run.py --cat ee-mumu --ecm 240-365 --run 2 --sels Baseline
```

### 5-run.py – Statistical Fit & Bias Testing

Performs maximum-likelihood fits to extract physics parameters and runs bias tests to validate fit performance.

**Stages:**
- Stage 1 = `fit`: Perform ML fit and extract parameters
- Stage 2 = `bias_test`: Run bias test with pseudo-experiments

**Default:** `--run 1-2` (fit and bias test)

**Optional Flags:**
- `--combine`: Include combined channel fit ('comb')
- `--sels SELECTIONS`: Specific selection strategies
- Fit options: `--quiet`, timing options, etc. (see script docstring)
- Bias test options: pseudo-experiment count, etc. (see script docstring)

**⚠️ Important:** This script may require a separate terminal or environment due to compatibility issues with fitting libraries.

**Examples:**
```bash
# Fit and bias test for all channels and energies
python 5-run.py --cat ee-mumu --ecm 240-365

# Fit only (no bias test)
python 5-run.py --cat ee --ecm 365 --run 1

# Fit with combined channel and bias test
python 5-run.py --cat ee-mumu --ecm 240-365 --run 1-2 --combine
```

### a-run.py – FSR (Final State Radiation) Optimization

Studies the impact of FSR corrections and optimizes FSR-related selection criteria.

**Stages:**
- Stage 1 = `pre-selection`: Apply pre-selection cuts with FSR studies
- Stage 2 = `plots`: Generate FSR-related distribution plots

**Default:** `--run 1-2` (both stages)

**Optional Flags:**
- `--sels SELECTIONS`: Specific selection strategies
- FSR optimization options (see script docstring)

**Examples:**
```bash
# Full FSR study for both channels
python a-run.py --cat ee-mumu --ecm 240-365

# Pre-selection only for FSR studies
python a-run.py --cat ee --ecm 240 --run 1
```

### b-run.py – BDT Hyperparameter Optimization

Performs chi-squared optimization to find optimal BDT hyperparameters and selection criteria.

**Stages:**
- Stage 1 = `pre-selection`: Apply pre-selection cuts for optimization studies
- Stage 2 = `optimize`: Run BDT hyperparameter optimization
- Stage 3 = `plots`: Generate optimization result visualization plots

**Default:** `--run 1-2-3` (all stages)

**Optional Flags:**
- `--sels SELECTIONS`: Specific selection strategies
- Optimization options (see script docstring)

**Examples:**
```bash
# Full optimization pipeline for both channels
python b-run.py --cat ee-mumu --ecm 240-365

# Optimization results only (pre-selection and optimization done)
python b-run.py --cat ee --ecm 365 --run 2-3

# Pre-selection and optimization only
python b-run.py --cat mumu --ecm 240 --run 1-2
```

## Key Features

- **Batch Execution**: Automatically iterates over multiple channels and energies in nested loops
- **Temporary Configuration**: Creates job-specific JSON configs passed to downstream scripts
- **Environment Flags**: Sets `RUN='1'` (and optionally `RUN_BATCH='1'`) to indicate automated execution mode
- **Streaming Output**: Child process output piped in real-time to terminal
- **Error Handling**: Stops pipeline on first error (non-zero exit code)
- **Logging**: Clear execution markers with status (✓ COMPLETED / ✗ FAILED) and timing
- **Modular Design**: Run individual stages or complete pipelines as needed

## Complete Pipeline Example

Run the full analysis from start to finish (execute in xsec/ folder):

```bash
cd /path/to/xsec/

# Stage 1-4: Main analysis pipeline (can run in same terminal)
python 0-Run/1-run.py --cat ee-mumu --ecm 240-365  # MVA inputs
python 0-Run/2-run.py --cat ee-mumu --ecm 240-365  # BDT training  
python 0-Run/3-run.py --cat ee-mumu --ecm 240-365  # Measurement
python 0-Run/4-run.py --cat ee-mumu --ecm 240-365  # Combine
```

Then (optionally in a separate terminal):

```bash
cd /path/to/xsec/
python 0-Run/5-run.py --cat ee-mumu --ecm 240-365  # Fit & bias test
```

### Custom Analysis Examples

**Quick validation (test sample):**
```bash
python 0-Run/1-run.py --cat ee --ecm 240 --test --run 2-3
```

**Full analysis with one channel:**
```bash
python 0-Run/1-run.py --cat ee --ecm 240-365 --run 1-2-3
python 0-Run/2-run.py --cat ee --ecm 240-365
python 0-Run/3-run.py --cat ee --ecm 240-365 --run 1-2-3-4
python 0-Run/4-run.py --cat ee --ecm 240-365
python 0-Run/5-run.py --cat ee --ecm 240-365
```

**BDT studies only:**
```bash
python 0-Run/a-run.py --cat ee-mumu --ecm 240-365  # FSR studies
python 0-Run/b-run.py --cat ee-mumu --ecm 240-365  # Optimization studies
```

## Output Structure

Each script generates outputs in the `output/` directory under the workspace root:

- **output/data/** – Processed events and histograms
  - `events/` – Event data after selection cuts (stage-specific)
  - `MVA/` – MVA input histograms per channel/energy
  - `histograms/` – Processed histograms (MVAInputs, measurement, combine stages)
  - `combine/` – Combine datacards and inputs

- **output/plots/** – Generated plots and distributions
  - `MVAInputs/`, `evaluation/`, `measurement/`, etc. – Stage-specific plots
  - `fsr/`, `optimisation/` – Special study outputs

- **output/tmp/** – Temporary files cleaned up after execution
  - `config_json/` – Job-specific JSON configurations (removed after each job)
  - `procDict/` – Process dictionaries and sample metadata

## Environment Setup

Scripts import configuration and utilities from the `package/` module:

- **[package/config.py](../package/config.py)** – Analysis parameters and timing utilities
- **[package/userConfig.py](../package/userConfig.py)** – Directory paths (`loc.ROOT`, etc.)
- **[package/parsing.py](../package/parsing.py)** – Argument parsing and configuration handling
- **[package/logger.py](../package/logger.py)** – Logging and terminal output formatting

## Help & Troubleshooting

Get detailed help for any script:

```bash
python 1-run.py --help  # Shows all available arguments
```

Common issues:

- **"Stage dependencies" errors**: Run stages in order (1, then 2, then 3, etc.) unless outputs already exist
- **"RUN='1' not detected"**: The scripts should automatically set this flag; check `package/parsing.py` if issues persist
- **Missing outputs**: Check that the parent stage completed successfully (look for ✓ COMPLETED marker)
- **Environment issues**: Some stages may require different Python environments; use separate terminal if needed
