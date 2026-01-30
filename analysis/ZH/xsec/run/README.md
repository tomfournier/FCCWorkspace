# Analysis Pipeline Scripts

This directory contains wrapper scripts that automate the complete ZH analysis pipeline. Each script orchestrates a specific stage of the analysis across multiple channels and center-of-mass energies.

## Overview

The analysis runs sequentially through five stages, each with its own runner script:

1. **1-run.py** – MVA Input Generation & Plotting
2. **2-run.py** – BDT Training & Evaluation
3. **3-run.py** – Physics Measurement
4. **4-run.py** – Histogram Combination
5. **5-run.py** – Statistical Fit & Bias Testing

## Usage

All scripts must be executed from the **xsec/** folder to properly locate the `package/` module:

```bash
cd /path/to/xsec/
python run/<N>-run.py [--cat CHANNEL] [--ecm ENERGY] [--run STAGES] [--script-args]
```

### Common Parameters

- `--cat`: Lepton channel (`ee`, `mumu`, `ee-mumu`, `mumu-ee`)
  - Default: `ee-mumu` (runs both sequentially)
  - Example: `--cat ee` (electron channel only)
  - Example: `--cat mumu-ee` (runs mumu first, then ee)

- `--ecm`: Center-of-mass energy in GeV (`240`, `365`, `240-365`, `365-240`)
  - Default: `240-365` (runs 240 then 365 sequentially)
  - Example: `--ecm 365` (high-energy only)
  - Example: `--ecm 365-240` (runs 365 first, then 240)

- `--run`: Pipeline stages to execute (dash-separated for multiple)
  - Stage options depend on the script (see below)
  - Default varies per script
  - Example: `--run 1-2` (first two stages)

### Script-Specific Arguments

Each script may support additional options. Use `--help` to see all available arguments:

```bash
python run/1-run.py --help
```

**Example:** Running 1-run.py with specific options:
```bash
python run/1-run.py --cat ee --ecm 240 --run 1-2 [additional script args]
```

Refer to individual script docstrings for script-specific arguments and their usage patterns.

### Stage Dependencies

⚠️ **Warning:** Running with `--run 2` (or higher stages) without first executing `--run 1` will raise an error, as each stage depends on the outputs of previous stages. Always run stages sequentially if it's your first time:

```bash
# Correct: Run all stages in order
python run/1-run.py --cat ee --ecm 240 --run 1-2-3

# Incorrect: Will fail
python run/1-run.py --cat ee --ecm 240 --run 2
```

## Script Details

### 1-run.py – Event Selection & MVA Inputs
Performs pre-selection and final-selection cuts, then generates MVA input histograms.

**Stages:** 1=pre-selection, 2=final-selection, 3=plots (default: 2-3)

```bash
python 1-run.py --cat ee --ecm 240 --run 1-2-3
```

### 2-run.py – Boosted Decision Tree
Processes MVA inputs, trains BDT models, and evaluates performance.

**Stages:** 1=process_input, 2=train_bdt, 3=evaluation (default: 1-2-3)

```bash
python 2-run.py --cat ee-mumu --ecm 240-365 --run 1-3
```

### 3-run.py – Physics Measurement
Applies BDT selection to events, generates measurement plots, and produces cutflow tables.

**Stages:** 1=pre-selection, 2=final-selection, 3=plots, 4=cutflow (default: 2-3)

```bash
python 3-run.py --cat mumu --ecm 365 --run 1-2-3-4
```

### 4-run.py – Histogram Combination
Merges histograms across channels and selections, prepares input for fitting.

**Stages:** 1=process_histogram, 2=combine (default: 2)

```bash
python 4-run.py --cat ee-mumu --ecm 240-365 --run 1-2
```

### 5-run.py – Statistical Fit & Validation
Performs maximum-likelihood fits and bias tests to extract physics parameters.

**Stages:** 1=fit, 2=bias_test (default: 1-2)

**Important:** Due to environment incompatibility, 5-run.py must be executed in a **separate terminal** from the other scripts.

```bash
python 5-run.py --cat ee --ecm 240 --run 2
```

## Key Features

- **Batch Execution**: Automatically loops over multiple channels/energies
- **Temporary Config**: Creates job-specific JSON configs passed to downstream scripts
- **Environment Flag**: Sets `RUN='1'` to indicate automated mode
- **Streaming Output**: Child process output printed in real-time
- **Modular Design**: Run individual stages or complete pipelines as needed

## Complete Pipeline Example

Run the full analysis from event selection through fitting. Execute in the xsec/ folder:

```bash
cd /path/to/xsec/
python run/1-run.py --cat ee-mumu --ecm 240-365  # MVA inputs
python run/2-run.py --cat ee-mumu --ecm 240-365  # BDT training
python run/3-run.py --cat ee-mumu --ecm 240-365  # Measurement
python run/4-run.py --cat ee-mumu --ecm 240-365  # Combine
```

Then in a **separate terminal** (due to environment compatibility):

```bash
cd /path/to/xsec/
python run/5-run.py --cat ee-mumu --ecm 240-365  # Fit
```

Or run with custom selections:

```bash
python run/1-run.py --cat ee --ecm 365 --run 1-2
python run/2-run.py --cat ee --ecm 365 --run 1-3
python run/3-run.py --cat ee --ecm 365 --run 2-3-4
python run/4-run.py --cat ee --ecm 365
```

Then separately:

```bash
python run/5-run.py --cat ee --ecm 365
```

## Output Structure

Each script generates outputs in the `output/` directory:
- `data/` – Processed events and histograms
- `plots/` – Generated plots and distributions
- `tmp/` – Temporary configuration files, process dictionaries, and sample metadata
  - `config_json/` – Job-specific configuration files (cleaned up after each stage)
  - `procDict/` – Process dictionaries used to extract sample metadata during pre-selection
  - `events.json` – Events file generated by the cutflow stage (3-run.py --run 4)

## Configuration

Analysis parameters are defined in [package/userConfig.py](../package/userConfig.py) and [package/config.py](../package/config.py).
