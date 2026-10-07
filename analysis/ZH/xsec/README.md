# FCC-ee ZH Cross-Section Analysis

This repository contains the analysis code for measuring the $e^+e^- \to ZH$ cross-section at the Future Circular Collider (FCC-ee) at two center-of-mass energies: 240 GeV and 365 GeV.

## Overview

The analysis workflow follows a sequential pipeline consisting of six main stages:

## Project Structure

### Analysis Stages

Each stage is contained in its own directory with a dedicated README:

- **`0-Run/`** — Wrapper orchestration scripts (`1-run.py` through `5-run.py`). These automate sequential execution across channels and energies. Execute these scripts from the xsec/ directory (not to confuse with the xsec/ module).

- **`1-MVAInputs/`** — Selects events from raw simulation and prepares kinematic variables for BDT training. Produces histograms of distributions for all processes (signal and backgrounds).

- **`2-BDT/`** — Trains XGBoost models to distinguish ZH signal from backgrounds using kinematic features. Evaluates model performance and determines optimal selection thresholds.

- **`3-Measurement/`** — Applies BDT classification to measurement samples and generates histograms of the Higgs recoil mass across signal-like and background-like regions.

- **`4-Combine/`** — Processes histograms and generates Combine-compatible datacards with systematic uncertainties for statistical fitting.

- **`5-Fit/`** — Performs maximum likelihood fits to extract the ZH cross-section. Validates fitting procedures through bias tests using pseudo-experiments.

- **`6-Self-coupling`** - Performs maximum likelihood fits and SMEFT models to extract the Higgs self-coupling precision

### Core Components

- **`xsec/`** — Central Python module providing configuration, utilities, and analysis functions used across all stages. Key components:
  - `config.py` — Physics constants, process definitions, and color palettes
  - `logger.py` — Logging configuration
  - `parsing.py` — Unified command-line argument parsing
  - `userConfig.py` — Path templates and global parameters (luminosity, energies)
  - `run.py` - Utilities for `0-Run/` folder scripts
  - `func/` — Analysis functions (BDT training/evaluation, bias testing)
  - `plots/` — Visualization utilities (cutflow, BDT evaluation, histogram plots)
  - `tools/` — Data processing (ROOT I/O, histogram manipulation)
  
  See [xsec/README.md](xsec/README.md) for detailed documentation.

- **`sel_xsec/`** — Event selection functions applied throughout the pipeline:
  - `sel_xsec/presel/` — Pre-selection functions (lepton/quark selection, kinematic cuts)
  - `sel_xsec/final/` — Final selection functions (histogram definitions, variable computation)
  
  See [sel_xsec/presel/README.md](sel_xsec/presel/README.md) for selection documentation.

- **`output/`** — Auto-generated directory containing all analysis outputs (see [Output Structure](#output-structure) below)

## Getting Started

### Prerequisites

- FCC software stack with FCCAnalysis framework (for stages 1-4)
- RooFit/Combine tools (for stage 5 statistical fitting, optional)

### Software Frameworks

**FCCAnalyses** (Stages 1-4)  
FCCAnalyses is the main analysis framework used for stages 1-4. It provides tools for event selection, reconstruction, and histogram generation with integration to FCC simulation and detector simulation. This framework is required for running MVA input preparation, BDT training, and physics measurement stages.

**RooFit/Combine-Limit** (Stage 5)  
RooFit and Combine are the statistical analysis tools used for maximum likelihood fitting and cross-section extraction. They are typically incompatible with the FCCAnalyses environment due to conflicting dependencies.

⚠️ **Environment Management Note:**  
If you encounter version incompatibility issues between FCCAnalyses and RooFit/Combine, consider:
- Running stages 1-4 in one environment (FCCAnalyses)
- Running stage 5 in a separate terminal with a different Python environment (RooFit/Combine)

### Running the Analysis

All scripts must be executed from the **xsec/** directory. The analysis is orchestrated through wrapper scripts in the `0-Run/` directory that automate sequential execution across channels and energies:

```bash
cd xsec/

# Stage 1: Event selection and MVA input preparation
python 0-Run/1-run.py --cat ee-mumu --ecm 240-365

# Stage 2: BDT training and evaluation
python 0-Run/2-run.py --cat ee-mumu --ecm 240-365

# Stage 3: Physics measurement with BDT classification
python 0-Run/3-run.py --cat ee-mumu --ecm 240-365

# Stage 4: Histogram processing and datacard generation
python 0-Run/4-run.py --cat ee-mumu --ecm 240-365

# Stage 5: Statistical fitting (see note below)
python 0-Run/5-run.py --cat ee-mumu --ecm 240-365
```

#### Argument Syntax

The wrapper scripts use flexible, modular argument parsing:

- **`--cat CHANNELS`**: Decay categories (single or multiple)
  - Single: `--cat ee` or `--cat mumu`
  - Multiple: `--cat ee-mumu` (dash-separated)
  
- **`--ecm ENERGIES`**: Center-of-mass energies (single or multiple)
  - Single: `--ecm 240` or `--ecm 365`
  - Multiple: `--ecm 240-365` (dash-separated)
  
- **`--run STAGES`**: Pipeline stages to execute (default varies by script)
  - Single stage: `--run 1` or `--run 2`
  - Multiple stages: `--run 1-2-3` (dash-separated)
  
- **`--sels SELECTIONS`**: Specific selection strategies to process (batch mode)
  - Default: `Baseline` and `test` selections
  - Multiple: `--sels Baseline-test-other`

#### Example Commands

```bash
# Pre-selection only (Stage 1)
python 0-Run/1-run.py --cat ee --ecm 240 --run 1

# MVA input preparation for both channels and energies
python 0-Run/1-run.py --cat ee-mumu --ecm 240-365

# BDT training with metric plots
python 0-Run/2-run.py --cat mumu --ecm 365 --run 2-3 --metric

# Full measurement pipeline
python 0-Run/3-run.py --cat ee-mumu --ecm 240-365 --run 1-2-3-4 --yields

# Specific selection strategies
python 0-Run/4-run.py --cat ee --ecm 240 --sels Baseline-test
```

#### Stage Details

Each stage has internal sub-stages (1, 2, 3, ...) corresponding to specific scripts:

- **Stage 1-MVAInputs:** 1=pre-selection, 2=final-selection, 3=plots
- **Stage 2-BDT:** 1=process input, 2=train model, 3=evaluate
- **Stage 3-Measurement:** 1=pre-selection, 2=final-selection, 3=plots, 4=cutflow
- **Stage 4-Combine:** 1=process histograms, 2=create datacards
- **Stage 5-Fit:** 1=fit, 2=bias test, options: `--combine`, `--num 100`

## Output Structure

The `output/` directory is automatically created and populated during analysis execution. It contains:

```
output/
├── data/                            # All analysis data outputs
│   ├── events/                      # Pre-selected event trees
│   │   └── {ecm}/{cat}/full/{stage}/
│   │       ├── training/            # Events for MVA training (1-MVAInputs)
│   │       └── analysis/            # Events for physics analysis (3-Measurement)
│   │
│   ├── histograms/                  # Processed histograms at each stage
│   │   ├── MVAInputs/{ecm}/{cat}/
│   │   │   ├── *_{sel}.root         # ROOT files for BDT training
│   │   │   └── *_{sel}_histo.root   # Kinematic distributions for all processes
│   │   ├── preprocessed/{ecm}/{cat}/{sel}/
│   │   │   └── *_histo.root         # Measurement histograms after BDT selection
│   │   └── processed/{ecm}/{cat}/{sel}/
│   │       └── {sample}.root        # Final histograms before datacard generation
│   │
│   ├── MVA/                         # BDT model outputs
│   │   └── {ecm}/{cat}/{sel}/
│   │       ├── BDT/
│   │       │   ├── xgb_bdt.root     # Trained BDT model (ROOT format)
│   │       │   ├── xgb_bdt.joblib   # Trained BDT model (sklearn format)
│   │       │   ├── feature.txt      # Feature importance mapping
│   │       │   └── BDT_cut.txt      # Optimal BDT threshold
│   │       └── MVAInputs/
│   │           └── preprocessed.pkl # Training data (before BDT training)
│   │
│   └── combine/                     # RooFit/Combine framework outputs
│       └── {sel}/{ecm}/{cat}/
│           ├── nominal/             # Nominal fit results
│           │   ├── WS/              # RooFit workspaces
│           │   ├── datacard/        # Combine datacards
│           │   ├── log/             # Fit logs
│           │   └── results/         # Extracted signal strengths
│           └── bias/                # Bias test results
│               ├── WS/              # Pseudo-data workspaces
│               ├── datacard/        # Pseudo-data datacards
│               ├── log/             # Fit logs
│               └── results/         # Bias analysis and plots
│
├── plots/                           # Analysis plots for validation
│   ├── MVAInputs/{ecm}/{cat}/       # Kinematic distributions
│   ├── evaluation/{ecm}/{cat}/      # BDT performance plots
│   └── measurement/{ecm}/{cat}/     # Physics measurement plots
│
└── tmp/                             # Temporary files (config JSON, process dicts)
```

**Key outputs by analysis stage:**

| Stage | Input | Main Output | Purpose |
|-------|-------|-------------|---------|
| **1-MVAInputs** | Raw simulation (EDM4Hep) | Event trees, kinematic histograms | Prepare data for BDT training |
| **2-BDT** | MVA input histograms | Trained XGBoost model, feature importance | Train signal/background classifier |
| **3-Measurement** | Raw simulation + trained BDT model | Recoil mass histograms (signal/background regions) | Measure physics distributions |
| **4-Combine** | Measurement histograms | RooFit/Combine datacards with uncertainties | Prepare input for statistical fitting |
| **5-Fit** | Combine datacards | Signal strengths, cross-section, bias estimates | Extract cross-section and validate fits |
| 6-Self-coupling | Combine datacards | SMEFT coefficient precision, Higgs self-coupling | Extract Higgs self-coupling precision |

The `output/` directory is listed in `.gitignore` and will not be tracked by git. Regenerate outputs by re-running the analysis stages.

## Development & Testing

### Creating Tests

For testing new functionality or debugging analysis code, use the `test/` directory:

```bash
mkdir -p test/my_test/
# Add test scripts and minimal test data to test/my_test/
```

The `test/` directory is configured in `.gitignore` and will not be tracked by git, making it ideal for development and troubleshooting.

### Analysis Code Organization

The analysis is organized into logical components:

- **`xsec/config.py`** — Physics constants (masses, decay modes), process definitions, color palettes, kinematic variable names
- **`xsec/logger.py`** — Logging setup and configuration
- **`xsec/parsing.py`** — Unified command-line argument parsing used by all scripts
- **`xsec/userConfig.py`** — Path templates and global parameters (luminosity, channel names, data fractions)
- **`xsec/func/`** — Utilities for the different stages of the analysis (BDT training, fit, bias test, self-coupling models)
- **`xsec/plots/`** — Visualization utilities (cutflow plots, evaluation metrics, histogram plots)
- **`xsec/tools/`** — Data processing utilities (ROOT I/O, histogram manipulation, significance calculations)
- **`sel_xsec/presel/`** — Pre-selection functions (lepton kinematics, event filters)
- **`sel_xsec/final/`** — Final selection and histogram definitions (binning, variable mapping)

See [xsec/README.md](xsec/README.md) and [sel_xsec/presel/README.md](sel_xsec/presel/README.md) for detailed documentation.

## Physics Process

This analysis measures the cross-section for Higgs-strahlung production at $e^+e^-$ colliders:

$$e^+e^- \to Z(\to f\bar{f}) H(\to \text{all})$$

where $f \in \{e, \mu, q\}$. The measurement uses:

- **Reconstruction:** Identify $Z$ candidates from dilepton/jet pairs with mass $m_Z \approx 91$ GeV
- **Higgs tagging:** Use Higgs recoil mass as primary observable ($m_{\text{recoil}} \approx 125$ GeV)
- **Background suppression:** Apply XGBoost multivariate selection to increase signal purity
- **Systematic treatment:** Include background normalization uncertainties in statistical framework
- **Statistical extraction:** Use maximum likelihood fitting to extract signal strength ($\mu$) and its precision
- **Higgs self-coupling extraction** Parametrize the cross-section by using SMEFT models to extract the Higgs self-coupling and its precision

## Troubleshooting

### Common Issues

**Import errors for `xsec` module:**
- Make sure you're running scripts from the `xsec/` directory, not from subdirectories
- Verify `xsec/__init__.py` exists

**Script not found errors:**
- Double-check the path: use `0-Run/1-run.py` (not `run/1-run.py`)
- Ensure you're in the `xsec/` directory

**Argument parsing errors:**
- Use dash-separated values for multiple options: `--cat ee-mumu` and `--ecm 240-365`
- Not comma-separated: `--cat ee,mumu` will fail
- Check the script's help: `python 0-Run/1-run.py --help`

**Missing output files:**
- Verify each stage ran successfully before moving to the next stage
- Check that `output/` directory was created
- Look at log files for error messages

**Environment incompatibility (Stage 5):**
- If running stage 5 fails due to environment conflicts, try running it in a separate terminal with the CombinedLimit environment


## Authors & References

This analysis is part of the FCC physics program for precision Higgs measurements. 
It was written by Tom Fournier with help from Ang Li for stages 1-MVAInput and 2-BDT, and from Jan Eysermans for the remaining stages. There was also a contribution from Amaury Lhoste for the improvement of the BDT training.

---

**Last updated:** October 2026
