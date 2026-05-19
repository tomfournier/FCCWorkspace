# FCC-ee ZH Cross-Section Analysis

This repository contains the analysis code for measuring the $e^+e^- \to ZH$ cross-section at the Future Circular Collider (FCC-ee) at two center-of-mass energies: 240 GeV and 365 GeV.

## Overview

The analysis workflow follows a sequential pipeline consisting of five main stages, plus optional study pipelines:

**Main Analysis Pipeline:**
1. **[1-MVAInputs](1-MVAInputs/README.md)** — Event selection and kinematic variable preparation for BDT training
2. **[2-BDT](2-BDT/README.md)** — XGBoost classifier training for signal/background discrimination  
3. **[3-Measurement](3-Measurement/README.md)** — Physics measurement with BDT-based event classification
4. **[4-Combine](4-Combine/README.md)** — Statistical datacard preparation for RooFit/Combine fitting
5. **[5-Fit](5-Fit/README.md)** — Maximum likelihood fits to extract cross-section and bias testing

**Study Pipelines:**
- **[a-FSR](a-FSR/README.md)** — Final State Radiation optimization studies
- **[b-Optimization](b-Optimization/README.md)** — BDT hyperparameter optimization

## Project Structure

### Analysis Stages

Each stage is contained in its own directory with a dedicated README:

- **`0-Run/`** — Wrapper orchestration scripts (`1-run.py` through `5-run.py`, plus `a-run.py` and `b-run.py` for study pipelines). These automate batch execution across channels and energies. Execute these scripts from the xsec/ directory.

- **`1-MVAInputs/`** — Selects events from raw simulation and prepares kinematic variables for BDT training. Produces histograms of distributions for all processes (signal and backgrounds).

- **`2-BDT/`** — Trains XGBoost models to distinguish ZH signal from backgrounds using 9 kinematic features. Evaluates model performance and determines optimal selection thresholds.

- **`3-Measurement/`** — Applies BDT classification to measurement samples and generates histograms of the Higgs recoil mass across signal-like and background-like regions.

- **`4-Combine/`** — Processes histograms and generates RooFit/Combine-compatible datacards with systematic uncertainties for statistical fitting.

- **`5-Fit/`** — Performs maximum likelihood fits to extract the ZH cross-section. Validates fitting procedures through bias tests using pseudo-experiments.

- **`a-FSR/`** — Final State Radiation optimization studies (optional advanced analysis)

- **`b-Optimization/`** — BDT hyperparameter optimization studies (optional advanced analysis)

### Core Components

- **`package/`** — Central Python module providing configuration, utilities, and analysis functions used across all stages. Key components:
  - `config.py` — Physics constants, process definitions, and color palettes
  - `userConfig.py` — Path templates and global parameters (luminosity, energies)
  - `parsing.py` — Unified command-line argument parsing
  - `logger.py` — Logging configuration
  - `plots/` — Visualization utilities (cutflow, BDT evaluation, histogram plots)
  - `func/` — Analysis functions (BDT training/evaluation, bias testing)
  - `tools/` — Data processing (ROOT I/O, histogram manipulation)
  
  See [package/README.md](package/README.md) for detailed documentation.

- **`sel/`** — Event selection functions applied throughout the pipeline:
  - `sel/presel/` — Pre-selection functions (lepton/quark selection, kinematic cuts)
  - `sel/final/` — Final selection functions (histogram definitions, variable computation)
  
  See [sel/presel/README.md](sel/presel/README.md) for selection documentation.

- **`output/`** — Auto-generated directory containing all analysis outputs (see [Output Structure](#output-structure) below)

## Getting Started

### Prerequisites

- FCC software stack with FCCAnalysis framework (for stages 1-4)
- Python 3.7+ with: ROOT, pandas, scikit-learn, xgboost, matplotlib, numpy
- RooFit/Combine tools (for stage 5 statistical fitting, optional)

### Quick Start

For a quick test to verify the analysis is working:

```bash
cd xsec/

# Test stage 1 (pre-selection only, for a single channel)
python 0-Run/1-run.py --cat ee --ecm 240 --run 1

# Test stage 2 (BDT training)
python 0-Run/2-run.py --cat ee --ecm 240

# Test stage 3 (measurement)
python 0-Run/3-run.py --cat ee --ecm 240 --run 1-2
```

This will create sample outputs in the `output/` directory. For the full analysis, run all stages across both channels and energies as shown in the next section.

### Software Frameworks

**FCCAnalyses** (Stages 1-4)  
FCCAnalyses is the main analysis framework used for stages 1-4. It provides tools for event selection, reconstruction, and histogram generation with integration to FCC simulation and detector simulation. This framework is required for running MVA input preparation, BDT training, and physics measurement stages.

**RooFit/Combine-Limit** (Stage 5)  
RooFit and Combine are the statistical analysis tools used for maximum likelihood fitting and cross-section extraction. They are typically incompatible with the FCCAnalyses environment due to conflicting dependencies.

⚠️ **Environment Management Note:**  
If you encounter version incompatibility issues between FCCAnalyses and RooFit/Combine, consider:
- Running stages 1-4 in one environment (FCCAnalyses)
- Running stage 5 in a separate terminal with a different Python environment (RooFit/Combine)
- Using environment management tools (conda/venv) to isolate dependencies

### Running the Analysis

All scripts must be executed from the **xsec/** directory. The analysis is orchestrated through wrapper scripts in the `0-Run/` directory that automate batch execution across channels and energies:

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

**Key outputs by analysis stage:**

| Stage | Input | Main Output | Purpose |
|-------|-------|---|---------|
| **1-MVAInputs** | Raw simulation (EDM4Hep) | Event trees, kinematic histograms | Prepare data for BDT training |
| **2-BDT** | MVA input histograms | Trained XGBoost model, feature importance | Train signal/background classifier |
| **3-Measurement** | Raw simulation + trained BDT model | Recoil mass histograms (signal/background regions) | Measure physics distributions |
| **4-Combine** | Measurement histograms | RooFit/Combine datacards with uncertainties | Prepare input for statistical fitting |
| **5-Fit** | Combine datacards | Signal strengths, cross-section, bias estimates | Extract cross-section and validate fits |

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

- **`package/config.py`** — Physics constants (masses, decay modes), process definitions, color palettes, kinematic variable names
- **`package/userConfig.py`** — Path templates and global parameters (luminosity, channel names, data fractions)
- **`package/parsing.py`** — Unified command-line argument parsing used by all scripts
- **`package/logger.py`** — Logging setup and configuration
- **`sel/presel/`** — Pre-selection functions (lepton kinematics, event filters)
- **`sel/final/`** — Final selection and histogram definitions (binning, variable mapping)
- **`package/func/bdt.py`** — BDT training, evaluation, and model I/O
- **`package/plots/`** — Visualization utilities (cutflow plots, evaluation metrics, histogram plots)
- **`package/tools/`** — Data processing utilities (ROOT I/O, histogram manipulation, significance calculations)

See [package/README.md](package/README.md) and [sel/presel/README.md](sel/presel/README.md) for detailed documentation.

## Documentation

Detailed information for each analysis stage and component:

- **[1-MVAInputs/README.md](1-MVAInputs/README.md)** — Event selection, variable computation, histogram generation
- **[2-BDT/README.md](2-BDT/README.md)** — BDT training procedure, hyperparameters, evaluation metrics
- **[3-Measurement/README.md](3-Measurement/README.md)** — Physics selection cuts, BDT application, control regions
- **[4-Combine/README.md](4-Combine/README.md)** — Datacard structure, systematic uncertainties, process definitions
- **[5-Fit/README.md](5-Fit/README.md)** — Fitting methodology, bias test procedures, result extraction
- **[a-FSR/README.md](a-FSR/README.md)** — Final State Radiation optimization studies
- **[b-Optimization/README.md](b-Optimization/README.md)** — BDT hyperparameter optimization
- **[package/README.md](package/README.md)** — Configuration modules, utilities, function reference
- **[sel/presel/README.md](sel/presel/README.md)** — Event selection functions and physics cuts

## Physics Process

This analysis measures the cross-section for Higgs-strahlung production at $e^+e^-$ colliders:

$$e^+e^- \to Z(\to \ell^+\ell^-) H(\to \text{all})$$

where $\ell \in \{e, \mu\}$. The measurement uses:

- **Reconstruction:** Identify $Z$ candidates from dilepton pairs with mass $m_Z \approx 91$ GeV
- **Higgs tagging:** Use Higgs recoil mass as primary observable ($m_{\text{recoil}} \approx 125$ GeV)
- **Background suppression:** Apply XGBoost multivariate selection to increase signal purity
- **Systematic treatment:** Include background normalization uncertainties in statistical framework
- **Statistical extraction:** Use maximum likelihood fitting to extract signal strength ($\mu$)

## Troubleshooting

### Common Issues

**Import errors for `package` module:**
- Make sure you're running scripts from the `xsec/` directory, not from subdirectories
- Verify `package/__init__.py` exists

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
- If running stage 5 fails due to environment conflicts, try running it in a separate terminal with a different Python environment
- Alternatively, see if RooFit is available in your current FCCAnalyses environment

## Authors & References

This analysis is part of the FCC physics program for precision Higgs measurements. 
It was written by Tom Fournier with help from Ang Li for stages 1-MVAInput and 2-BDT, and from Jan Eysermans for the remaining stages. There was also a contribution from Amaury Lhoste in the early development of the repository.

---

**Last updated:** May 2026
