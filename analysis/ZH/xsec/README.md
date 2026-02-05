# FCC-ee ZH Cross-Section Analysis

This repository contains the analysis code for measuring the $e^+e^- \to ZH$ cross-section at the Future Circular Collider (FCC-ee) at two center-of-mass energies: 240 GeV and 365 GeV.

## Overview

The analysis workflow follows a sequential pipeline consisting of five stages:

1. **[1-MVAInputs](1-MVAInputs/README.md)** — Event selection and feature preparation for machine learning
2. **[2-BDT](2-BDT/README.md)** — Boosted Decision Tree training for signal/background discrimination  
3. **[3-Measurement](3-Measurement/README.md)** — Physics measurement with BDT-based event selection
4. **[4-Combine](4-Combine/README.md)** — Statistical datacard preparation for fitting
5. **[5-Fit](5-Fit/README.md)** — Maximum likelihood fits and bias testing

## Project Structure

### Analysis Stages

Each stage is contained in its own directory with a dedicated README:

- **`1-MVAInputs/`** — Selects events from raw simulation and prepares kinematic variables for BDT training. Produces histograms of distributions for all processes (signal and backgrounds).

- **`2-BDT/`** — Trains XGBoost models to distinguish ZH signal from backgrounds using kinematic features. Evaluates model performance and determines optimal selection thresholds.

- **`3-Measurement/`** — Applies BDT selection to measurement samples and generates histograms of the Higgs recoil mass across signal-like and background-like regions.

- **`4-Combine/`** — Processes histograms and generates COMBINE-compatible datacards with systematic uncertainties for statistical fitting.

- **`5-Fit/`** — Performs maximum likelihood fits to extract the ZH cross-section. Validates fitting procedures through bias tests using pseudo-experiments.

### Core Components

- **`package/`** — Central Python module providing configuration, utilities, and analysis functions used across all stages. See [package/README.md](package/README.md) for details.

- **`run/`** — Pipeline automation scripts that orchestrate the analysis across multiple channels and energies. See [run/README.md](run/README.md) for usage instructions.

- **`output/`** — Auto-generated directory containing all analysis outputs (see [Output Structure](#output-structure) below)

## Getting Started

### Prerequisites

- FCC software stack with FCCAnalysis framework
- Python 3.7+ with: ROOT, pandas, scikit-learn, xgboost, matplotlib
- COMBINE tool (for statistical fitting in stage 5)

### Software Frameworks

**FCCAnalyses** (Stages 1-4)  
FCCAnalyses is the main analysis framework used for stages 1-4. It provides tools for event selection, reconstruction, and histogram generation with integration to FCC simulation and detector simulation. This framework is required for running MVA input preparation, BDT training, and physics measurement stages.

**COMBINE-Limit** (Stage 5)  
COMBINE-Limit is the statistical analysis tool used for maximum likelihood fitting and limit/significance calculations. It is incompatible with the FCCAnalyses environment due to conflicting dependencies and uses an older version of Python libraries compared to FCCAnalyses.

⚠️ **Important Version Incompatibility Note:**  
The version of COMBINE-Limit used in this analysis is **older** than the version of Python libraries in FCCAnalyses. When writing code intended to work with both frameworks, exercise caution regarding:
- Python version features (avoid using features only available in Python 3.9+)
- Package dependencies (versions must be compatible with both environments)

Code that runs successfully with FCCAnalyses may fail in the COMBINE-Limit environment and vice versa.

### Running the Analysis

Execute stages sequentially from the **xsec/** directory:

```bash
cd xsec/

# Stage 1: Event selection and MVA input preparation
python run/1-run.py --cat ee-mumu --ecm 240-365

# Stage 2: BDT training and evaluation
python run/2-run.py --cat ee-mumu --ecm 240-365

# Stage 3: Physics measurement with BDT selection
python run/3-run.py --cat ee-mumu --ecm 240-365

# Stage 4: Histogram processing and datacard generation
python run/4-run.py --cat ee-mumu --ecm 240-365
```

**Important:** Stages 1-4 must be executed sequentially, as each stage depends on outputs from previous stages. See [run/README.md](run/README.md) for detailed parameter documentation and advanced usage.

#### ⚠️ Stage 5 — Execute in a Separate Terminal

Stage 5 (Statistical Fits and Bias Tests) requires the COMBINE-Limit environment, which is **incompatible** with FCCAnalyses. Execute stage 5 scripts in a **separate terminal** with a different software environment:

```bash
# In a NEW terminal, setup COMBINE-Limit environment (not FCCAnalyses)
cd ../../../
source setup_CombinedLimit.sh

# Then run stage 5:
cd analysis/ZH/xsec/
python run/5-run.py --cat ee-mumu --ecm 240-365
```

Do **NOT** try to run stage 5 in the same terminal where FCCAnalyses is loaded, as the conflicting dependencies will cause failures.

### Partial Execution

Run specific stages or channels as needed:

```bash
# Run only stage 2 (BDT training) for electron channel at 240 GeV
python run/2-run.py --cat ee --ecm 240

# Run stages 3 and 4 for both channels at 365 GeV
python run/3-run.py --cat ee-mumu --ecm 365 --run 1-2-3-4
python run/4-run.py --cat ee-mumu --ecm 365
```

Refer to individual stage READMEs for more detailed information and manual script execution.

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
│   │   │   ├── *_{sel}.root         # TTree files for the BDT training
│   │   │   └── *_{sel}_histo.root   # Kinematic distributions for all processes
│   │   ├── preprocessed/{ecm}/{cat}/{sel}/
│   │   │   └── *_histo.root         # Measurement histograms after BDT selection
│   │   └── processed/{ecm}/{cat}/{sel}/
│   │       └── {sample}.root        # Final histograms before making the datacards
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
│   └── combine/                     # COMBINE framework outputs
│       └── {sel}/{ecm}/{cat}/
│           ├── nominal/             # Nominal fit results
│           │   ├── WS/              # RooFit workspaces
│           │   ├── datacard/        # COMBINE datacards
│           │   ├── log/             # Fit logs
│           │   └── results/         # Extracted signal strengths
│           └── bias/                # Bias test results
│               ├── WS/              # Pseudo-data workspaces
│               ├── datacard/        # Pseudo-data datacards
│               ├── log/             # Fit logs
│               └── results/         # Bias analysis and plots
│
└── plots/                           # Analysis plots for validation
    ├── MVAInputs/{ecm}/{cat}/       # Kinematic distributions
    ├── evaluation/{ecm}/{cat}/      # BDT performance plots
    └── measurement/{ecm}/{cat}/     # Physics measurement plots
```

**Key outputs by analysis stage:**

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **1-MVAInputs** | Raw simulation (EDM4Hep) | Event trees, kinematic histograms | Prepare data for ML training |
| **2-BDT** | Kinematic histograms | Trained models, feature importance | Train signal/background discriminator |
| **3-Measurement** | Raw simulation + BDT models | Recoil mass histograms (signal/background regions) | Measure physics distributions |
| **4-Combine** | Measurement histograms | COMBINE datacards with uncertainties | Prepare input for statistical fitting |
| **5-Fit** | COMBINE datacards | Signal strengths, bias estimates, plots | Extract cross-section and validate fits |

The `output/` directory is listed in `.gitignore` and will not be tracked by git. Regenerate outputs by re-running the analysis stages.

## Development & Testing

### Creating Tests

For testing new functionality or debugging analysis code, create a `test/` directory with test scripts and data:

```bash
mkdir -p test/
# Add test scripts and minimal test data
```

The `test/` directory is configured in `.gitignore` and will not be tracked by git, making it ideal for development and troubleshooting without cluttering the repository.

### Analysis Code Organization

- **`package/config.py`** — Physics constants, decay modes, process definitions
- **`package/tools/`** — Data I/O, histogram utilities, significance calculations
- **`package/func/`** — Analysis functions (BDT operations, bias calculations)
- **`package/plots/`** — Visualization utilities

See [package/README.md](package/README.md) for full documentation.

## Documentation

Detailed information for each analysis stage:

- **[1-MVAInputs/README.md](1-MVAInputs/README.md)** — Event selection, variable computation, histogram generation
- **[2-BDT/README.md](2-BDT/README.md)** — BDT training procedure, hyperparameters, evaluation metrics
- **[3-Measurement/README.md](3-Measurement/README.md)** — Physics selection cuts, BDT application, control regions
- **[4-Combine/README.md](4-Combine/README.md)** — Datacard structure, systematic uncertainties, process definitions
- **[5-Fit/README.md](5-Fit/README.md)** — Fitting methodology, bias test procedures, result extraction
- **[run/README.md](run/README.md)** — Pipeline automation, parameter documentation, execution examples
- **[package/README.md](package/README.md)** — Configuration modules, utilities, function reference

## Physics Process

This analysis measures the cross-section for Higgs-strahlung production at $e^+e^-$ colliders:

$$e^+e^- \to Z(\to \ell^+\ell^-) H(\to \text{all})$$

where $\ell \in \{e, \mu\}$. The measurement uses:

- **Reconstruction:** Identify $Z$ candidates from dilepton pairs with mass $m_Z \approx 91$ GeV
- **Higgs tagging:** Use Higgs recoil mass as primary observable ($m_{\text{recoil}} \approx 125$ GeV)
- **Background suppression:** Apply XGBoost multivariate selection to increase signal purity
- **Systematic treatment:** Include background normalization uncertainties in statistical framework
- **Statistical extraction:** Use maximum likelihood fitting to extract signal strength ($\mu$)

## Authors & References

This analysis is part of the FCC physics program for precision Higgs measurements. 
It was written by Tom Fournier with the help from Ang Li for 1-MVAInput and 2-BDT and from Jan Eysermans for the rest of the analysis.
There also was a contribution from Amaury Lhoste in the early development of the repository.

---

**Last updated:** February 2026
