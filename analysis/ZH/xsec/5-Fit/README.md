# 5-Fit: Statistical Fitting & Bias Testing

Statistical analysis module for measuring the ZH cross-section using maximum likelihood fits and validating fit performance through bias tests.

## Overview

This module performs:
- **Nominal fits**: Extract signal strength and cross-section from processed histograms
- **Bias tests**: Validate fitting procedure by recovering known signal variations from pseudo-experiments
- **Results extraction**: Determine signal strength ($\mu$) and measurement uncertainty

## Scripts

### Main Executable Scripts

#### `fit.py` — Nominal Cross-Section Fit

Performs maximum likelihood fit on histograms to extract the ZH cross-section.

**Usage:**
```bash
python fit.py --cat ee --ecm 240 --sel Baseline
python fit.py --combine --ecm 365 --sel Baseline_sep
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)
- `--sel`: Selection strategy (e.g., `Baseline`, `Baseline_sep`)
- `--combine`: Combine ee and mumu channels for joint fit
- `--bias`: Run in bias test mode (internal use with bias_test.py)
- `--target`: Higgs decay mode for bias tests (e.g., `bb`, `cc`, `gg`, `mumu`, `WW`, `ZZ`, `inv`)
- `--pert`: Prior uncertainty on signal cross-section for bias tests (default: 1.05)
- `--noprint`: Suppress output of extraction values
- `--timer`: Display elapsed time

**Workflow:**
1. Combines datacards from individual channels (if `--combine` specified)
2. Converts datacard to RooFit workspace using `text2workspace.py`
3. Runs maximum likelihood fit with `combine` tool
4. Extracts signal strength ($\mu$) and uncertainty from fit results

**Output:** Signal strength, uncertainty, and logs in workspace directory

**Results Location:**
- Signal strength & uncertainty: `output/data/combine/{sel}/{ecm}/{cat}/nominal/results/results.txt`
- Fit workspace: `output/data/combine/{sel}/{ecm}/{cat}/nominal/WS/higgsCombineXsec.MultiDimFit.mH125.root`
- Workspace file: `output/data/combine/{sel}/{ecm}/{cat}/nominal/WS/ws.root`
- Logs: `output/data/combine/{sel}/{ecm}/{cat}/nominal/log/`

---

#### `bias_test.py` — Bias Test Suite

Validates fitting procedure by generating pseudo-data with known signal variations and checking recovery. Systematically tests each Higgs decay channel.

**Usage:**
```bash
python bias_test.py --cat ee --ecm 240 --pert 1.05
python bias_test.py --combine --ecm 365 --pert 1.05 --freeze
python bias_test.py --cat ee --ecm 240 --tot                        # Include all Z decays
python bias_test.py --cat mumu --ecm 365 --plot_dc --no-print      # Generate diagnostic plots
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)
- `--sel`: Selection strategy (default: `Baseline`)
- `--pert`: Prior uncertainty on signal cross-section (default: 1.05 = +5%)
- `--combine`: Combine channels for joint fit
- `--tot`: Include all Z decay channels in signal definition (default: single channel)
- `--freeze`: Freeze background yields (don't float in fit)
- `--float`: Float backgrounds in fit
- `--plot_dc`: Generate diagnostic plots of datacards
- `--polL` / `--polR`: Scale to left/right beam polarization
- `--ILC`: Scale results to ILC luminosity
- `--extra`: Extra fit options (pass as `--extra tot`, `--extra onlyrun`, etc.)
- `--no-print`: Suppress output of extraction values
- `--timer`: Display elapsed time

**Workflow:**
1. **Histogram caching** — Preloads histograms and cross-section metadata once to minimize repeated I/O across all decay modes
2. For each Higgs decay channel:
   - Generates pseudo-data with known signal variation (controlled by `--pert`) using cached histograms
   - Runs fit on pseudo-data to recover signal strength
   - Measures difference between fitted and injected signal (bias = 100 × (μ_fitted - pert))
   - Generates pseudo-ratio validation plots in high/low BDT score regions (single channels only)
3. Collects bias estimates across all decay modes
4. Generates bias summary plot and stores results in CSV and formatted text files

**Output:** Pseudo-data fits, bias estimates, validation plots, and structured results

**Results Location:**
- Bias fit results: `output/data/combine/{sel}/{ecm}/{cat}/bias/`
- Bias analysis summary:
  - **Bias results (text)**: `output/data/combine/{sel}/{ecm}/{cat}/bias/results/bias/bias_results.txt` — Numerical bias values for each decay mode
  - **Bias results (CSV)**: `output/data/combine/{sel}/{ecm}/{cat}/bias/results/bias/bias_results.csv` — Tabular bias data for analysis
  - **Bias plot**: `output/data/combine/{sel}/{ecm}/{cat}/bias/results/bias/bias.png` — Bias validation plot showing bias for each Higgs decay mode
- Pseudo-ratio validation plots:
  - **For individual channels** (`{cat}` = `ee` or `mumu`): 
    - `output/data/combine/{sel}/{ecm}/{cat}/bias/results/bias/high/PseudoRatio_{decay_mode}.png` — Ratio plots in high BDT score region
    - `output/data/combine/{sel}/{ecm}/{cat}/bias/results/bias/low/PseudoRatio_{decay_mode}.png` — Ratio plots in low BDT score region
  - **For combined channels** (`{cat}` = `combined`): Bias plot only (no high/low split)
- Individual fit results: `output/data/combine/{sel}/{ecm}/{cat}/bias/results/fit/results_{decay_mode}.txt` — Signal strength per decay mode

---

### Internal Utility Script

#### `make_pseudo.py` — Pseudo-Data Generation & Bias Fit Execution

Generates pseudo-data histograms and datacards for a specific Higgs decay channel, with optional automatic fit execution.

**Usage:**
```bash
python make_pseudo.py --cat ee --ecm 240 --target bb --pert 1.05
python make_pseudo.py --cat mumu --ecm 365 --target inv --pert 1.05 --run        # Generate and run fit
python make_pseudo.py --cat ee --ecm 240 --target WW --pert 1.05 --onlyrun      # Run fit only (skip datacard generation)
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`)
- `--ecm`: Center-of-mass energy (240 or 365 GeV)
- `--sel`: Selection strategy (default: `Baseline`)
- `--target`: Target Higgs decay mode (`bb`, `cc`, `gg`, `mumu`, `WW`, `ZZ`, `inv`, etc.)
- `--pert`: Prior uncertainty on signal cross-section (default: 1.05 = +5%)
- `--tot`: Include all Z decay channels (default: single channel)
- `--freeze`: Freeze background yields in fit
- `--float`: Float backgrounds in fit
- `--plot_dc`: Generate diagnostic plots of datacards
- `--run`: Generate pseudo-data and automatically execute fit
- `--onlyrun`: Skip pseudo-data generation and only run fit (assumes datacard exists)
- `--nobias`: Internal flag indicating script is not called from bias_test.py
- `--timer`: Display elapsed time

**Purpose:** Can be run standalone to generate and inspect pseudo-data for specific decay modes, or called automatically by `bias_test.py`. The `--run` and `--onlyrun` flags allow fit execution without separate script invocation.

---

### Batch Execution

#### `../0-Run/5-run.py` — Automated Script Execution

Executes `fit.py` and `bias_test.py` for multiple combinations of category (`cat`), energy (`ecm`), and selection strategy (`sel`). Avoids repeated manual script invocations.

**Usage:**
```bash
python 0-Run/5-run.py
```

**Purpose:** Streamlines batch processing of nominal fits and bias tests across all analysis configurations. Automatically manages parameter combinations defined in the script.

For detailed information about batch execution, see [0-Run/README.md](../0-Run/README.md).

---

## Workflow Integration

```
Processed Histograms (4-Combine)
    ↓
┌──────────────────────────┬──────────────────────────────┐
│                          │                              │
fit.py (nominal)    bias_test.py (automated loop)         │
│                          │                              │
│                    ┌─────┴──────┐                       │
│                    ↓            ↓                       │
│            make_pseudo.py    PseudoRatio plots          │
│                    ↓                                    │
│            fit.py (bias)                                │
│                    ↓                                    │
│            Bias, PseudoRatio plot                       │
│                                                         │
└────────────────────┬────────────────────────────────────┘
         ↓                       ↓
Nominal Signal Strength   Bias Validation
& Cross-Section           & Uncertainty
```

---

## Setup & Requirements

### Environment Setup

⚠️ **Important:** FCCAnalyses and CombinedLimit have conflicting dependencies and **must be run in separate terminals**.

**For fit.py and bias_test.py:**
```bash
cd ../../../  # Navigate to FCCWorkspace directory
source setup_CombinedLimit.sh
cd analysis/ZH/xsec/
python 5-Fit/fit.py --cat ee --ecm 240 --sel Baseline
```

The `setup_CombinedLimit.sh` script is located in the `FCCWorkspace` directory (parent of `analysis/`). Source it in a separate terminal before executing the scripts. Do not source this in terminals running earlier analysis stages (1-4).

### Configuration & Paths

Configuration is centralized in [package/](../package/README.md):
- **[userConfig.py](../package/userConfig.py)**: File paths, energy-dependent parameters (luminosity), channel definitions
- **[config.py](../package/config.py)**: Physics constants, process definitions, decay modes (Z and Higgs)
- **[func/bias.py](../package/func/bias.py)**: Pseudo-data generation utilities (`pseudo_datacard` function)
- **[plots/plotting.py](../package/plots/README.md)**: Bias and pseudo-ratio visualization (`Bias`, `PseudoRatio` functions)
- **[tools/process.py](../package/tools/README.md)**: Histogram caching and cross-section utilities

## Output Structure

Results are organized by selection strategy, energy, and category:
```
output/data/combine/
└── {sel}/                                          # Selection strategy (e.g., Baseline, Baseline_sep)
    └── {ecm}/                                      # Energy: 240 or 365 GeV
        └── {cat}/                                  # Category: ee, mumu, or combined
            ├── nominal/                            # Nominal fit results
            │   ├── WS/                             # RooFit workspace
            │   │   ├── higgsCombineXsec.MultiDimFit.mH125.root
            │   │   ├── ws.root                     # Workspace file
            │   │   └── ws.root.dot
            │   ├── datacard/                       # Input datacard
            │   │   └── datacard.txt (or datacard_combined.txt for combined)
            │   ├── log/                            # Conversion logs
            │   │   ├── log_text2workspace.txt
            │   │   └── log_results.txt
            │   └── results/
            │       └── results.txt                 # Signal strength + uncertainty
            │
            └── bias/                               # Bias test results (per decay mode)
                ├── WS/                             # Workspaces per decay
                │   ├── ws_{decay_mode}.root
                │   └── higgsCombineXsec.MultiDimFit.mH125.root (final combined)
                ├── datacard/                       # Datacards per decay
                │   ├── datacard_{decay_mode}.txt
                │   └── datacard_{decay_mode}_combined.txt (for combined category)
                ├── log/                            # Logs per decay
                │   ├── log_text2workspace_{decay_mode}.txt
                │   └── log_results_{decay_mode}.txt
                └── results/
                    ├── bias/                       # Bias analysis results
                    │   ├── bias.png                # Bias validation plot
                    │   ├── bias_results.txt        # Bias values per decay
                    │   ├── bias_results.csv        # Bias data (tabular)
                    │   ├── high/                   # PseudoRatio in high BDT score region (ee, mumu only)
                    │   │   └── PseudoRatio_{decay_mode}.png
                    │   └── low/                    # PseudoRatio in low BDT score region (ee, mumu only)
                    │       └── PseudoRatio_{decay_mode}.png
                    └── fit/                        # Individual fit results
                        └── results_{decay_mode}.txt
```

Where:
- `{sel}` = selection strategy (e.g., `Baseline`, `Baseline_sep`)
- `{ecm}` = center-of-mass energy (240 or 365 GeV)
- `{cat}` = category (`ee`, `mumu`, or `combined`)
- `{decay_mode}` = Higgs decay mode (bb, cc, gg, WW, ZZ, Za, aa, inv, mumu, ss, tautau)

## Requirements

- **Statistical Tools**: RooFit, CombinedLimit (`combine`, `combineCards.py`, `text2workspace.py`)
- **Dependencies**: numpy, pandas, ROOT
- **Setup**: `setup_CombinedLimit.sh` must be sourced before running scripts
- **Configuration**: See [package/README.md](../package/README.md)

## References

- Combine documentation: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
- Bias testing methodology: See [package/func/bias.py](../package/func/bias.py) for implementation details

