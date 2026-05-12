# 3-Measurement: Physics Analysis & Event Selection

Event selection and physics measurement stage for the FCC-ee ZH cross-section analysis. Applies kinematic cuts and BDT-based signal/background discrimination to generate measurement histograms.

## Overview

This module performs three main tasks:

1. **Pre-selection** (`pre-selection.py`): Initial kinematic filtering of events from raw simulation
2. **Final selection** (`final-selection.py`): BDT-based signal selection and histogram generation  
3. **Visualization** (`plots.py`): Physics validation plots and significance scans

## Scripts

### Main Executable Scripts

#### `pre-selection.py` — Event Pre-Selection

Applies initial kinematic cuts to raw simulated events to reduce background and prepare data for final selection.

**Key operations:**
- Selects dilepton events from $e^+e^- \to Z(\ell^+\ell^-)X$ processes
- Applies lepton identification and kinematic cuts
- Filters events to high-purity $Z \to \ell^+\ell^-$ candidates
- Outputs event trees for BDT training and final selection

**Input:** Raw EDM4Hep events from FCC centralized productions
```
FCCee/winter2023/IDEA/ (via procDict)
```

**Output:** Pre-selected event samples
```
output/data/events/{ecm}/{cat}/full/analysis/
```

**Usage:**

Execute from the **xsec/** folder:
```bash
cd xsec/
fccanalysis run 3-Measurement/pre-selection.py
```

Or automate through the pipeline runner (see [Workflow Integration](#workflow-integration)):
```bash
python run/1-run.py --run 1
```

**Configuration:**
- Integration with FCCAnalysis framework
- Custom C++ analysis functions for advanced cuts
- Configurable CPU parallelization and batch submission
- Large samples split into chunks for efficient processing

---

#### `final-selection.py` — BDT Selection & Histogram Production

Applies BDT-based signal/background discrimination and generates analysis histograms from pre-selected events.

**Key operations:**
- Loads trained BDT models for each final state/energy combination
- Evaluates BDT score on kinematic variables: lepton kinematics, dilepton mass/momentum, event topology
- Applies BDT cut determined from significance scan in 2-BDT evaluation stage (see [2-BDT/README.md](../2-BDT/README.md) for details)
- Defines baseline selection cuts:
  - Dilepton mass window: $86 < m_{\ell^+\ell^-} < 96\text{ GeV}$
  - Momentum cuts (energy-dependent): 20-70 GeV (240 GeV), 50-150 GeV (365 GeV)
  - Recoil mass window (365 GeV only): $100 < m_{\text{recoil}} < 150\text{ GeV}$
  - Visible energy threshold (energy-dependent)
- Generates selection variants:
  - Base selections: `Baseline`, `Baseline_miss`, `Baseline_sep`
  - High BDT score region: `{sel}_high` (signal-like events)
  - Low BDT score region: `{sel}_low` (background-like events)
- Processes separate signal samples for each Z and Higgs decay combination with sample naming: `wzp6_ee_{Z_decay}H_H{Higgs_decay}_ecm{ecm}`
  - Allows reconstruction of selection effects and acceptance for individual decay channels
  - Supports physics analysis of decay-dependent properties
- Produces histograms of Higgs recoil mass ($m_{\text{recoil}}$) across all three regions for each decay combination
- Scales yields to integrated luminosity ($10.8\text{ ab}^{-1}$ at 240 GeV; $3.12\text{ ab}^{-1}$ at 365 GeV)

**Input:** Pre-selected event trees from `pre-selection.py`
```
output/data/events/{ecm}/{cat}/full/analysis/
```

**Output:** Preprocessed histograms
```
output/data/histograms/preprocessed/{ecm}/{cat}/{sample}_{sel}_histo.root
```

**Usage:**

Execute from the **xsec/** folder:
```bash
cd xsec/
fccanalysis final 3-Measurement/final-selection.py
```

Or automate through the pipeline runner (see [Workflow Integration](#workflow-integration)):
```bash
python run/1-run.py --run 2
```

**Configuration:**
- Parallel processing with configurable CPU count
- Event selection fractions for testing (configurable in `package.userConfig`)

**Key functions** (from `package`):
- BDT loading and evaluation (see [package/func/bdt.py](../package/func/README.md))
- Histogram caching and ROOT I/O (see [package/tools/](../package/tools/README.md))

---

#### `plots.py` — Physics Validation & Significance Plots

Generates visualization plots for signal/background distributions and statistical significance estimates.

**Key operations:**
- Plots kinematic distributions for signal and backgrounds
- Generates Higgs decay mode composition plots
- Computes and visualizes signal significance

**Input:** Preprocessed histograms from `final-selection.py`
```
output/data/histograms/preprocessed/{ecm}/{cat}/{sample}_{sel}_histo.root/
```

**Output:** Validation plots
```
output/plots/measurement/{ecm}/{cat}/
```

**Usage:**
```bash
python plots.py [--cat CHANNEL] [--ecm ENERGY] [--yields] [--decay] [--make] [--scan]
```

**Arguments:**
- `--cat`: Final state (`ee`, `mumu`, or `ee-mumu`, default: `ee-mumu`)
- `--ecm`: Center-of-mass energy (`240` or `365`, default: `240`)
- `--yields`: Skip yield plots
- `--decay`: Skip Higgs decay mode plots
- `--make`: Skip distribution plots
- `--scan`: Generate significance scan plots (default: off)

**Example:**
```bash
python plots.py --cat ee --ecm 240 --scan
```

---

### Workflow Integration

The complete measurement stage can be automated using the pipeline runner:

```bash
cd run/
python 3-run.py --cat ee-mumu --ecm 240-365 --run 1-2-3-4
```

**Stages:**
- Stage 1: Pre-selection
- Stage 2: Final selection (default)
- Stage 3: Plots (default)
- Stage 4: Cutflow summary

For more details on pipeline execution, see [run/README.md](../run/README.md).

---

## Output Structure

### Data Output

```
output/
├── data/
│   ├── events/{ecm}/{cat}/full/analysis/           # Pre-selection event trees
│   └── histograms/preprocessed/{ecm}/{cat}/        # Final selection histograms
│       ├── {sample}_{sel}_hist.root
│       └── ... (all analysis samples)
└── plots/
    └── measurement/{ecm}/{cat}/                    # Validation plots
```

Each selection variant generates its own directory:
- `{sel}/` - Base selection (all selected events)
- `{sel}_high/` - High BDT score region (signal-like events)
- `{sel}_low/` - Low BDT score region (background-like events)

### Validation Plots Structure

```
measurement/{ecm}/{cat}/
├── higgsDecays/          # Higgs decay mode composition
│   └── {sel}/{region}/   # Plots for each selection and BDT region
├── makePlot/             # Kinematic distributions
│   └── {sel}/{region}/
├── significance/         # Signal significance estimates
│   └── {sel}/{region}/
└── yield/                # Selection yields and efficiencies
    └── {sel}/{region}/
        ├── cutflow/
        └── efficiency/
```

Where:
- `{sel}` = selection variant (`Baseline`, `Baseline_miss`, `Baseline_sep`, etc.)
- `{region}` = `nominal` (all events), `high` (signal-like), `low` (background-like)

---

## Configuration Reference

Configuration is managed through the `package/` module. For detailed information on:
- Kinematic variable definitions: see [package/config.py](../package/config.py#L1)
- Path templates and I/O: see [package/userConfig.py](../package/userConfig.py#L1)
- BDT model loading and evaluation: see [package/func/bdt.py](../package/func/README.md)
- Histogram caching and processing: see [package/tools/](../package/tools/README.md)

---

## Physics Notes

- **Signal process:** $e^+e^- \to Z(\ell^+\ell^-)H$ with $\ell \in \{e, \mu\}$ with $H$ decaying to quarks, leptons and gauge bosons exclusively
- **Background processes:** $ZZ$, $W^+W^-$, $Z/\gamma$, rare processes (see [4-Combine/README.md](../4-Combine/README.md) for complete process list)
- **Higgs reconstruction:** Recoil mass computed from $Z \to \ell^+\ell^-$ four-momentum
- **Signal discrimination:** BDT trained on 9 kinematic variables (see [package/config.py](../package/config.py#L1) for details)
- **Systematic considerations:** MC statistics and background modeling effects propagated through histogram uncertainties

