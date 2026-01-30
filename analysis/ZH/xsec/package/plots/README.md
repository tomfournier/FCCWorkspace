# Plots Module

Comprehensive plotting and visualization utilities for particle physics analysis, with support for both ROOT and matplotlib backends. This module provides production-ready functions for creating publication-quality plots with FCC-ee styling conventions.

## Overview

The `plots` module handles all visualization tasks for cutflow analysis, BDT evaluation, and physics distributions. It's organized into three main components:

| Module | Purpose |
|--------|---------|
| **cutflow.py** | Event selection efficiency and cumulative event counts across selection cuts |
| **eval.py** | BDT performance diagnostics (ROC curves, training metrics, feature importance) |
| **plotting.py** | General-purpose histogram plotting and decay-mode comparisons |

## Features

### Cutflow Analysis
- Event count aggregation across sequential selection cuts
- Stacked signal/background histograms with significance computation
- Per-decay-mode efficiency curves with uncertainty quantification
- ASCII table export for publication

**Key functions:**
- `get_cutflow()` — Main entry point for cutflow computation and plotting
- `CutFlow()` — Render stacked cutflow histograms
- `CutFlowDecays()` — Plot efficiency curves normalized to first cut for each Higgs decay mode
- `Efficiency()` — Generate efficiency summary with uncertainty bands

### BDT Evaluation
- Training diagnostics: log loss, classification error, AUC curves
- ROC curves with threshold optimization
- BDT score distributions (train/validation splits)
- Feature importance visualization
- Significance optimization by scanning BDT score thresholds
- Decision tree visualization with Graphviz

**Key functions:**
- `log_loss()`, `classification_error()`, `AUC()` — Training metrics
- `roc()` — ROC curve and AUC computation
- `bdt_score()`, `mva_score()` — Score distributions
- `importance()` — Feature importance ranking
- `hist_check()` — Validate input variables with per-decay-mode histograms (train/validation split comparison)
- `tree_plot()` — Decision tree structure visualization

### General Plotting
- Signal/background histogram overlays with optional stacking
- Higgs decay mode comparisons with normalized distributions
- Ratio plots for pseudo-signal analysis
- Significance plots for cut optimization
- Yield summary canvases with process metadata

**Key functions:**
- `makePlot()` — Draw signal/background histograms
- `PlotDecays()` — Compare Higgs decay modes
- `PseudoRatio()` — Create ratio plots comparing nominal and pseudo-signal distributions
- `significance()` — Running significance and signal efficiency
- `Bias()` — Bias distributions with uncertainty bands

## Visualization Backend

### matplotlib (Python subdirectory)
Located in `python/`, provides:
- `set_plt_style()` — Apply consistent styling (serif fonts, figure dimensions)
- `set_labels()` — Add axis labels and FCC-ee branding watermark
- `savefigs()` — Multi-format figure export (PNG, PDF, etc.)

**Default styling:**
- Figure size: 12×8 inches (1440×960 px)
- Font: serif (Roman), 30pt text, 25pt axes
- Grid enabled by default for easier value reading

### ROOT (Root subdirectory)
Located in `root/`, provides:
- `canvas()`, `canvasRatio()` — Create standard and dual-pad canvases
- `dummy()`, `dummyRatio()` — Template histograms with configured axes
- `finalize_canvas()`, `save_canvas()` — Polish and export ROOT plots
- Support for logarithmic scaling and ratio plot layouts

## Configuration & Conventions

- **Color scheme & labels:** Derived from `config.py` (process colors, process names, variable labels)
- **Significance formula:** S/√(S+B) where S=signal counts, B=total background
- **Uncertainties:** Poisson (√N) on raw event counts, scaled linearly with weights
- **Luminosity/cross-section scaling:** Applied when metadata available
- **Batch mode:** ROOT operations execute in batch mode for efficiency
- **LaTeX rendering:** Enabled for mathematical symbols and publication formatting

## Usage Examples

### Create a Cutflow Plot
```python
from package.plots import cutflow

cutflow.get_cutflow(
    data_frame=df,
    cuts={
        "baseline": {"cut0": "", "cut1": "pt > 20", "cut2": "eta < 2.5"},
        "tight": {"cut0": "", "cut1": "pt > 30", "cut2": "eta < 2.0"}
    },
    output_dir="output/plots/measurement",
    lumi=5.0  # ab^-1
)
```

### Evaluate BDT with Training Plots
```python
from package.plots import eval

eval.roc(X_train, y_train, X_test, y_test, model,
         output_dir="output/plots/evaluation")

eval.importance(model, feature_names, 
                output_dir="output/plots/evaluation")
```

### Plot Signal and Background
```python
from package.plots import plotting

plotting.makePlot(
    signal_hist=signal_data,
    background_hists={"Z": z_data, "ttbar": ttbar_data},
    output_dir="output/plots/distributions",
    logy=True
)
```

## Dependencies

- **numpy, pandas** — Data manipulation (lazy-loaded)
- **matplotlib** — Python plotting backend
- **ROOT** — ROOT data analysis framework (lazy-loaded)
- **xgboost** — BDT model (for evaluation plots)
- **tqdm** — Progress bars

## Directory Structure

```
plots/
├── __init__.py             # Module initialization
├── cutflow.py              # Cutflow & efficiency analysis
├── eval.py                 # BDT performance evaluation
├── plotting.py             # General histogram plotting
├── python/                 # Matplotlib utilities
│   ├── plotter.py          # Styling and export
│   └── helper.py           # Matplotlib helpers
└── root/                   # ROOT utilities
    ├── plotter.py          # Canvas and histogram creation
    └── helper.py           # ROOT formatting helpers
```

## Output

Generated plots are organized hierarchically by:
- **Selection** (baseline, tight, custom, etc.)
- **Category** (ee/mumu channels)
- **Format** (PNG, PDF)

**Category hierarchy:**
- `ee/` — Electron-positron channel
- `mumu/` — Muon-antimuon channel

Typical output structure:
```
output/plots/
├── measurement/
│   ├── 240/  (√s = 240 GeV)
│   │   ├── ee/
│   │   └── mumu/
│   └── 365/  (√s = 365 GeV)
│       ├── ee/
│       └── mumu/
├── evaluation/
│   ├── 240/
│   └── 365/
└── MVAInputs/
    ├── 240/
    └── 365/
```
