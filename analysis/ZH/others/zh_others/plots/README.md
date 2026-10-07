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
- Variable extraction from cut filter expressions

**Key functions:**
- `get_cutflow()` — Main entry point for cutflow computation and plotting
- `CutFlow()` — Render stacked cutflow histograms with signal overlay and significance values
- `CutFlowDecays()` — Plot efficiency curves (normalized to first cut) for each Higgs decay mode
- `Efficiency()` — Generate efficiency summary plots with uncertainty bands and per-decay tables
- `branches_from_cuts()` — Extract variable names referenced in cut filter expressions

### BDT Evaluation
- Training diagnostics: log loss, classification error, AUC curves
- ROC curves with threshold optimization and train/validation splits
- BDT score distributions with signal/background breakdown
- Feature importance visualization using XGBoost F-scores
- Significance optimization by scanning BDT score thresholds
- Decision tree visualization with Graphviz
- Input variable validation with histograms and statistical comparisons

**Key functions:**
- `log_loss()`, `classification_error()`, `AUC()` — Training metrics across boosting iterations
- `roc()`, `plot_roc_curve()` — ROC curves and AUC computation
- `bdt_score()`, `mva_score()` — Score distributions (train/validation breakdown)
- `importance()` — Feature importance ranking using split counts
- `significance()` — Scan BDT score thresholds for optimal significance (S/√(S+B))
- `efficiency()` — Signal/background efficiency vs. BDT cut threshold
- `hist_check()` — Validate input variables with per-decay-mode histograms
- `tree_plot()` — Visualize subset of boosted decision trees

### General Plotting
- Signal/background histogram overlays with optional stacking
- Higgs decay mode comparisons with unit-integral normalization
- Ratio plots for pseudo-signal analysis
- Significance plots for cut optimization
- Yield summary canvases with process metadata
- Bias distribution analysis with uncertainty bands
- Flexible argument processing with hierarchical lookup and wildcard matching

**Key functions:**
- `makePlot()` — Draw signal/background histograms with optional stacking and scaling
- `PlotDecays()` — Compare Higgs decay modes with unit-integral normalization
- `PseudoRatio()` — Create ratio plots comparing nominal and pseudo-signal distributions
- `significance()` — Plot running significance and signal efficiency for cut optimization
- `AAAyields()` — Render yields summary canvas with process yields and metadata
- `Bias()` — Plot bias distributions per Higgs decay mode with uncertainty bands
- `get_args()`, `args_decay()` — Extract plotting arguments with hierarchical lookup
- `_parse_selection_dir()` — Build standardized output directory paths
- `hist_to_arrays()` — Convert ROOT histograms to numpy arrays with errors

## Plotting Utilities

### Plotting Arguments (`plotting.py`)
Flexible argument extraction system with hierarchical lookup and wildcard matching:
- `get_args()` — Extract plotting config for variable/selection with hierarchical defaults
- `args_decay()` — Extract decay-specific plotting arguments (excludes 'make' mode)
- `_extract_nested_args()` — Navigate nested argument structures with wildcard pattern matching
- `_parse_selection_dir()` — Build standardized output directory paths
- `_ensure_plt_style()` — Initialize matplotlib styling once per session

**Argument Hierarchies:**
- Direct parameters: `args[var]`
- By energy: `args[var][ecm]`
- By energy and selection: `args[var][ecm][sel_pattern]`
- By selection only: `args[var][sel_pattern]`

**Pattern Matching Supported:**
- Exact match: `sel == pattern`
- Wildcard: `Baseline*` matches `Baseline`, `Baseline_high`, `Baseline_low`
- Pipe-separated: `Baseline_sep|Baseline_high` matches either pattern

### Visualization Backend

#### matplotlib (Python subdirectory)

**Styling and export** (`python/plotter.py`):
- `set_plt_style()` — Apply consistent styling (serif fonts, figure dimensions)
- `set_labels()` — Add axis labels and FCC-ee branding watermark
- `savefigs()` — Multi-format figure export (PNG, PDF, etc.)

**Default styling:**
- Figure size: 12×8 inches (1440×960 px)
- Font: serif (Roman), 30pt text, 25pt axes
- Grid enabled by default for easier value reading

**Event flow and data utilities** (`python/helper.py`):
- `find_sample_files()` — Locate ROOT files for samples within a directory
- `get_processed()` — Sum processed event counts from ROOT file metadata
- `get_cut()` — Sum event yields for a cut from pre-computed histograms
- `get_count()` — Get event count from histogram or dataframe with filtering
- `is_there_events()` — Check if ROOT files contain an 'events' tree
- `getcut()` — Apply filter expressions to dataframe with mask combination
- `get_flow()` — Build process-level event flow histograms across cut steps
- `get_flow_decay()` — Build Higgs-decay-level flow histograms
- `get_flows()` — Generate both process and decay-level flows together
- `dump_json()` — Serialize flow data to JSON for downstream analysis
- `branches_from_cuts()` — Extract variable names referenced in cut expressions using regex

**Histogram Access Pattern:**
- Efficient LRU caching (no size limit) to minimize file I/O
- Cut histograms stored at: `custom_objects/{HIST_TYPE}/cutFlow`
- Cut naming: "cut0", "cut1", ... where ROOT bin 1 = cut0, bin 2 = cut1

#### ROOT (Root subdirectory)

**Canvas and histogram creation** (`root/plotter.py`):
- `canvas()`, `canvasRatio()` — Create standard and dual-pad canvases
- `dummy()`, `dummyRatio()` — Template histograms with configured axes
- `aux()`, `auxRatio()` — Render auxiliary labels (luminosity, channel, energies)
- `setup_cutflow_hist()` — Configure canvas and histogram for cutflow visualization
- `finalize_canvas()` — Apply final cosmetics (grid, axis redraw, auxiliary labels)
- `save_canvas()` — Save canvas to file in multiple formats with proper formatting

**Histogram and axis configuration** (`root/helper.py`):
- **Configuration:** `make_cfg()`, `build_cfg()` — Complete plotting configuration with defaults and validation
- **Canvas layout:** `canvas_margins()`, `pad_margins()` — Configure canvas and pad margins
- **Legends:** `mk_legend()` — Create configured ROOT legend with automatic sizing based on entry count
- **Axis formatting:** `configure_axis()`, `axis_limits()` — Set axis title, range, fonts, and log-scale padding
- **Histogram styling:** `style_hist()`, `style_hists_batch()` — Apply colors, widths, styles, and scaling
- **Histogram loading:** `load_hists()`, `_get_hist_cached()` — Load histograms with LRU caching
- **Text annotation:** `setup_latex()`, `draw_latex()`, `y_offset()` — Create and position TLatex annotations

**ROOT Batch Configuration:**
- All operations execute in batch mode (no interactive displays)
- Stat and title boxes disabled by default
- Graphics system pre-warmed for faster first-plot rendering

## Configuration & Conventions

### Plotting Configuration
- **Color scheme & labels:** Derived from `config.py` (process colors, process names, variable labels)
- **Argument hierarchies:** Three-level nesting with wildcard pattern matching for flexible configuration
- **Automatic defaults:** Missing arguments auto-populated with sensible defaults (xmin, xmax, ymin, ymax, rebin, lumi, ecm)

### Data Analysis
- **Significance formula:** S/√(S+B) where S=signal counts, B=total background
- **Uncertainties:** Poisson (√N) on raw event counts, scaled linearly with weights
- **Efficiency normalization:** Computed relative to first cut (cut0) for decay mode comparisons
- **Luminosity/cross-section scaling:** Applied when metadata available

### Cut and Flow Data Structures
- **Cut naming:** Sequential (cut0, cut1, ...) matching provided labels
- **Event counts:** Cumulative (events passing up to and including that cut)
- **Flow dict format:** `events[sample][selection]["cut"][cut_name] = int (count)`
- **Errors dict format:** `events[sample][selection]["err"][cut_name] = float (uncertainty)`
- **Decay flows:** Pattern-matched from signal samples: `wzp6_ee_{Z_decay}H_H{Higgs_decay}_ecm{ECM}`

### ROOT Graphics
- **Batch mode:** All ROOT operations execute in batch mode for efficiency
- **Styling:** Stat and title boxes disabled by default for cleaner appearance
- **Margins:** Configurable per canvas type (standard, ratio, cutflow)
- **LaTeX rendering:** Enabled for mathematical symbols and publication formatting
- **Log-scale padding:** 0.999x–1.001x applied to prevent edge clipping in zoomed plots

### Matplotlib Configuration
- **Figure setup:** 12×8 inches at 120 DPI (1440×960 pixels)
- **Fonts:** Serif family (Roman) with 30pt base size, 25pt axes
- **Axes:** Grid enabled by default, 25pt title and label sizes
- **Legend:** 14pt font, 0.6 opacity by default
- **Text rendering:** LaTeX enabled for math symbols and special formatting

## Usage Examples

### Build Event Flow from ROOT Files

Event flow uses the Python helper utilities to aggregate yields from ROOT files:

```python
from package.plots.python.helper import get_flows

# Define processes and signal samples
processes = {
    'ZeeH': ['wzp6_ee_eeH_ecm240', 'wzp6_ee_eeH_Hmumu_ecm240', ...],
    'Zbb': ['wzp6_ee_bbZ_ecm240', ...],
}

z_decays = ['ee', 'mumu']
h_decays = ['bb', 'WW', 'tau', 'aa']

# Build both process-level and decay-level flows
flows = get_flows(
    procs=['ZeeH', 'Zbb'],
    processes=processes,
    cuts={'pre-selection': {'cut0': '', 'cut1': 'pt > 20', ...}},
    inDir='output/data/events',
    ecm=240,
    json_file=True,  # Export to JSON
    z_decays=z_decays,
    H_decays=h_decays
)

# flows is dict: flows['process_name']['cut'][cut_name] = count
# Export to JSON for downstream analysis
```

### Create a Cutflow Plot

Use the event flow data to render publication-quality cutflow plots:

```python
from package.plots.cutflow import get_cutflow

get_cutflow(
    inDir='output/data/events',
    outDir='output/plots/measurement',
    cat='ee',  # Electron-positron channel
    sels=['pre-selection', 'final-selection'],
    procs=['ZeeH', 'Zbb'],  # [signal, background1, ...]
    procs_decays=['bb', 'WW', 'tau'],
    processes=processes,
    colors=colors_dict,  # from config.py
    legend=legend_dict,  # from config.py
    cuts=cuts_dict,
    cuts_label=labels_dict,
    z_decays=z_decays,
    H_decays=h_decays,
    ecm=240,
    lumi=10.8,
    sig_scale=1.0,
    tot=True  # Include total plots
)
```

### Evaluate BDT with Training Diagnostics

Monitor BDT training convergence and optimize cut thresholds:

```python
from package.plots.eval import log_loss, significance, importance
import xgboost as xgb
import pandas as pd

# After training XGBoost model...
model = xgb.train(params, dtrain, evals=[(dval, 'validation')], evals_result=results)

# Plot training metrics
log_loss(results, range(len(results['training'])), 
         label='Signal vs Background', 
         outDir='output/plots/evaluation')

# Score predictions and find optimal cut
df_scored = pd.DataFrame({
    'BDTscore': model.predict(dtest),
    'isSignal': y_test,
    'weights': weights_test
})

significance(df_scored, label='240 GeV', outDir='output/plots/evaluation')

# Visualize feature importance
importance(model, feature_names, latex_mapping, label='BDT Features',
           outDir='output/plots/evaluation')
```

### Plot Signal and Background Distributions

Compare decay modes with flexible argument configuration:

```python
from package.plots.plotting import makePlot, PlotDecays, get_args

# Configure plotting arguments with hierarchical lookup
plot_args = get_args(
    var='m_ll',
    sel='final-selection',
    ecm=240,
    lumi=10.8,
    args=args_dict  # From your configuration
)

# Draw signal vs background
makePlot(
    variable='m_ll',
    inDir='output/histograms/MVAInputs',
    outDir='output/plots/MVAInputs',
    sel='final-selection',
    procs=['ZeeH', 'Zbb', 'ttbar'],  # [signal, backgrounds...]
    processes=processes,
    colors=colors_dict,
    legend=legend_dict,
    **plot_args  # Apply extracted configuration
)

# Compare Higgs decay modes
PlotDecays(
    variable='m_ll',
    inDir='output/histograms/MVAInputs',
    outDir='output/plots/MVAInputs',
    sel='final-selection',
    z_decays=['ee', 'mumu'],
    h_decays=['bb', 'WW', 'tau', 'aa'],
    ecm=240,
    lumi=10.8
)
```

### Validate Input Variables with Histograms

Check for signal/background separation and train/validation consistency:

```python
from package.plots.eval import hist_check

hist_check(
    df=training_data,  # DataFrame with 'isSignal', 'valid' columns
    label='Before selection',
    outDir='output/plots/evaluation',
    modes=['ee', 'mumu'],  # Decay channels
    modes_label={'ee': 'Electron pair', 'mumu': 'Muon pair'},
    modes_color={'ee': 'blue', 'mumu': 'red'},
    var='pt_ll',
    xlabel=r'$p_T^{\ell\ell}$ [GeV]',
    ncols=3,
    unity=False  # Don't normalize
)

## Design Patterns and Architecture

### Lazy Imports for Performance
All heavy dependencies (numpy, pandas, ROOT, xgboost) are imported locally at function level, not at module import. This ensures:
- Fast module loading even on first import
- Unused functionality doesn't load dependencies
- Reduced memory footprint for analysis workflows

### Efficient Caching Strategies

**DataFrame Split Caching (eval.py):**
- `_splits_cache` maintains up to 3 cached splits (train/valid, signal/background)
- Uses DataFrame object ID as key to detect reuse
- Reduces redundant boolean masking in repeated evaluations

**Histogram Caching (root/helper.py):**
- LRU cache with 128-entry limit for `_get_hist_cached()`
- Minimizes repeated file I/O for common variables
- Thread-safe and transparent to callers

**Cut Histogram Caching (python/helper.py):**
- Unlimited LRU cache for `_key_from_file()` and `_cut_from_file()`
- Metadata and cut yields cached once per file

### Hierarchical Argument Processing (plotting.py)

The argument extraction system enables flexible configuration with minimal duplication:

```
args[var] = {...parameters...}  # Level 1: Direct parameters
args[var][ecm] = {...}          # Level 2: Organized by energy
args[var][ecm][sel] = {...}     # Level 3: By energy and selection
args[var][sel_pattern] = {...}  # Alternative: By selection pattern only
```

**Pattern Matching Logic:**
1. Exact match: `sel == "final-selection"`
2. Pipe-separated: `"sel1|sel2|*pattern*"` with wildcard support
3. Wildcard prefixes: `"*pattern*"` (contains), `"*pattern"` (ends), `"pattern*"` (starts)
4. Defaults applied only if no match found

### ROOT Batch Processing
ROOT graphics operations execute in batch mode with pre-warming:
- Batch mode prevents interactive display overhead
- Graphics system pre-warmed on first import (avoids first-plot latency)
- All stat/title boxes disabled by default for publication-quality output
- Margins, fonts, and scaling fully configurable per plot type

## Dependencies

- **numpy, pandas** — Data manipulation (lazy-loaded at function level)
- **matplotlib** — Python plotting backend (lazy-loaded)
- **ROOT** — ROOT data analysis framework (lazy-loaded)
- **xgboost** — BDT model evaluation (lazy-loaded)
- **sklearn.metrics** — ROC curve computation (lazy-loaded)
- **uproot** — Efficient ROOT file I/O with caching
- **tqdm** — Progress bars for long-running operations
- **Graphviz** — Decision tree visualization (optional)

## Directory Structure

```
plots/
├── __init__.py                     # Module initialization
├── cutflow.py                      # Cutflow & efficiency analysis
│   └── Functions: get_cutflow(), CutFlow(), CutFlowDecays(), Efficiency(), 
│                   write_table(), branches_from_cuts()
├── eval.py                         # BDT performance evaluation
│   └── Functions: log_loss(), classification_error(), AUC(), roc(), bdt_score(),
│                   mva_score(), importance(), significance(), efficiency(), 
│                   tree_plot(), hist_check()
├── plotting.py                     # General histogram plotting
│   └── Functions: makePlot(), PlotDecays(), PseudoRatio(), Bias(), AAAyields(),
│                   significance(), get_args(), args_decay(), hist_to_arrays()
├── python/                         # Matplotlib and event flow utilities
│   ├── __init__.py
│   ├── plotter.py                  # Styling and export
│   │   └── Functions: set_plt_style(), set_labels(), savefigs()
│   └── helper.py                   # Event flow and histogram access
│       └── Functions: get_flow(), get_flow_decay(), get_flows(), dump_json(),
│                       find_sample_files(), get_processed(), get_cut(),
│                       getcut(), get_count(), is_there_events()
└── root/                           # ROOT utilities
    ├── __init__.py
    ├── plotter.py                  # Canvas and histogram creation
    │   └── Functions: canvas(), canvasRatio(), dummy(), dummyRatio(),
    │                   aux(), auxRatio(), setup_cutflow_hist(),
    │                   finalize_canvas(), save_canvas()
    └── helper.py                   # Histogram and axis configuration
        └── Functions: make_cfg(), build_cfg(), canvas_margins(), pad_margins(),
                        mk_legend(), load_hists(), axis_limits(), configure_axis(),
                        style_hist(), style_hists_batch(), setup_latex(),
                        draw_latex(), y_offset(), savecanvas()
```

**Module Responsibilities:**

| Module | Purpose | Main Exports |
|--------|---------|--------------|
| `plotting.py` | High-level plotting interface with flexible configuration | Main plotting functions, argument extractors |
| `cutflow.py` | Selection efficiency and event flow analysis | Cutflow rendering, efficiency tables, utility functions |
| `eval.py` | BDT model evaluation and diagnostics | Training metrics, ROC, feature importance, optimization |
| `python/plotter.py` | Matplotlib styling and figure export | Global style configuration, label decoration, multi-format export |
| `python/helper.py` | Event flow aggregation from ROOT files | Flow building, histogram access, file discovery, JSON export |
| `root/plotter.py` | ROOT canvas and histogram creation | Canvas types, dummy templates, plot finalization, export |
| `root/helper.py` | ROOT axis/histogram configuration and styling | Legend creation, axis setup, histogram styling, text rendering |

## Common Workflows

### Event Selection and Cutflow Analysis

**Typical flow:**
1. **Aggregate events** using `python/helper.get_flows()` → build flow dictionaries from ROOT files
2. **Render cutflow plots** using `cutflow.CutFlow()` → stacked backgrounds with signal overlay
3. **Analyze efficiency** using `cutflow.CutFlowDecays()` → compare selection efficiency per Higgs decay mode
4. **Export summary tables** using `cutflow.write_table()` → publish yields with uncertainties

**Key insight:** Event flow data structure enables efficient aggregation and reuse across multiple plot types.

### BDT Model Development and Validation

**Typical flow:**
1. **Monitor training** using `eval.log_loss()`, `eval.classification_error()`, `eval.AUC()` → diagnose convergence
2. **Evaluate performance** using `eval.roc()` → compute and visualize AUC
3. **Validate inputs** using `eval.hist_check()` → verify signal/background separations per decay mode
4. **Find optimal cut** using `eval.significance()` → scan BDT scores to maximize S/√(S+B)
5. **Visualize features** using `eval.importance()` → rank features by split contribution
6. **Inspect trees** using `eval.tree_plot()` → understand decision logic for interpretability

**Key insight:** BDT evaluation is split into training diagnostics (convergence), test metrics (performance), and threshold optimization (efficiency).

### Physics Distribution Comparison

**Typical flow:**
1. **Load histograms** using `root/helper.load_hists()` → batch-load with caching for performance
2. **Extract arguments** using `plotting.get_args()` → hierarchical lookup for consistent styling
3. **Draw signal/background** using `plotting.makePlot()` → render stacked histograms
4. **Compare decay modes** using `plotting.PlotDecays()` → unit-integral normalized comparison
5. **Analyze ratios** using `plotting.PseudoRatio()` → compare nominal vs. pseudo-signal distributions
6. **Summarize yields** using `plotting.AAAyields()` → render yield summary canvas with metadata

**Key insight:** Argument hierarchies enable configuration reuse; caching prevents redundant file I/O.

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

## Tips for New Readers

### Understanding the Module Organization

1. **Start with high-level functions:**
   - For cutflow: `cutflow.get_cutflow()` is your entry point
   - For BDT eval: `eval.log_loss()`, `eval.roc()`, `eval.significance()` cover most use cases
   - For plotting: `plotting.makePlot()` and `plotting.PlotDecays()` handle typical comparisons

2. **Helper modules are transparent:**
   - Event flow building (`python.helper.get_flows()`) is called by `cutflow.get_cutflow()`
   - Histogram loading with caching happens automatically in ROOT helper functions
   - Don't need to interact with helpers directly unless doing custom analysis

3. **Configuration is hierarchical:**
   - Store plot args in nested dicts: `args[var][ecm][sel] = {...}`
   - Use wildcards for multiple selections: `"Baseline*"` matches `Baseline`, `Baseline_high`, `Baseline_low`
   - Missing parameters auto-populate with sensible defaults

### Performance Considerations

1. **Lazy imports:** Heavy libraries load only when needed → fast module import
2. **Caching:** Repeated calls reuse cached data → faster batch operations
3. **Batch ROOT:** Graphics operations never show interactive windows → faster execution
4. **Vectorized operations:** Use numpy/pandas operations, not loops

### Common Gotchas

1. **Selection naming:** Cuts named "cut0", "cut1", not "cut1", "cut2" (zero-indexed for bin matching)
2. **Energy filtering:** If `args[var]['ecm'] != current_ecm`, those args are cleared (not applied)
3. **Wildcard precedence:** More specific patterns matched first; pipe operators combine with OR
4. **ROOT in batch mode:** Can't display plots interactively; use `savefigs()` to export

### Debugging and Introspection

To understand what's happening in your analysis:

```python
# Check what arguments were extracted
from package.plots.plotting import get_args
config = get_args('m_ll', 'final-selection', 240, 10.8, args_dict)
print(config)  # See full config with defaults

# Inspect event flow structure
from package.plots.python.helper import get_flows
flows = get_flows(...)
print(flows.keys())         # See process names
print(flows['ZeeH'].keys()) # ['cut', 'err', 'hist'] dicts

# Check cached histogram status
from package.plots.root.helper import _get_hist_cached
print(_get_hist_cached.cache_info())  # View cache statistics
```
