# `package/` Module: Core Analysis Framework

Central Python package providing configuration, utilities, and analysis tools for the FCC-ee ZH cross-section measurement. This module serves as the backbone for all analysis stages—from MVA input preparation through statistical fitting.

## Overview

The `package` module is organized into several key components:

| Component | Purpose |
|-----------|---------|
| **`config.py`** | Physics constants, decay modes, color palettes, and process builders |
| **`userConfig.py`** | Path templates and global analysis parameters |
| **`parsing.py`** | Centralized argument parsing for analysis scripts |
| **`logger.py`** | Unified logging configuration and management |
| **`plots/`** | Visualization utilities (cutflow, BDT evaluation, histogram plotting) |
| **`func/`** | Analysis functions (BDT training/evaluation, bias testing) |
| **`tools/`** | Data processing (ROOT I/O, histogram caching, significance calculations) |

### Development Status

- **`presel/`** and **`pyplots/`**: Under active development; APIs may change
- **`userConfig_old.py`**: Deprecated; will be removed in future versions

---

## Core Configuration Modules

### `config.py` — Physics Configuration

Provides physics constants, decay modes, color schemes, and process builders for the FCC-ee ZH analysis.

#### Key Datasets

**BDT Input Variables:**
- `input_vars`: Tuple of 9 kinematic variables used as features for BDT training
  - Leading/subleading lepton: momentum, polar angle
  - Di-lepton system: mass, momentum, polar angle, acolinearity, acoplanarity

**Decay Modes:**
- `Z_DECAYS`: Standard Z boson decay channels (`'bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu'`)
- `H_DECAYS`: Standard Higgs decay channels (`'bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa'`) — excludes invisible
- `H_DECAYS_WITH_INV`: All Higgs decay channels including H → Invisible (`H_DECAYS + ('inv',)`)
- `H_DECAYS_ALL`: Extended Higgs decay modes including `'ZZ_noInv'`
- `QUARKS`: Quark decay channels (`'bb', 'cc', 'ss', 'qq'`)

Lowercase aliases (`z_decays`, `h_decays`, `H_decays`, `quarks`) provided for backward compatibility.

**Color Palettes:**
- `colors`: Process → ROOT color code mapping (uses lazy loading via `LazyDict`)
  - Signal: ZH processes mapped to red
  - Backgrounds: WW/ZZ/rare mapped to orange/blue/gray respectively
  - Lazy-loaded on first access to avoid unnecessary ROOT imports
- `h_colors`: Higgs decay mode → ROOT color code (lazy-loaded)
  - Maps decay modes (bb, cc, ss, gg, mumu, tautau, ZZ, WW, Za, aa, inv) to distinct colors
- `modes_color`: Analysis mode → matplotlib tab color mapping (matplotlib colors, not lazy)
  - Maps analysis modes (ZmumuH, ZZ, etc.) to tab colors for matplotlib plots

**Physics Labels:**
- `labels`: ROOT TLatex labels for main processes (ZH, ZmumuH, ZeeH, ZqqH, WW, ZZ, Zgamma, Rare)
- `h_labels`: ROOT TLatex labels for Higgs decay modes (e.g., `'H#rightarrowb#bar{b}'`)
- `vars_label`: LaTeX labels for kinematic variables using `$...$` notation for matplotlib
- `vars_xlabel`: LaTeX labels with units appended (e.g., `'$p_{\ell}$ [GeV]'`)
- `modes_label`: LaTeX labels for analysis modes/physics processes (e.g., `'$e^+e^-\rightarrow Z(\mu^+\mu^-)H$'`)
- `process_label`: LaTeX labels for Higgs and boson decay channels

#### Key Functions

**`mk_processes(procs=None, z_decays=None, h_decays=None, H_decays=None, quarks=None, ecm=240)`**
- Generate process dictionary with optional filtering and custom decay modes
- Uses cached defaults if all decay parameters are None
- Supports filtering to specific processes (e.g., `procs=['ZH', 'WW']`)
- Returns dict mapping process keys to tuples of full FCC process names
- Examples:
  ```python
  all_procs = mk_processes()  # All processes with defaults
  zh_ww = mk_processes(procs=['ZH', 'WW'], ecm=365)  # Specific processes at 365 GeV
  bb_cc = mk_processes(h_decays=['bb', 'cc'])  # Custom Higgs decays
  ```

**`get_process_list(cat, ecm, z_decays=Z_DECAYS, h_decays=H_DECAYS_ALL, train=False, batch=False, ...)`**
- Full-featured process builder with signal/background handling for analysis workflows
- **Args:**
  - `cat` (str): Channel ('ee' or 'mumu')
  - `ecm` (int): Center-of-mass energy (240 or 365 GeV)
  - `z_decays`, `h_decays`: Custom decay mode tuples
  - `train` (bool): Use training-mode signals if True
  - `batch` (bool): Batch processing mode
  - `onlysig`, `onlybkg`: Return only signal or background processes
  - `frac`, `chunks`: Per-process data fractions and chunk counts
  - `include`, `exclude`: Include/exclude specific processes
- **Returns:** Dict mapping process names to dicts with properties (luminosity, cross-section, fraction, chunks)
- Use case: Build complete process configuration for BDT training, measurement, or bias tests

**`warning(log_msg, lenght=-1, abort_msg=' ERROR CODE ')`**
- Print formatted error message and raise exception
- Auto-calculates box width if not specified
- Useful for user-facing error reporting

**`timer(t)`**
- Print formatted elapsed time since timestamp
- Outputs human-readable format: hours, minutes, seconds, milliseconds
- Call at end of scripts: `from time import time; t = time(); ...; timer(t)`

---

### `userConfig.py` — Path Management & Global Parameters

Manages file paths, expansion of path templates, and global analysis parameters.

#### Global Parameters

```python
plot_file = ['png']          # Output plot file formats
frac, nb  = 1, 10            # Fraction of data to use, number of chunks
ww = True                    # Flag to include WW process
cat, ecm = 'mumu', 240       # Category ('ee' or 'mumu') and CoM energy
lumi = 10.8                  # Integrated luminosity (ab⁻¹)
```

- `lumi` is automatically set: 10.8 ab⁻¹ at 240 GeV; 3.12 ab⁻¹ at 365 GeV

#### Path Templates

Templates use placeholders (`cat`, `ecm`, `sel`) for dynamic expansion:

```python
loc.EVENTS      = "output/data/events/ecm/cat/full/analysis"
loc.MVA_INPUTS  = "output/data/MVA/ecm/cat/sel/MVAInputs"
loc.BDT         = "output/data/MVA/ecm/cat/sel/BDT"
loc.HIST_PROCESSED = "output/data/histograms/processed/ecm/cat/sel"
loc.PLOTS_BDT   = "output/plots/evaluation/ecm/cat/sel"
```

**Complete template list** organized by category:

| Category | Templates | Purpose |
|----------|-----------|---------|
| **Base Paths** | ROOT, PACKAGE, OUT, PLOTS, DATA, TMP | Repository and output structure |
| **Configuration** | JSON, RUN | Config files and per-run settings |
| **Events** | EVENTS, EVENTS_TEST, EVENTS_TRAINING, EVENTS_TRAIN_TEST | Analysis and training event samples |
| **Optimization** | OPTIMISATION, OPTIMISATION_TEST, OPTIMISATION_RES | Selection optimization inputs/results |
| **FSR** | FSR_TREE, FSR_TEST, FSR_RES | Final-state radiation analysis |
| **MVA** | MVA, MVA_INPUTS, BDT | BDT input variables and trained models |
| **Histograms** | HIST, HIST_MVA, HIST_PREPROCESSED, HIST_PROCESSED, HIST_OPTIMISATION | Histogram outputs at different stages |
| **Plots** | PLOTS_MVA, PLOTS_BDT, PLOTS_MEASUREMENT, PLOTS_OPTIMISATION, PLOTS_FSR | Visualization outputs |
| **Statistical** | COMBINE, COMBINE_NOMINAL, COMBINE_BIAS, NOMINAL_*, BIAS_* | Combine tool inputs/outputs and fit results |

#### Path Expansion & Type Conversion

**LocPath class** (str-like):
- `LocPath.get(name=None, cat=None, ecm=None, sel=None, type=str)`: Expand template
- `LocPath.astype(type)`: Convert between LocPath (str) and PathObj (Path)

**PathObj class** (Path wrapper):
- `PathObj.get(name, cat=None, ecm=None, sel=None, type=str)`: Expand named template
- `PathObj.astype(type)`: Convert between PathObj (Path) and LocPath (str)

**Usage examples:**
```python
from package.userConfig import loc

# Expand template with LocPath.get()
path1 = loc.EVENTS.get(cat='ee', ecm=240, sel='Baseline')

# Expand template with loc.get()
path2 = loc.get('EVENTS', cat='ee', ecm=240, sel='Baseline', type=str)
path3 = loc.get('EVENTS', cat='mumu', ecm=365, type=Path)

# Type conversion
str_path = path3.astype(str)      # PathObj → LocPath
path_obj = str_path.astype(Path)  # LocPath → PathObj
```

#### Utility Functions

**`event(procs, path='', end='.root')`**
- Filter processes that contain valid ROOT event trees
- Verifies each process has 'events' TTree in ROOT files
- Returns list of valid process names

**`get_params(env, cfg_json, is_final=False, qq_allowed=False)`**
- Extract analysis parameters (category, energy, luminosity)
- In automated mode (when `RUN` env variable or HTCondor is detected), reads from JSON config file
- Otherwise, prompts user interactively for channel, energy, and settings
- **Overload variants:**
  - `is_final=False` (default): Returns `(cat, ecm, test_flag)` — basic parameters
  - `is_final=True`: Returns `(cat, ecm, lumi, test_flag)` — includes luminosity
- `qq_allowed`: If True, allows 'qq' channel; default channels are ['ee', 'mumu']
- Returns category ('ee', 'mumu', or 'qq'), energy (240/365 GeV), and flags

---

### `logger.py` — Unified Logging Configuration

Provides centralized logging system for consistent output across all analysis scripts and modules.

#### Setup & Usage

**`setup_logging(verbose=False, logger_name='FCCAnalysis')`**
- Configure the root logging system (call once at start of main script)
- **Args:**
  - `verbose` (bool): If True, show DEBUG messages; if False, show INFO and above
  - `logger_name` (str): Root logger name (default: 'FCCAnalysis')
- **Use in main script:**
  ```python
  from package.logger import setup_logging, get_logger
  setup_logging(verbose=args.verbose)  # Configure once
  LOGGER = get_logger(__name__)
  ```

**`get_logger(name, logger_root='FCCAnalysis')`**
- Get logger instance for a module (call in every module that needs logging)
- **Args:**
  - `name` (str): Logger name, typically `__name__` for hierarchical logging
  - `logger_root` (str): Root logger name (must match setup_logging call)
- **Returns:** `logging.Logger` instance inheriting root configuration
- **Use in package modules:**
  ```python
  from package.logger import get_logger
  LOGGER = get_logger(__name__)
  LOGGER.debug('Debug message')
  LOGGER.info('Info message')
  ```

**`get_verbosity_level(logger_name='FCCAnalysis')`**
- Get current logging level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR')

#### Key Features

- **Multi-line formatting**: Continuation lines are indented for readability
- **Hierarchical logger names**: Enables organized output for different modules
- **Single configuration point**: All modules inherit settings from root logger
- **Verbose flag integration**: Connect to argument parser for easy control via `-v` flag

#### Logging Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Development, variable values, flow tracing (shown only with `-v`) |
| INFO | Normal execution status, milestones (always shown) |
| WARNING | Suspicious conditions, recoverable errors |
| ERROR | Serious problems, non-fatal failures |

---

### `parsing.py` — Centralized Argument Parsing

Provides modular, reusable argument builders for consistent CLI across analysis scripts. Avoids code duplication through composition of low-level builders into high-level factories.

#### Low-Level Argument Builders

These add individual argument groups or single arguments:

| Function | Purpose |
|----------|---------|
| `add_cat_argument()` | Add `--cat`/`--cats` for channel selection (ee, mumu, qq) |
| `add_ecm_argument()` | Add `--ecm`/`--ecms` for energy selection (240, 365 GeV) |
| `add_sel_argument()` | Add `--sel` for single selection strategy |
| `add_sels_argument()` | Add `--sels` for batch selection processing |
| `add_run_argument()` | Add `--run` for pipeline stage selection (1, 2, 3, 4) |
| `add_verbose_argument()` | Add `-v`/`--verbose` for debug output |
| `add_batch_argument()` | Add `--batch` for HTCondor job submission |

#### Feature Group Builders

These add sets of related arguments:

| Function | Purpose |
|----------|---------|
| `add_bdt_eval()` | BDT evaluation flags (metric, tree, check, hl) |
| `add_plots_args()` | Plot generation flags (yields, decay, make, scan) |
| `add_cutflow_args()` | Cutflow visualization (tot flag) |
| `add_optimize_args()` | Optimization parameters (procs, method, nevents, incr, metric) |
| `add_polarization()` | Polarization/luminosity scaling (polL, polR, ILC) |
| `add_fit_args()` | Fitting parameters (pert, target, combine, bias, timer, print) |
| `add_bias_args()` | Bias test parameters (freeze, float, plot_dc) |

#### Factory Function

**`create_parser(cat_single=False, cat_multi=False, ecm_multi=False, ..., description='Analysis script')`**
- Compose modular builders into complete ArgumentParser
- **Select components by passing relevant flags:**
  ```python
  # For BDT training script
  parser = create_parser(
      cat_multi=True, ecm_multi=True,
      include_sels=True,
      bdt_eval=True
  )
  
  # For optimization script
  parser = create_parser(
      cat_single=True,
      optimize=True,
      polarization=True
  )
  
  # For fitting script
  parser = create_parser(
      cat_single=True,
      fit=True, bias=True,
      default_pert=1.0
  )
  ```
- **Returns:** `ArgumentParser` with selected arguments configured

#### Validation & Processing

**`parse_args(parser, validate_cat=False, comb=False)`**
- Parse arguments and apply validation
- **Args:**
  - `parser`: ArgumentParser instance from `create_parser()`
  - `validate_cat`: Validate category argument
  - `comb`: Allow combined categories (e.g., 'ee-mumu')
- **Returns:** `argparse.Namespace` with parsed arguments

**`set_log(args)`**
- Configure logging based on parsed arguments
- Call after `parse_args()` and before main script logic

#### Usage Example

```python
from package.parsing import create_parser, parse_args, set_log

# In analysis script (e.g., 2-BDT/train_bdt.py)
parser = create_parser(
    cat_multi=True,
    ecm_multi=True, 
    include_sels=True,
    bdt_eval=True
)
args = parse_args(parser)
set_log(args)

# Now args has all parsed values
print(f"Categories: {args.cats}")
print(f"Energies: {args.ecms}")
print(f"Plot metrics: {args.metric}")
```

#### Key Design Principles

- **Minimal base functions**: Only add arguments actually used by script
- **Composition over inheritance**: Mix and match builders for flexibility
- **Reduced duplication**: Centralized builders avoid copy-paste across scripts
- **Easy extensibility**: Add new builders without modifying existing code
- **Backward compatibility**: Legacy `include_polarizations()` function preserved

---



### `plots/` — Visualization Utilities

Comprehensive plotting and visualization for cutflow analysis, BDT evaluation, and physics distributions. See [plots/README.md](plots/README.md) for full documentation.

**Summary:** Provides `cutflow.py`, `eval.py`, and `plotting.py` functions for production-quality visualizations with ROOT and matplotlib backends.

### `func/` — Analysis Functions

Core multivariate analysis and bias testing utilities. See [func/README.md](func/README.md) for full documentation.

**Summary:** Provides BDT training/evaluation (`bdt.py`) and pseudo-data bias testing (`bias.py`).

### `tools/` — Data Processing Utilities

ROOT file operations, histogram caching, and statistical analysis. See [tools/README.md](tools/README.md) for full documentation.

**Summary:** Provides histogram I/O (`process.py`), significance calculations, and file utilities (`utils.py`).

---

## Quick Start

### Complete Analysis Script Template

```python
#!/usr/bin/env python3
"""Example analysis script using all core modules."""

# Step 1: Set up logging (FIRST, before other imports)
from package.logger import setup_logging, get_logger
from package.parsing import create_parser, parse_args, set_log

parser = create_parser(
    cat_multi=True,
    ecm_multi=True,
    include_sels=True,
    bdt_eval=True
)
args = parse_args(parser)
setup_logging(verbose=args.verbose)
set_log(args)

LOGGER = get_logger(__name__)
LOGGER.info('Starting analysis...')

# Step 2: Import configuration
from package.config import (
    mk_processes, 
    input_vars,
    labels,
    colors
)

# Step 3: Import paths and parameters
from package.userConfig import loc, get_params
from pathlib import Path

# Step 4: Get analysis parameters (from args or interactive)
import os
cat, ecm, test_flag = get_params(os.environ, 'config.json')

# Step 5: Generate processes and access paths
LOGGER.info(f'Generating processes for {cat} at {ecm} GeV')
processes = mk_processes(procs=['ZH', 'WW'], ecm=ecm)

events_path = loc.EVENTS.get(cat=cat, ecm=ecm)
bdt_path = loc.BDT.get(cat=cat, ecm=ecm, sel=args.sel, type=Path)

LOGGER.debug(f'Events path: {events_path}')
LOGGER.info('Analysis setup complete!')
```

### Basic Usage (Module Imports)

```python
# Import configuration
from package.config import (
    mk_processes, 
    input_vars, 
    Z_DECAYS, 
    H_DECAYS,
    labels,
    colors
)

# Import paths and parameters
from package.userConfig import loc, cat, ecm, lumi

# Generate process dictionary for ZH and WW at 365 GeV
processes = mk_processes(procs=['ZH', 'WW'], ecm=365)

# Access path templates
events_path = loc.EVENTS.get(cat='mumu', ecm=365, sel='Baseline')
bdt_path = loc.BDT.get(cat='ee', ecm=240, sel='Baseline', type=Path)

# Use physics constants
print(f"BDT inputs: {input_vars}")
print(f"Process labels: {labels}")
```

### Integration with Other Modules

The package is designed to be imported by analysis scripts and sub-modules:

```python
# Logging setup (in main script)
from package.logger import setup_logging, get_logger
setup_logging(verbose=True)
LOGGER = get_logger(__name__)

# Argument parsing (in main script)
from package.parsing import create_parser, parse_args
parser = create_parser(cat_multi=True, ecm_multi=True)
args = parse_args(parser)

# In all modules
from package.logger import get_logger
LOGGER = get_logger(__name__)

# Configuration access
from package.config import mk_processes, input_vars
from package.userConfig import loc, get_params

# In plotting scripts
from package.plots import plotting, cutflow, eval

# In BDT training scripts
from package.func import bdt

# In data processing scripts
from package.tools import process, utils
```

---

## Naming Conventions

### Process Names

FCC process naming follows standard patterns:

- **ZH signal**: `wzp6_ee_{Z_decay}H_H{H_decay}_ecm{energy}`
  - Examples: `wzp6_ee_mumuH_Hbb_ecm240`, `wzp6_ee_eeH_Hcc_ecm365`

- **WW background**: `p8_ee_WW_ecm{energy}` or `p8_ee_WW_{channel}_ecm{energy}`
  - Examples: `p8_ee_WW_ecm240`, `p8_ee_WW_mumu_ecm365`

- **Other backgrounds**: `p8_ee_ZZ_ecm{energy}`, `wzp6_ee_tautau_ecm{energy}`, etc.

### Path Placeholders

- `cat`: Analysis category ('ee' or 'mumu')
- `ecm`: Center-of-mass energy in GeV (240 or 365)
- `sel`: Selection name (e.g., 'Baseline', 'Baseline_sep')

---

## Configuration Examples

### Custom Process Filtering

```python
# Create processes with specific Higgs decay modes
zh_only = mk_processes(
    procs=['ZH'],
    h_decays=['bb', 'cc', 'ss'],  # Only 3 decays
    ecm=365
)

# Build all decay channels manually
all_z = ['bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu']
custom = mk_processes(
    z_decays=all_z,
    h_decays=['bb', 'WW', 'ZZ'],
    ecm=240
)
```

### Global Parameter Override

```python
# Temporarily adjust analysis parameters
from package import userConfig

userConfig.cat = 'ee'
userConfig.ecm = 365
userConfig.lumi = 3.12
```

---

## Related Analysis Scripts

- **1-MVAInputs/**: Prepare BDT input variables and histograms
- **2-BDT/**: Train and evaluate BDT models
- **3-Measurement/**: Perform cross-section measurement using BDT
- **4-Combine/**: Statistical combination using combine tool
- **5-Fit/**: Perform bias tests and final fits

---

## Dependencies

- **Core**: Python 3.7+
- **Data processing**: `uproot` (for ROOT file I/O), `numpy`, `pandas`
- **BDT training**: `xgboost`
- **Plotting**: `ROOT`, `matplotlib`
- **Statistical analysis**: `scipy`

Install via:
```bash
pip install uproot numpy pandas xgboost matplotlib scipy
```

---

## Version & Maintenance

This module is actively maintained as part of the FCC-ee ZH cross-section analysis. 

**Deprecated components:**
- `presel/` and `pyplots/`: Under active development
- `userConfig_old.py`: Scheduled for removal (use `userConfig.py` instead)

---

## Support

For questions or issues, refer to the individual sub-module READMEs or consult the analysis documentation.
