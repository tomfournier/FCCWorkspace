# `package/` Module: Core Analysis Framework

Central Python package providing configuration, utilities, and analysis tools for the FCC-ee ZH cross-section measurement. This module serves as the backbone for all analysis stages—from MVA input preparation through statistical fitting.

## Overview

The `package` module is organized into several key components:

| Component | Purpose |
|-----------|---------|
| **`config.py`** | Physics constants, decay modes, color palettes, and process builders |
| **`userConfig.py`** | Path templates and global analysis parameters |
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
- `Z_DECAYS`: Standard Z boson decay channels (8 modes)
- `H_DECAYS`: Standard Higgs decay channels (10 modes, excluding invisible)
- `H_DECAYS_WITH_INV`: All Higgs decay channels including H → Invisible
- `QUARKS`: Quark decay channels for Z → qq̄

Lowercase aliases (`z_decays`, `h_decays`, etc.) provided for backward compatibility.

**Color Palettes:**
- `colors`: Process → color code mapping (ZH signal in red, WW/ZZ/rare backgrounds)
- `h_colors`: Higgs decay mode → color mapping
- `modes_color`: Analysis mode → matplotlib tab color mapping (for matplotlib plots)

All colors are lazily loaded on first access to avoid unnecessary ROOT imports.

**Physics Labels:**
- `labels`: ROOT TLatex labels for main processes (ZH, WW, ZZ, etc.)
- `h_labels`: ROOT TLatex labels for Higgs decay modes
- `vars_label`: LaTeX labels for kinematic variables (for matplotlib)
- `vars_xlabel`: LaTeX labels with units (e.g., "GeV", "GeV²")
- `modes_label`: LaTeX labels for decay modes (analysis modes)

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

**Full template list:** ROOT, PACKAGE, OUT, PLOTS, DATA, TMP, JSON, RUN, EVENTS, EVENTS_TRAINING, EVENTS_TEST, MVA, MVA_INPUTS, BDT, HIST, HIST_MVA, HIST_PREPROCESSED, HIST_PROCESSED, PLOTS_MVA, PLOTS_BDT, PLOTS_MEASUREMENT, COMBINE, COMBINE_NOMINAL, COMBINE_BIAS, and logging/result subdirectories.

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

**`get_params(env, cfg_json, is_final=False, is_presel3=False)`**
- Extract analysis parameters (category, energy, luminosity)
- Reads from JSON config file if `RUN` env variable is set; otherwise prompts user
- Flexible overloads support different return types:
  - `(cat, ecm)` — basic parameters
  - `(cat, ecm, lumi)` — with luminosity (if `is_final=True`)
  - `(cat, ecm, ww)` — with WW flag (if `is_presel3=True`)

---

## Sub-Modules

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

### Basic Usage

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
# In analysis scripts
from package.config import mk_processes, input_vars
from package.userConfig import loc

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
