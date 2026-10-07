# Tools Module

Essential utility functions for processing, analyzing, and visualizing ROOT histograms and physics data in the FCC analysis pipeline. This module provides production-ready functions for histogram I/O, statistical analysis, and data manipulation with intelligent caching and lazy-loading for optimal performance.

## Overview

The `tools` module handles all data processing and analysis tasks for the FCC-ee physics analysis. It's organized into two main components:

| Module | Purpose |
|--------|---------|
| **process.py** | ROOT file operations, histogram retrieval, caching, and process-specific scaling |
| **utils.py** | File I/O, significance calculations, histogram utilities, and statistical functions |

## Features

### Histogram Processing & Caching
- Intelligent multi-level caching for optimal performance (process dictionaries, metadata, histograms)
- Efficient histogram preloading from multiple ROOT files
- Automatic WW cross-section corrections for leptonic decay channels
- Process-specific scaling factor application
- Histogram concatenation and combination utilities

**Key functions:**
- `preload_histograms()` — Cache histograms for rapid batch access
- `getHist()` — Retrieve and combine histograms across multiple processes
- `get_hist()` — Single-process histogram retrieval with scaling
- `clear_histogram_cache()` — Free memory after batch operations

### Statistical Analysis
- Significance calculations using multiple statistical methods (Z0, Zmu, Z)
- Optimized vectorized operations for large datasets
- Bin-by-bin significance computation across score ranges
- Signal/background separation and efficiency measurements

**Key functions:**
- `Significance()` — Full significance scan with multiple calculation methods
- `Z0()`, `Zmu()`, `Z()` — Individual significance estimators
- `get_xsec()` — Cross-section lookup with training/validation selection

### Histogram Range & Visualization Utilities
- Smart axis range determination with empty bin handling
- Logarithmic scale support with proper handling of zero/negative values
- Stacking and overlaying capabilities
- Decay-mode-specific range optimization

**Key functions:**
- `get_range()` — Optimal plot range for signal/background histograms
- `get_xrange()`, `get_yrange()` — Individual axis range computation
- `get_stack()` — Stack multiple histograms
- `get_range_decay()` — Range calculation for decay mode comparisons

### Data & File Management
- ROOT file discovery with glob pattern support
- DataFrame loading from ROOT trees and pickle files
- Process dictionary management with environment variable support
- JSON serialization for configuration storage

**Key functions:**
- `get_paths()` — File discovery with mode-based patterns
- `get_df()` — Load DataFrames from ROOT files
- `load_data()`, `to_pkl()` — Pickle file I/O
- `get_procDict()` — Process dictionary loading with env variable fallback

---

## process.py Reference

### Process Dictionary Management

#### `_get_procDict(procFile, fcc)`
Load and cache process dictionary metadata from FCC JSON file.

- **Args:**
  - `procFile` (str): Dictionary filename. Default: `'FCCee_procDict_winter2023_IDEA.json'`
  - `fcc` (str): FCC directory path. Default: `'/cvmfs/fcc.cern.ch/FCCDicts'`
- **Returns:** dict with process metadata (cross-sections, decay modes, etc.)
- **Caching:** Results cached by `(procFile, fcc)` to avoid repeated I/O

#### `getMetaInfo(proc, info, rmww, fcc, procFile)`
Retrieve process metadata with optional WW cross-section corrections.

- **Args:**
  - `proc` (str): Process name identifier
  - `info` (str): Metadata field name. Default: `'crossSection'`
  - `rmww` (bool): Remove leptonic WW decay channels. Default: `False`
  - `fcc` (str): FCC directory path
  - `procFile` (str): Dictionary filename
- **Returns:** float or None (if process not found)
- **WW Handling:** When `rmww=True` for WW processes, subtracts ee and mumu channels from total
- **Caching:** Results keyed by `(proc, info, rmww, procFile, fcc)` for context-aware lookup

### Histogram I/O and Caching

#### `preload_histograms(procs, inDir, suffix, hNames, rebin, rmww)`
Batch-load histograms from multiple ROOT files into memory cache for efficient repeated access.

- **Args:**
  - `procs` (list[str]): Process names to preload
  - `inDir` (str): Input directory path
  - `suffix` (str): File suffix. Default: `''`
  - `hNames` (list[str]): Specific histogram names to load; if None, loads all. Default: `None`
  - `rebin` (int): Rebinning factor to apply to all histograms. Default: `1`
  - `rmww` (bool): Apply WW cross-section correction. Default: `True`
- **Caching:** Stores histograms in global `HIST_CACHE` keyed by (proc, suffix, inDir)
- **Processing:**
  - Opens each ROOT file once
  - Applies rebinning and WW corrections
  - Detaches histograms with `SetDirectory(0)`
  - Caches WW scale factors to avoid recomputation in subsequent calls
- **Use Case:** Significantly speeds up plotting loops by eliminating repeated file I/O
- **Output:** Prints progress with histogram count

#### `get_hist(hName, proc, processes, inDir, suffix, rebin, proc_scales)`
Retrieve single-process histogram with automatic scaling applied.

- **Args:**
  - `hName` (str): Histogram name
  - `proc` (str): Process name
  - `processes` (dict[str, list[str]]): Process configuration mapping
  - `inDir` (str): Input directory
  - `suffix` (str): File suffix. Default: `''`
  - `rebin` (int): Rebinning factor. Default: `1`
  - `proc_scales` (dict[str, float]): Process scaling factors. Default: `{}`
- **Returns:** Scaled ROOT histogram, or None if retrieval fails
- **Processing:** Applies WW cross-section corrections, rebinning, then process-specific scales sequentially
- **Error Handling:** Returns None if file not found, file cannot be opened, or histogram not found
- **Auto-scaling:** WW scaling factor = (xsec with rmww) / (xsec without rmww)

#### `getHist(hName, procs, inDir, suffix, rebin, lazy, proc_scale, rmww, use_cache)`
Retrieve and sum histograms from multiple processes with cache support.

- **Args:**
  - `hName` (str): Histogram name
  - `procs` (list[str]): Process names to combine
  - `inDir` (str): Input directory
  - `suffix` (str): File suffix. Default: `''`
  - `rebin` (int): Rebinning factor. Default: `1`
  - `lazy` (bool): Skip missing files silently (True) or exit with error (False). Default: `True`
  - `proc_scale` (float): Global scaling factor applied after combining. Default: `1.0`
  - `rmww` (bool): Apply WW cross-section correction. Default: `True`
  - `use_cache` (bool): Check `HIST_CACHE` before opening files. Default: `True`
- **Returns:** Combined histogram, or None if no histograms found
- **Caching Behavior:** 
  - When `use_cache=True`, checks cache first and clones cached histograms to avoid modifying them
  - Accumulates histograms from cache and file I/O
  - Applies rebinning and global scaling as post-processing
- **Lazy Mode:** When `lazy=True`, missing files are skipped silently; when `False`, errors cause exit
- **Processing Order:** Cache/file retrieval → Sum histograms → Rebin → Apply global scale

#### `clear_histogram_cache()`
Clear all cached histograms and WW scaling factors to free memory.

- **Output:** Confirmation message printed
- **When to use:** After batch operations or when processing new data

### Histogram Utilities

#### `concat(h_list, hName, outName)`
Concatenate multiple 1D histograms into a single unrolled 1D histogram.

- **Args:**
  - `h_list` (list[ROOT.TH1]): Histograms to concatenate
  - `hName` (str): Name for logging/identification
  - `outName` (str): Output histogram name. If empty, uses `hName`. Default: `''`
- **Returns:** Unrolled 1D histogram with concatenated bin contents and errors
- **Use Case:** Flatten multiple histograms (e.g., jet features) into single input for BDT training
- **Implementation:** Creates TH1D with total bins = sum of all input histogram bins, copies bin contents sequentially

#### `proc_scale(hist, proc, processes, proc_scales)`
Apply process-specific scaling factor to a histogram.

- **Args:**
  - `hist` (ROOT.TH1): Histogram to scale
  - `proc` (str): Process name
  - `processes` (dict[str, list[str]]): Process configuration mapping category names to process lists
  - `proc_scales` (dict[str, float]): Scaling factors by process category. Default: `{}`
- **Returns:** Scaled histogram (modified in place)
- **Behavior:** Looks up process in `processes` dict, finds matching category, applies corresponding scale from `proc_scales`
- **No-op:** If `proc_scales` is empty or process not found in mapping, histogram returned unchanged

### Conventions

- **File naming:** `{proc}{suffix}.root` in input directory
- **Histogram detachment:** All histograms detached with `SetDirectory(0)` to prevent cleanup issues
- **WW corrections:** Automatically applied for `p8_ee_WW_ecm*` processes when `rmww=True`
- **Caching:** Global caches prevent redundant I/O (clear with `clear_histogram_cache()`)

---

## utils.py Reference

### File and Path Management

#### `get_paths(mode, path, modes, suffix)`
Discover ROOT file paths using mode-based glob patterns.

- **Args:**
  - `mode` (str): Mode key to filter paths
  - `path` (str): Base directory path
  - `modes` (dict): Mode-to-pattern mapping
  - `suffix` (str): File suffix to append. Default: `''`
- **Returns:** list of matching ROOT file paths

#### `mkdir(mydir)`
Create directory if it does not exist.

- **Args:** `mydir` (str): Directory path to create
- **Behavior:** Uses `os.makedirs()` with `exist_ok=True`

#### `get_df(filename, branches)`
Load DataFrame from ROOT file with optional branch selection.

- **Args:**
  - `filename` (str): Path to ROOT file
  - `branches` (list[str]): Specific branches to load; if empty, loads all branches. Default: `[]`
- **Returns:** pd.DataFrame from 'events' tree
- **Special Cases:**
  - If tree has no entries, returns empty DataFrame
  - Uses uproot for ROOT file I/O with pandas library
  - If no branches specified, all branches loaded from tree

#### `get_procDict(procFile, fcc)`
Load process dictionary from JSON file with environment variable support.

- **Args:**
  - `procFile` (str): Dictionary filename
  - `fcc` (str): Default FCC directory path. Default: `'/cvmfs/fcc.cern.ch/FCCDicts'`
- **Returns:** dict with process metadata (cross-sections, decay modes, etc.)
- **Search Path:** 
  - Checks `$FCCDICTSDIR` environment variable first (uses first path if colon-separated)
  - Falls back to `fcc` parameter if env var not set
- **Error Handling:** Logs error and calls `exit(1)` if file not found
- **Integration:** Used by `process.py`'s `_get_procDict()` for caching

#### `load_data(inDir, filename)`
Load preprocessed data from pickle file.

- **Args:**
  - `inDir` (str): Input directory path
  - `filename` (str): Filename without extension. Default: `'preprocessed'`
- **Returns:** pd.DataFrame
- **Path:** Constructs `{inDir}/{filename}.pkl`

#### `to_pkl(df, path, filename)`
Save DataFrame to pickle file.

- **Args:**
  - `df` (pd.DataFrame): DataFrame to save
  - `path` (str): Output directory path (created if missing)
  - `filename` (str): Filename without extension. Default: `'preprocessed'`
- **Output:** Confirmation message printed to stdout

#### `dump_json(arg, file, indent)`
Save dictionary to JSON file with formatting.

- **Args:**
  - `arg` (dict): Dictionary to save
  - `file` (str): Output file path
  - `indent` (int): JSON indentation level. Default: `4`

#### `load_json(file)`
Load dictionary from JSON file.

- **Args:** `file` (str): JSON file path
- **Returns:** dict

### Significance Calculation

#### `Z0(S, B)`
Calculate significance using the Z0 (log-likelihood ratio) method.

Formula: $\sqrt{2[(S+B)\ln(1+S/B) - S]}$

- **Args:**
  - `S` (float): Signal count
  - `B` (float): Background count
- **Returns:** float - calculated significance, or NaN if B ≤ 0
- **Use Case:** Optimal for discovery-type analyses (standard in HEP)

#### `Zmu(S, B)`
Calculate significance using the Zmu (profile likelihood ratio) method.

Formula: $\sqrt{2[S - B\ln(1+S/B)]}$

- **Args:**
  - `S` (float): Signal count
  - `B` (float): Background count
- **Returns:** float - calculated significance, or NaN if B ≤ 0
- **Use Case:** Alternative to Z0, often used for limit setting

#### `Z(S, B)`
Calculate significance using the simple statistical method.

Formula: $S/\sqrt{S+B}$

- **Args:**
  - `S` (float): Signal count
  - `B` (float): Background count
- **Returns:** float - calculated significance:
  - Returns 0.0 if both S and B ≤ 0
  - Returns NaN if B < 0
- **Use Case:** Quick approximation, less accurate than Z0/Zmu for small samples

#### `Significance(df_s, df_b, column, weight, func, score_range, nbins)`
Calculate significance across score bins using specified calculation method.

- **Args:**
  - `df_s` (pd.DataFrame): Signal data
  - `df_b` (pd.DataFrame): Background data
  - `column` (str): Score column name. Default: `'BDTscore'`
  - `weight` (str): Weight column name. Default: `'norm_weight'`
  - `func` (Callable): Significance calculation function (Z0, Zmu, Z). Default: `Z0`
  - `score_range` (tuple): Score range (min, max) for binning. Default: `(0, 1)`
  - `nbins` (int): Number of histogram bins. Default: `50`
- **Returns:** pd.DataFrame with columns `['S', 'B', 'Z']` indexed by bin edges
- **Algorithm:**
  - Bins signal and background data using specified range and bin count
  - Computes cumulative sums from high to low score (decreasing threshold)
  - Calculates significance at each threshold using provided function
  - Logs initial and inclusive significance values
- **Optimization:** Uses numpy vectorization for speed (single-pass binning, batch significance calculation)

#### `update_keys(procDict, modes)`
Update dictionary keys using reversed mode name mappings.

- **Args:**
  - `procDict` (dict): Original process dictionary
  - `modes` (list): Mode name mappings (dict with key-value pairs)
- **Returns:** dict with updated keys

#### `get_xsec(modes, training)`
Retrieve cross-section values for specified modes.

- **Args:**
  - `modes` (list): Mode names to retrieve cross-sections for
  - `training` (bool): Use training dataset process dictionary if True, nominal if False. Default: `True`
- **Returns:** dict mapping mode names to cross-section values (in fb)
- **Process Dictionary Selection:**
  - `training=True`: Uses `'FCCee_procDict_winter2023_training_IDEA.json'`
  - `training=False`: Uses `'FCCee_procDict_winter2023_IDEA.json'`
- **Key Mapping:** Applies `update_keys()` transformation to dictionary before extracting values

### ROOT Histogram Range Utilities

#### `get_stack(hists)`
Create stacked histogram by summing multiple histograms.

- **Args:** `hists` (list[ROOT.TH1]): Histograms to stack (must contain at least one)
- **Returns:** Combined histogram with `'_stack'` suffix appended to name
- **Implementation:** Clones first histogram, adds all others to it using ROOT's `Add()` method
- **Detachment:** Result is detached with `SetDirectory(0)` to prevent cleanup issues
- **Raises:** ValueError if list is empty

#### `get_xrange(hist, strict, xmin, xmax)`
Determine x-axis range based on histogram bin content.

- **Args:**
  - `hist` (ROOT.TH1): Histogram to analyze
  - `strict` (bool): If True, only count non-zero bins when determining range. Default: `True`
  - `xmin` (float | None): Lower boundary constraint (inclusive). Default: `None`
  - `xmax` (float | None): Upper boundary constraint (inclusive). Default: `None`
- **Returns:** tuple (x_min, x_max) bin edge coordinates
- **Algorithm:** 
  - Extracts bin contents using numpy vectorization (excludes under/overflow)
  - When `strict=True`, creates mask of non-zero bins and applies boundary constraints
  - Returns edge positions of first and last valid bins
- **Handles:** Fixed and variable bin widths automatically

#### `get_yrange(hist, logY, ymin, ymax, scale_min, scale_max)`
Determine y-axis range based on histogram content with optional logarithmic scaling.

- **Args:**
  - `hist` (ROOT.TH1): Histogram to analyze
  - `logY` (bool): If True, use logarithmic scale and exclude zero/negative values
  - `ymin` (float | None): Minimum y override. Default: `None`
  - `ymax` (float | None): Maximum y override. Default: `None`
  - `scale_min` (float): Scale factor for minimum (padding/zoom). Default: `1.0`
  - `scale_max` (float): Scale factor for maximum (padding/zoom). Default: `1.0`
- **Returns:** tuple (y_min, y_max) scaled bin values
- **Log Scale Handling:** When `logY=True`, filters out zero and negative bin contents before computing min/max
- **Linear Scale:** Uses all bin contents including zeros
- **Scaling:** y_min = (min value) × scale_min, y_max = (max value) × scale_max

#### `get_range(h_sigs, h_bkgs, logY, strict, stack, scale_min, scale_max, xmin, xmax, ymin, ymax)`
Determine optimal plot range for combined signal and background histograms.

- **Args:**
  - `h_sigs` (list[ROOT.TH1]): Signal histograms
  - `h_bkgs` (list[ROOT.TH1]): Background histograms
  - `logY` (bool): Use logarithmic y-axis. Default: `False`
  - `strict` (bool): Exclude empty bins from x-range calculation. Default: `True`
  - `stack` (bool): Use stacked histogram for y-max determination. Default: `False`
  - `scale_min`, `scale_max` (float): Scale factors for y-axis padding. Default: `1.0`
  - `xmin`, `xmax` (float | None): X-axis constraint bounds. Default: `None`
  - `ymin`, `ymax` (float | None): Y-axis constraint bounds. Default: `None`
- **Returns:** tuple (xmin, xmax, ymin, ymax) ready for plotting
- **Algorithm:**
  - X-range: Computed from stacked histogram of all signal + background
  - Y-min: Minimum of all individual histogram y-ranges (scaled by scale_min)
  - Y-max: If `stack=True`, computed from stacked histogram; else maximum of all individual maxima (scaled by scale_max)
- **Use Case:** Prepare axis ranges for overlaid signal/background plots

#### `get_range_decay(h_sigs, logY, strict, scale_min, scale_max, xmin, xmax, ymin, ymax)`
Determine optimal plot range for decay mode comparison histograms.

- **Args:**
  - `h_sigs` (list[ROOT.TH1]): Signal/decay mode histograms to compare
  - `logY` (bool): Use logarithmic y-axis. Default: `False`
  - `strict` (bool): Exclude empty bins from x-range. Default: `True`
  - `scale_min`, `scale_max` (float): Scale factors for y-axis. Default: `1.0`
  - `xmin`, `xmax` (float | None): X-axis constraints. Default: `None`
  - `ymin`, `ymax` (float | None): Y-axis constraints. Default: `None`
- **Returns:** tuple (xmin, xmax, ymin, ymax)
- **Algorithm:**
  - X-range: From stacked histogram of all decay modes
  - Y-range: Min/max across all individual decay histograms (no stacking for y-max)
- **Use Case:** Compare multiple Higgs decay channels on same plot without background

### Utilities

#### `high_low_sels(sels, list_hl)`
Expand selection list with high/low variants for specified selections.

- **Args:**
  - `sels` (list[str]): Original selection list
  - `list_hl` (str | list[str]): Selection names to expand
- **Returns:** Updated list with `{sel}_high` and `{sel}_low` variants appended

### Conventions

- **ROOT tree name:** Expected as `'events'` for all DataFrame loading operations
- **Process dictionaries:** Searched via `$FCCDICTSDIR` (first path if colon-separated), falls back to `/cvmfs/fcc.cern.ch/FCCDicts`
- **File globbing:** Patterns in `modes` mapping used with `os.path.join()` to build full paths
- **Range calculations:** `strict=True` excludes empty bins; `logY=True` ignores zero/negative contents
- **Lazy imports:** Heavy dependencies (numpy, pandas, ROOT, uproot) imported only when function called

---

## Usage Example

```python
from package.tools.process import getHist, preload_histograms, clear_histogram_cache
from package.tools.utils import get_paths, Significance

# Preload histograms for faster access
preload_histograms(
    procs=['p8_ee_WW_ecm240', 'p8_ee_Zcc_ecm240'],
    inDir='./histograms/',
    suffix='',
    rmww=True
)

# Retrieve a combined histogram from multiple processes
hist = getHist(
    hName='jet_n',
    procs=['p8_ee_WW_ecm240', 'p8_ee_Zcc_ecm240'],
    inDir='./histograms/',
    use_cache=True,
    rmww=True
)

# Calculate significance
from package.tools.utils import Z0
import pandas as pd

df_signal = pd.DataFrame({'BDTscore': [...], 'norm_weight': [...]})
df_background = pd.DataFrame({'BDTscore': [...], 'norm_weight': [...]})

z_df = Significance(df_signal, df_background, func=Z0, nbins=50)

# Clean up
clear_histogram_cache()
```

---

## Caching Architecture

The module implements a three-level caching system for production-grade performance:

1. **Process Dictionary Cache (`PROCDICT_CACHE`)**
   - Avoids repeated JSON file I/O
   - Keyed by `(procFile, fcc)` tuple
   - Shared across all `getMetaInfo()` and `_get_procDict()` calls

2. **Metadata Cache (`XSEC_CACHE`)**
   - Stores computed cross-section values and WW corrections
   - Keyed by `(proc, info, rmww, procFile, fcc)` for context-aware results
   - Enables efficient repeated lookups of WW scaling factors

3. **Histogram Cache (`HIST_CACHE`)**
   - Preloaded histograms for rapid batch-access patterns
   - Keyed by `(proc, suffix, inDir)` with histogram name lookup
   - Cleared with `clear_histogram_cache()` to free memory after batch operations

## Implementation Details

- **Lazy imports:** Heavy dependencies imported only when needed (numpy, pandas, ROOT, uproot)
- **Vectorized operations:** Significance and range calculations use numpy for vectorized speed
- **WW scale persistence:** Per-process scaling factors cached during `preload_histograms()` and `getHist()` to avoid redundant computations
- **Error handling:** 
  - `lazy=True` (default): Missing files/histograms logged as LOGGER warnings, processing continues
  - `lazy=False`: Missing files/histograms cause LOGGER.error() and program exit with code 1
- **Histogram safety:** All histograms detached with `SetDirectory(0)` to prevent ROOT cleanup issues
- **Backward compatibility:** All function signatures stable with sensible defaults

## Dependencies

- **numpy, pandas** — Data manipulation and analysis (lazy-loaded)
- **ROOT** — ROOT data analysis framework (lazy-loaded on histogram access)
- **uproot** — High-performance ROOT file I/O
- **tqdm** — Progress bars for batch operations
- **json, os** — File system and configuration management

## Directory Structure

```
tools/
├── __init__.py             # Module initialization
├── process.py              # Histogram I/O, caching, processing
├── utils.py                # File I/O, statistics, histogram utilities
└── README.md               # This file
```
