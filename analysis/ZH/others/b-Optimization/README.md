# b-Optimization: Chi2 Pairing Optimization

Chi2-based pairing optimization for the FCC-ee ZH cross-section measurement. Optimizes the weighting parameter of chi2 distance metrics to maximize pairing efficiency for dilepton and dijet final states.

## Overview

This module performs three main tasks:

1. **Pre-selection** (`pre-selection.py`): Preparation of optimization input data with chi2 distance metrics
2. **Optimization** (`optimize.py`): Parameter space scan to find optimal chi2 weighting values
3. **Visualization** (`plots.py`): Analysis of optimization results and efficiency curves

## Scripts

### Main Executable Scripts

#### `pre-selection.py` — Chi2 Optimization Data Preparation

Applies chi2 optimization algorithms to compute pairing distance metrics and prepares data for parameter scanning.

**Key operations:**
- Loads pre-selected events
- Applies channel-specific chi2 optimization algorithms:
  - **Leptonic channels** (`ee`, `mumu`): `optimize_ll()` function
    - Computes chi2 distances between reconstructed lepton pairs and true Z boson kinematics
    - Calculates: mass distance (to $m_Z = 91.2$ GeV), recoil distance (to $m_H = 125$ GeV), momentum distance (for pll method)
    - Supports two chi2 methods:
      - **mll**: $\chi^2 = d_r + \text{frac} \times (d_m - d_r)$ (mass + recoil)
      - **pll**: $\chi^2 = (1-\text{frac}) \times (d_r + 0.6 \times (d_m - d_r)) + \text{frac} \times d_p$ (adds momentum)
  - **Hadronic channels** (`qq`): `optimize_qq()` function
    - Similar optimization for dijet reconstruction
    - Handles additional jet-specific variables and b-tagging information
- Computes MC truth information for pairing validation
  - Leading/subleading lepton pairs and their true kinematics
  - Z system momentum and recoil mass
  - Pair matching indicators (both leptons correct, one correct, or both incorrect)
- Selects relevant branches (distance metrics, MC truth, event variables)
- Outputs events with chi2 information for parameter space scanning

**Input:** Pre-selected dilepton/dijet events from earlier analysis stages

**Output:** Optimization input samples with chi2 branches
```
output/data/optimisation/{ecm}/{cat}/
  ├── {process_name}/
  │   ├── chunk0.root
  │   ├── chunk1.root
  │   └── ... (chunked event trees with chi2 metrics)
```

**Usage:**

Execute from the **xsec/** folder:
```bash
cd xsec/
fccanalysis run b-Optimization/pre-selection.py
```

Or automate through the pipeline runner:
```bash
python 0-Run/b-run.py --run 1
```

**Configuration:**
- Custom C++ optimization functions (see [functions/optimization.h](../../functions/optimization.h))
- Integration with FCCAnalysis framework
- Configurable CPU parallelization and batch submission
- Large samples split into chunks for efficient processing

---

#### `optimize.py` — Chi2 Parameter Space Optimization

Scans chi2 weighting parameter space to find optimal values that maximize pairing efficiency.

**Key operations:**
- **Data loading and filtering**:
  - Loads ROOT event trees with pre-computed chi2 distances
  - Applies mass cut: removes pairs within 3 GeV of Higgs mass (125 GeV)
  - Computes event statistics: breakdown by number of pairs per event (zero, one, multiple)
  - Uses uproot/awkward arrays for efficient data handling

- **Chi2 parameter scanning**:
  - Tests chi2_frac parameter from 0 to 1 (configurable increment, default: 0.01)
  - For each parameter value:
    - Selects best pair in each event based on minimum chi2 distance
    - Compares reconstructed pairing against MC truth
    - Computes efficiency: fraction of events with fully correct pairing
  - Tracks statistics separately for events with 0, 1, or multiple pairs
  - Precomputes all distance metrics once for efficiency

- **Efficiency metrics**:
  - **Overall efficiency**: (correct pairings) / (all events)
  - **Category breakdown**:
    - Zero pair events: No valid pairs after mass cut
    - One pair events: Single reconstructed pair → efficiency typically 100%
    - Multi-pair events: Multiple valid pairs → efficiency reduced by combinatorial background
  - **Pairing quality**: counts of correct (both leptons match), partial (one match), and incorrect (no match) assignments

- **Result output**:
  - Saves optimization results as JSON with chi2 values and efficiency metrics
  - Exports kinematic distributions for baseline and optimal chi2 values as ROOT files
  - Baseline distributions: at fixed chi2 value (0.6 for mll, 0.0 for pll)
  - Optimal distributions: at chi2 value with maximum multi-pair efficiency

**Input:** Optimization input samples from `pre-selection.py`
```
output/data/optimisation/{ecm}/{cat}/{process_name}/
```

**Output:** Optimization results and distributions
```
output/data/optimisation/results/{ecm}/{cat}/{process_name}/{chi2_method}/
  ├── results.json                    # Chi2 scan results for all parameter values
  ├── results_baseline.root           # Kinematic distributions at baseline chi2
  └── results_optimal.root            # Kinematic distributions at optimal chi2
```

**Usage:**

Execute from the **xsec/** folder:
```bash
cd xsec/
python b-Optimization/optimize.py --cat CHANNEL --ecm ENERGY [--method METHOD] [--incr STEP] [--procs PROCESSES]
```

**Arguments:**
- `--cat`: Final state (`ee`, `mumu`, or `qq`, required)
- `--ecm`: Center-of-mass energy (`240` or `365`, required)
- `--method`: Chi2 methods to optimize, separated by `-` (default: `mll-pll`)
  - `mll` — Mass + recoil chi2
  - `pll` — Mass + recoil + momentum chi2
- `--incr`: Increment step for chi2_frac scan (default: 0.01)
  - Smaller values (e.g., 0.001) for finer parameter space exploration
  - Larger values (e.g., 0.1) for quick tests
- `--procs`: Specific process(es) to optimize, separated by `-` (default: all processes)
  - `all` — Base process without decay suffix (e.g., `wzp6_ee_eeH_ecm240`)
  - `a`, `bb`, `cc`, etc. — Higgs decay modes (short names)
  - Full process names (e.g., `wzp6_ee_eeH_Haa_ecm240`)
- `--nevents`: Maximum number of events to load per process (default: -1, load all)

**Examples:**
```bash
python optimize.py --cat ee --ecm 240                         # Optimize both methods
python optimize.py --cat mumu --ecm 365 --method mll          # Optimize mll only
python optimize.py --cat ee --ecm 240 --incr 0.05 --procs all # Coarse scan on base process
python optimize.py --cat qq --ecm 240 --method pll            # Hadronic channel optimization
```

**Output interpretation:**
- `results.json` contains efficiency for each chi2_frac value and event category
- Optimal chi2 is identified as the value with maximum efficiency in multi-pair events (most challenging category)
- Baseline vs optimal distributions show kinematic differences resulting from better pairing

**Key functions** (detailed in docstrings):
- `Optimizer.load_data()`: Efficient data loading with mass cut application
- `Optimizer.best_pair_idx()`: Best pair selection based on chi2 metric
- `Optimizer.test_chi2()`: Efficiency computation for given chi2 parameter
- `Optimizer.extract_distributions()`: Kinematic variable extraction for specific chi2
- `Optimizer.optimize()`: Full parameter space scan
- `Optimizer.save_results()`: JSON output of optimization results

---

#### `plots.py` — Optimization Results Visualization

Generates comprehensive plots from optimization scans showing efficiency curves and kinematic distributions.

**Key operations:**
- **Efficiency plots** (`efficiency()`):
  - Plots pairing efficiency vs chi2 weighting parameter
  - Highlights optimal chi2 value with maximum efficiency
  - Vertical line at optimal value for easy reference
  - Supports full y-axis range option for detailed view

- **Pairing composition analysis** (`pairing_composition()`):
  - Stacked area plots showing fraction of:
    - Both leptons correctly paired (green)
    - One lepton correctly paired (orange)
    - Both leptons incorrectly paired (red)
  - Tracks efficiency improvement from baseline to optimal chi2
  - Percentage of events in each category

- **Event count plots** (`event_counts()`):
  - Absolute event counts for each pairing outcome category
  - Log scale to handle wide range of background/signal ratios
  - Helps identify how many events contribute to efficiency

- **Kinematic distribution comparisons** (`plot_comp()`):
  - Overlays reconstructed and true kinematic distributions
  - Compares baseline vs optimal chi2 values
  - Shows improvement in reconstruction quality from optimization
  - Both linear and log scale versions

- **Origin-colored distributions** (`plot_origin()`):
  - Overlays histograms colored by pairing match quality
  - Separates incorrect, partial, and correct pairings
  - Shows how each pairing category contributes to distribution
  - Baseline reconstructions shown as scatter points

**Input:** Optimization results and distributions
```
output/data/optimisation/results/{ecm}/{cat}/{process_name}/{chi2_method}/
  ├── results.json
  ├── results_baseline.root
  └── results_optimal.root
```

**Output:** Optimization visualization plots
```
output/plots/optimisation/{ecm}/{cat}/{process_name}/{chi2_method}/
  ├── overall/              # Plots for all events
  │   ├── efficiency.png
  │   ├── pairing_composition.png
  │   ├── event_counts.png
  │   └── {var}_{scale}.png (kinematic distributions)
  ├── one/                  # Plots for one-pair events
  │   └── ... (same structure)
  └── several/              # Plots for multi-pair events (most important)
      └── ... (same structure)
```

**Usage:**
```bash
python plots.py --cat CHANNEL --ecm ENERGY [--method METHOD] [--procs PROCESSES]
```

**Arguments:**
- `--cat`: Final state (`ee`, `mumu`, or `qq`, required)
- `--ecm`: Center-of-mass energy (`240` or `365`, required)
- `--method`: Chi2 methods to plot, separated by `-` (default: `mll-pll`)
- `--procs`: Specific process(es) to plot, separated by `-` (default: all processes)
- `--metrics`: Generate optimization metric plots (default: enabled)
- `--dist`: Generate kinematic distribution comparison plots (default: enabled)

**Examples:**
```bash
python plots.py --cat ee --ecm 240                    # Generate all plots
python plots.py --cat mumu --ecm 365 --method mll    # Plot mll method only
python plots.py --cat ee --ecm 240 --procs all       # Plot base process only
python plots.py --cat qq --ecm 240 --dist            # Distribution plots only
```

**Plot interpretation:**
- **Efficiency curves**: Shows how pairing efficiency varies across chi2 parameter range
- **Composition plots**: Stacked bars tell story of reconstruction quality improvement
- **Distribution plots**: Linear comparison shows overall kinematic patterns; log scale reveals tails
- **Origin-colored plots**: Signal (green) vs background (red/orange) separation in different pairing regimes

**Key functions** (detailed in docstrings):
- `load_results()`: Load JSON optimization results
- `extract_arrays()`: Transform JSON to numpy arrays for plotting
- `_get_chi2_params()`: Method-specific formatting and baseline values
- `efficiency()`, `pairing_composition()`, `event_counts()`: Optimization metric plots
- `load_data()`: Load ROOT distributions
- `plot_comp()`, `plot_origin()`: Kinematic distribution plots
- `compare_dists()`: Orchestrates distribution comparison analysis

---

### Workflow Integration

The optimization stage can be integrated into the analysis pipeline:

```bash
cd 0-Run/
python b-run.py --cat ee-mumu --ecm 240-365 --run 1-2-3
```

**Stages:**
- Stage 1: Pre-selection (data preparation)
- Stage 2: Optimization (parameter scanning)
- Stage 3: Plots (results visualization)

For more details on pipeline execution, see [0-Run/README.md](../0-Run/README.md).

---

## Output Structure

### Data Output

```
output/
├── data/
│   └── optimisation/
│       ├── {ecm}/{cat}/
│       │   └── {process_name}/
│       │       ├── chunk0.root
│       │       └── ... (input data with chi2 metrics)
│       └── results/{ecm}/{cat}/
│           └── {process_name}/{chi2_method}/
│               ├── results.json
│               ├── results_baseline.root
│               └── results_optimal.root
└── plots/
    └── optimisation/
        └── {ecm}/{cat}/
            └── {process_name}/{chi2_method}/
                ├── overall/
                ├── one/
                └── several/
```

### Results JSON Structure

The `results.json` file contains chi2 optimization scan results:

```json
{
  "0.00": {
    "chi2_frac": 0.0,
    "overall": {"efficiency": 0.75, "n_correct": 1500, ...},
    "zero_pair": {"efficiency": 1.0, "n_total": 100, ...},
    "one_pair": {"efficiency": 1.0, "n_total": 500, ...},
    "multi_pair": {"efficiency": 0.67, "n_total": 900, ...}
  },
  "0.01": { ... },
  ...
  "1.00": { ... }
}
```

Each category contains:
- `efficiency`: Fraction of events with fully correct pairing
- `n_correct`: Events with both leptons correctly assigned
- `n_partial`: Events with one lepton correctly assigned
- `n_incorrect`: Events with incorrect pairings
- `n_no_pairs`: Events with zero valid pairs (only in category breakdowns)
- `n_total`: Total events in category

### Plot Organization

Each chi2 method subdirectory contains event category folders:
- **overall/** — Results across all events
- **one/** — Results for events with exactly one valid pair (typically ~100% efficiency)
- **several/** — Results for events with multiple valid pairs (most informative for optimization)

Each category folder contains:
- `efficiency.png` — Chi2 parameter vs pairing efficiency
- `pairing_composition.png` — Stacked area of pairing outcome fractions
- `event_counts.png` — Absolute event counts by outcome
- `{var}_linear.png` / `{var}_log.png` — Distribution comparisons (linear and log scales)
  - For lepton variables: `leading_p`, `leading_pt`, `leading_theta`, `subleading_*`
  - For Z system: `zll_p`, `zll_pt`, `zll_theta`
  - For pair variables: `mass`, `recoil`

---

## Configuration Reference

Configuration is managed through the `package/` and `sel/` modules. For detailed information on:
- Chi2 optimization algorithms: see [sel/presel/chi2.py](../sel/presel/chi2.py)
- Chi2 computation functions: see [functions/optimization.h](../../functions/optimization.h)
- Path templates and I/O: see [package/userConfig.py](../package/userConfig.py)
- Plotting utilities: see [package/plots/](../package/plots/README.md)

**Default parameters**:
- **mll method**:
  - Baseline chi2_frac: 0.6 (60% mass distance, 40% recoil distance weighting)
  - Distance formula: $\chi^2 = d_r + 0.6 \times (d_m - d_r)$
- **pll method**:
  - Baseline chi2_frac: 0.0 (pure mass + recoil, no momentum distance)
  - Distance formula: $\chi^2 = (1-\text{frac}) \times (d_r + 0.6 \times (d_m - d_r)) + \text{frac} \times d_p$
  - Target Z momentum: 51 GeV (240 GeV ECM), 146 GeV (365 GeV ECM)
- **Mass definitions**:
  - Z boson mass: 91.2 GeV
  - Higgs mass: 125 GeV
- **Mass cut**: |mass - 125 GeV| > 3 GeV (removes pairs too close to Higgs mass)
- **Optimization range**: chi2_frac from 0 to 1 (default increment: 0.01)

---

## Physics Notes

- **Signal process:** $e^+e^- \to Z(\ell^+\ell^-)H$ with $\ell \in \{e, \mu\}$ for leptonic channels
- **Pairing problem:** Multiple leptons/jets in event → combinatorial ambiguity in identifying true pairs
- **Chi2 approach:** Minimize distance to known particle masses and kinematics to select most likely pairing
- **Two chi2 methods**:
  - **mll** (Mass + recoil): Uses Z mass and Higgs recoil mass constraints
  - **pll** (plus momentum): Adds Z four-momentum constraint for improved precision
- **Efficiency definition**: Fraction of events where algorithm correctly identifies both leading leptons
- **Category dependence**: Efficiency strongly depends on number of pairs (easier with one pair, challenging with multiple)
- **Systematic approach**: Baseline vs optimal comparison validates optimization procedure

---

## Data Flow Diagram

```
Raw EDM4Hep Events (pre-selected)
        ↓
   [pre-selection.py]  ← Chi2 computation (mll & pll)
        ↓
Events with chi2 metrics + MC truth
        ↓
     [optimize.py]     ← Parameter space scan
        ↓
Efficiency vs chi2_frac + Kinematic distributions
        ↓
      [plots.py]       ← Visualization
        ↓
Efficiency curves, composition plots, distributions
```

---

## Tips for New Users

1. **Start with overall efficiency**: Look at `overall/efficiency.png` to see parameter space shape
2. **Focus on multi-pair events**: The `several/` directory shows optimization performance where it matters most
3. **Compare baseline to optimal**: Distribution plots show concrete improvement in reconstruction quality
4. **Check JSON for values**: `results.json` contains exact optimal chi2 values and efficiency numbers
5. **Method comparison**: Run both mll and pll to understand contribution of momentum constraint
6. **Event categories**: One-pair events have ~100% efficiency; multi-pair efficiency is the real challenge
7. **Validation**: Overlaid distributions (baseline vs true) validate chi2 approach effectiveness
8. **Batch processing**: Use pipeline runner (`0-Run/b-run.py`) for automatic processing of multiple channels/energies/methods
9. **Fine-tuning**: Use `--incr 0.001` for final optimization; `--incr 0.1` sufficient for initial exploration
10. **Memory**: For large sample sets, process per-process or use `--nevents` to limit events per run
