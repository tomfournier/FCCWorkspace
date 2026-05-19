# a-FSR: Final State Radiation Recovery Analysis

Final State Radiation (FSR) recovery and analysis stage for the FCC-ee ZH cross-section measurement. Reconstructs true photon kinematics from detector response and analyzes FSR properties to validate lepton reconstruction quality.

## Overview

This module performs two main tasks:

1. **Pre-selection** (`pre-selection.py`): FSR recovery algorithm application and branch selection
2. **Visualization & Analysis** (`plots.py`): FSR distributions, efficiency scans, and statistical significance optimization

## Scripts

### Main Executable Scripts

#### `pre-selection.py` — FSR Recovery Pre-Selection

Applies FSR (Final State Radiation) recovery algorithms to reconstruct true photon kinematics and prepare analysis branches for downstream studies.

**Key operations:**
- Loads dilepton events from pre-selected samples
- Applies FSR recovery algorithm (see [sel/presel/chi2.py](../sel/presel/chi2.py) for implementation)
  - Correlates reconstructed photons with lepton kinematics
  - Recovers true photon momentum and angles from detector response
  - Computes lepton-photon associations and kinematic correlations
- Computes FSR analysis variables:
  - Photon kinematics (momentum, transverse momentum, angles) split by origin (ISR/FSR/other)
  - Lepton kinematics after FSR recovery (compared to true values)
  - Lepton-photon correlation variables (angular distances, acolinearity, etc.)
  - Number of radiated photons per event
  - Same-parent indicators for photon-lepton pair associations
- Selects relevant branches for FSR analysis
- Outputs events with full FSR information for plotting and studies

**Input:** Pre-selected dilepton events
```
Requires input from earlier stages (typically 1-MVAInputs/ or 3-Measurement/)
Raw EDM4Hep events from FCC centralized productions
```

**Output:** FSR-recovered event samples with analysis branches
```
output/data/FSR/Inputs/{ecm}/{cat}/
```

**Usage:**

Execute from the **xsec/** folder:
```bash
cd xsec/
fccanalysis run a-FSR/pre-selection.py
```

Or automate through the pipeline runner:
```bash
python 0-Run/a-run.py --run 1
```

**Configuration:**
- Custom C++ FSR recovery functions (see [functions/FSR_recovery.h](../../functions/FSR_recovery.h))
- Integration with FCCAnalysis framework
- Configurable CPU parallelization and batch submission
- Large samples split into chunks for efficient processing

---

#### `plots.py` — FSR Distributions & Significance Analysis

Generates comprehensive visualizations and performs efficiency/significance optimization for FSR recovery.

**Key operations:**
- **Photon distributions** (`photon_distributions()`): 
  - Plots reconstructed vs true photon kinematics
  - Split by origin: ISR (Initial State Radiation), FSR (Final State Radiation), or other radiations
  - Variables: momentum, transverse momentum, angle ($\theta_\gamma$)
  - Compares reconstruction quality across radiation sources

- **Lepton distributions by origin** (`leptons_origin()`):
  - Plots reconstructed vs true lepton kinematics
  - Split by parent particle: signal, $\tau$ decay, $Z$ decay, $W$ decay, $H$ decay, hadron decay
  - Variables: momentum, transverse momentum, angle, relative isolation
  - Applies isolation cut (default: $I_{rel} < 1.8$ at 240 GeV, $< 0.95$ at 365 GeV)
  - Validates lepton reconstruction with and without FSR recovery

- **Lepton-photon correlation distributions** (`correlation_distributions()`):
  - Plots angular and kinematic correlations between leptons and photons
  - Split by same_parent criterion (photon and lepton from same parent particle)
  - Variables: $\cos\theta_{\ell\gamma}$, acolinearity, acoplanarity, acopolarity, $\Delta R$
  - Evaluates photon-lepton pairing accuracy

- **Correlation scan with significance optimization** (`correlation_scan()`):
  - Scans correlation variable thresholds to maximize signal significance
  - Computes significance as $S/\sqrt{S+B}$ for each cut value
  - Identifies optimal correlation cuts for signal/background separation
  - Handles directional cuts (left-to-right vs right-to-left)

- **Number of radiated photons** (`n_radiated()`):
  - Plots distribution of photon multiplicities
  - Shows FSR activity levels and radiation patterns
  - Useful for understanding reconstruction efficiency dependence on radiation

- **Isolation cut optimization** (`significance()`):
  - Optimizes lepton isolation threshold to maximize signal significance
  - Separates signal (leptons not from hadronization) and background (from hadrons)
  - Identifies optimal isolation cut value

**Input:** FSR-recovered event trees from `pre-selection.py`
```
output/data/FSR/Inputs/{ecm}/{cat}/{process_name}/
```

**Output:** Validation plots and efficiency curves
```
output/plots/fsr/{ecm}/{cat}/
  ├── {process_name}/
  │   ├── photon/           # Photon distribution plots (linear/log scales)
  │   ├── origin/           # Lepton origin plots (linear/log scales)
  │   ├── correlation/      # Correlation distributions (linear/log scales)
  │   └── efficiency/       # Significance scans and optimization results
```

**Usage:**
```bash
python plots.py --cat CHANNEL --ecm ENERGY [--procs PROCESSES] [--optimize]
```

**Arguments:**
- `--cat`: Final state (`ee` or `mumu`, required)
- `--ecm`: Center-of-mass energy (`240` or `365`, required)
- `--procs`: Specific process(es) to plot, separated by `-` (default: all processes)
  - Format: process names as they appear in input directory
  - Example: `wzp6_ee_eeH_ecm240-wzp6_ee_bbH_Haa_ecm240`
- `--optimize`: Enable optimization mode (scan for optimal cuts)

**Examples:**
```bash
python plots.py --cat ee --ecm 240                    # Generate all plots for ee channel
python plots.py --cat mumu --ecm 365                  # Generate all plots for mumu channel
python plots.py --cat ee --ecm 240 --procs wzp6_ee_eeH_ecm240  # Plot specific process
python plots.py --cat ee --ecm 240 --optimize         # Include significance optimization
```

**Plot Options:**
- No optional flags: generates all standard plots
- Each variable category generates both linear and logarithmic scale versions
- Efficiency plots show optimal cut values identified from scans
- Scatter points overlay reconstructed data on true MC distributions for validation

**Key functions** (detailed in docstrings):
- `load_data()`: Efficiently loads ROOT tree data using uproot/awkward arrays
- `_plot_histograms()`: Helper for multi-dataset histogram overlays
- `_compute_cumsum_and_z()`: Cumulative sum and significance computation
- `photon_distributions()`, `leptons_origin()`, `correlation_distributions()`: Main distribution plots
- `correlation_scan()`, `significance()`: Optimization with significance metrics

---

### Workflow Integration

The FSR recovery stage can be integrated into the analysis pipeline:

```bash
cd 0-Run/
python a-run.py --cat ee-mumu --ecm 240-365 --run 1-2
```

**Stages:**
- Stage 1: Pre-selection (FSR recovery)
- Stage 2: Plots (FSR distributions and analysis)

For more details on pipeline execution, see [0-Run/README.md](../0-Run/README.md).

---

## Output Structure

### Data Output

```
output/
├── data/
│   └── FSR/
│       └── Inputs/
│           └── {ecm}/{cat}/
│               ├── {process_name}/
│               │   ├── chunk0.root
│               │   ├── chunk1.root
│               │   └── ... (chunked event trees)
│               └── ... (other processes)
└── plots/
    └── fsr/
        └── {ecm}/{cat}/
            └── {process_name}/
                ├── photon/       # ISR/FSR-split photon distributions
                ├── origin/       # Lepton origin distributions
                ├── correlation/  # Lepton-photon correlation plots
                └── efficiency/   # Significance optimization curves
```

### Plot Organization

Each process directory contains:
- **photon/** — Photon kinematics split by radiation source
  - `ph_p_lin.png` / `ph_p_log.png` — Photon momentum
  - `ph_pT_lin.png` / `ph_pT_log.png` — Photon transverse momentum
  - `ph_theta_lin.png` / `ph_theta_log.png` — Photon angle

- **origin/** — Lepton kinematics split by decay origin
  - `leps_p_lin.png` / `leps_p_log.png` — Lepton momentum
  - `leps_pT_lin.png` / `leps_pT_log.png` — Lepton transverse momentum
  - `leps_theta_lin.png` / `leps_theta_log.png` — Lepton angle
  - `leps_iso_lin.png` / `leps_iso_log.png` — Relative isolation

- **correlation/** — Angular and kinematic correlations
  - `cosTheta_*.png` — Cosine of lepton-photon angle
  - `acolinearity_*.png` — Acolinearity angle
  - `acoplanarity_*.png` — Acoplanarity angle
  - `acopolarity_*.png` — Acopolarity angle
  - `deltaR_*.png` — Angular distance ($\Delta R$)

- **efficiency/** — Optimization results
  - `significance_iso.png` — Isolation cut optimization
  - `significance_{var}.png` — Correlation variable scans (one per variable)
  - JSON logs with optimal cut values

---

## Configuration Reference

Configuration is managed through the `package/` and `sel/` modules. For detailed information on:
- FSR recovery algorithm: see [sel/presel/chi2.py](../sel/presel/chi2.py)
- FSR kinematics functions: see [functions/FSR_recovery.h](../../functions/FSR_recovery.h)
- Path templates and I/O: see [package/userConfig.py](../package/userConfig.py)
- Plotting utilities: see [package/plots/](../package/plots/README.md)

**Default cut values** (from `plots.py`):
- Isolation cut: $I_{rel} < 1.80$ at 240 GeV, $I_{rel} < 0.95$ at 365 GeV
- Increment for optimization scans: 0.01 for correlation variables, 0.1 for isolation

---

## Physics Notes

- **Signal process:** $e^+e^- \to Z(\ell^+\ell^-)H$ with $\ell \in \{e, \mu\}$
- **FSR recovery:** Reconstructs true photon kinematics from detector response to improve lepton reconstruction
- **Radiation sources:**
  - **ISR** — Initial State Radiation (from incoming electrons/positrons)
  - **FSR** — Final State Radiation (from final-state leptons)
  - **Other** — Radiations not classified as ISR/FSR
- **Validation approach:** Compare reconstructed kinematics (with and without FSR recovery) against MC truth
- **Efficiency metrics:** 
  - Significance $Z = S/\sqrt{S+B}$ for signal/background separation
  - Pairing efficiency for lepton-photon associations
  - Reconstruction improvement factors from FSR recovery

---

## Data Flow Diagram

```
Raw EDM4Hep Events
        ↓
   [pre-selection.py]  ← FSR Recovery Algorithm
        ↓
FSR-Recovered Event Trees (with branches)
        ↓
     [plots.py]        ← Validation & Optimization
        ↓
   Distribution Plots + Efficiency Curves
```

---

## Tips for New Users

1. **Start with validation plots**: Run `plots.py` on a signal process to understand FSR recovery performance
2. **Check optimization results**: Look at `efficiency/` subdirectory for optimal cut values
3. **Understand plot structure**: Each plot shows reconstructed (scatter points) vs true (histograms) for validation
4. **Color coding**: ISR (blue), FSR (green), Other (red) in photon distributions; multiple colors for lepton origins
5. **Significance scans**: Red star marks optimal cut; blue line shows significance trend across cut values
6. **Batch processing**: Use pipeline runner (`0-Run/a-run.py`) for automatic processing of multiple channels/energies
