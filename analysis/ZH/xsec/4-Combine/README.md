# 4-Combine: Statistical Analysis & Datacard Generation

This stage prepares input data for statistical fitting by processing and combining histograms from the measurement stage and generating COMBINE-compatible datacards for the ZH cross-section measurement.

## Overview

The `4-Combine` folder contains two main scripts:

| Script | Purpose |
|--------|---------|
| **`process_histogram.py`** | Process and combine control region histograms into analysis-ready distributions |
| **`combine.py`** | Generate COMBINE datacards with process definitions and systematic uncertainties |

## Workflow

### 1. `process_histogram.py` — Histogram Processing & Combination

Processes histograms from the measurement stage (3-Measurement) and combines them into final distributions ready for statistical fitting.

**Key operations:**
- Retrieves preprocessed histograms for signal and background processes
- Splits recoil mass spectra into high/low control regions based on BDT score (signal-like vs background-like)
- Concatenates control regions into a single histogram for analysis
- Applies optional cross-section scaling (beam polarization, ILC) to signal/background samples
- Writes processed histograms to ROOT files organized by final state and selection

**Input:** Preprocessed histograms from 3-Measurement (after BDT selection)
```
output/data/histograms/preprocessed/{ecm}/{cat}/{sample}_{sel}_histo.root
```

**Output:** Processed histograms
```
output/data/histograms/processed/{ecm}/{cat}/{sel}/
```

**Usage:**
```bash
python process_histogram.py --cat ee --ecm 240
python process_histogram.py --cat ee-mumu --ecm 365 --polL  # Apply left polarization scaling
python process_histogram.py --cat mumu --ecm 240 --ILC      # Apply ILC cross-section scaling
python process_histogram.py --cat ee --ecm 240 --sels Baseline-Baseline_miss  # Process specific selections
```

Alternatively, automate execution across all combinations:
```bash
python 0-Run/4-run.py
```

The `0-Run/4-run.py` script processes histograms for all combinations of channels, center-of-mass energies, and selections, avoiding repeated manual execution. For more details on the run scripts, see [0-Run/README.md](../0-Run/README.md).

**Command-line options:**
- `--cat` — Final state: `ee`, `mumu`, or `ee-mumu` (default: `ee-mumu`)
- `--ecm` — Center-of-mass energy: `240` or `365` GeV (default: `240`)
- `--sels` — Selection strategies to process, separated by `-` (default: `Baseline-Baseline_miss-Baseline_sep-test`)
- `--polL` — Scale to left polarization cross-sections (applies cross-section weighting)
- `--polR` — Scale to right polarization cross-sections (applies cross-section weighting)
- `--ILC` — Scale to ILC machine conditions (applies cross-section weighting)

Note: `process_histogram.py` is run standalone with command-line arguments, not through the FCCAnalysis framework. Only samples with preprocessed histograms for the specified selection are processed; missing samples are skipped with a warning.

**Configuration:**
- Selection strategies: `Baseline`, `Baseline_miss`, `Baseline_sep`, `test`
- Control regions: High BDT score (signal-like) and low BDT score (background-like) are concatenated into combined histograms
- Processed histogram: `zll_recoil_m` (Higgs recoil mass computed from $Z\rightarrow\ell^+\ell^-$)
- Cross-section scaling: Applied before histogram writing when `--polL`, `--polR`, or `--ILC` flags are specified

### 2. `combine.py` — COMBINE Datacard Generation

Generates COMBINE-compatible datacards for statistical fitting of the ZH signal and backgrounds.

**Key operations:**
- Defines signal processes: $e^+e^-\rightarrow Z(f\bar{f})H$ with all Higgs decay modes ($b\bar{b}$, $c\bar{c}$, $s\bar{s}$, $gg$, $\mu^+\mu^-$, $\tau^+\tau^-$, $ZZ^*$, $WW^*$, $Z\gamma$, $\gamma\gamma$, $\textrm{Inv}$)
    - Includes additional signal modes: $e^+e^- \rightarrow e^+e^-H$, $e^+e^- \rightarrow \mu^+\mu^-H$, and $H \rightarrow \textrm{invisible}$
    - Also accounts for $Z$ decays from quarks, $\tau$, and neutrinos
- Defines background processes: $ZZ$, $W^+W^-$, $Z/\gamma$, and rare processes
- Configures systematic uncertainties (1% log-normal normalization uncertainties for backgrounds, uncorrelated between processes)
- Prepares datacard structure for RooFit/COMBINE statistical framework
- Supports multiple analysis categories based on Higgs production/decay modes

**Execution Modes:**

1. **Interactive mode** (standalone execution):
   ```bash
   fccanalysis combine 4-Combine/combine.py
   ```
   Prompts for channel (`ee` or `mumu`), center-of-mass energy, and selection strategy.

2. **Automated mode** (via run script):
   ```bash
   python 0-Run/4-run.py --run 2
   ```
   Loads configuration from `output/tmp/config_json/4-run.json` and processes all combinations automatically. For more details on the run scripts, see [0-Run/README.md](../0-Run/README.md).

**Input:** Configuration from command-line or JSON config
```
cat:  ee or mumu (final state channel)
ecm:  240 or 365 (center-of-mass energy)
sel:  Baseline, Baseline_miss, Baseline_sep, test (selection strategy)
```

**Output:** COMBINE datacard files
```
output/data/combine/{sel}/{ecm}/{sel}/nominal/datacard/
```

**Configuration:**
- MC statistics: Excluded from default uncertainties
- Histogram rebinning: factor of 1 (no rebinning)
- Integrated luminosity scaling: 1.0 (histograms already scaled at final-selection step in 3-Measurement)
- Systematic uncertainties: 1% background normalization (log-normal, `lnN`, uncorrelated between backgrounds)

**Process categories:**
- **Signal:** 
  - $e^+e^-\rightarrow Z(f\bar{f})H$ with 10 Higgs decay modes
  - $ZZ_\textrm{noInv}$ variant (additional Higgs decay mode)
  - $e^+e^- \rightarrow e^+e^-H$ (signal)
  - $e^+e^- \rightarrow \mu^+\mu^-H$ (signal)
  - $e^+e^- \rightarrow ZH$ with invisible Higgs decays
- **Backgrounds:**
  - `ZZ` — Pair production via $e^+e^- \rightarrow ZZ$
  - `WW` — Pair production via $e^+e^- \rightarrow W^+W^-$ (all decay channels: hadronic, electron, muon)
  - `Zgamma` — Drell-Yan and photon production ($e^+e^- \rightarrow e^+e^-$, $\mu^+\mu^-$, $\tau^+\tau^-$)
  - `Rare` — Rare processes ($e^\pm\gamma \rightarrow e^\pm Z$, $\gamma\gamma \rightarrow \ell^+\ell^-$, $\tau^+\tau^-$, invisible $Z$ decays)

## Input/Output Data Flow

```
Measurement Stage (3-Measurement)
    ↓ [BDT applied: signal-like vs background-like regions]
Preprocessed histograms
    ↓ [process_histogram.py]
Processed histograms (high/low control regions)
    ↓ [combine.py]
COMBINE datacards
    ↓
Statistical fitting (5-Fit)
```

## Configuration & Parameters

### `combine.py` Configuration

`combine.py` reads configuration from two modes:

1. **Interactive mode** (default):
   - Prompted to select channel (`ee` or `mumu`), selection strategy, and other parameters
   
2. **Automated mode** (`RUN` environment variable):
   - Loads configuration from `output/tmp/config_json/4-run.json`
   - Automatically selected when running via `0-Run/4-run.py`

**Execution:**

*Standalone (interactive mode):*
```bash
fccanalysis combine 4-Combine/combine.py
```

*Automated (via run script):*
```bash
python 0-Run/4-run.py --run 2
```

The `0-Run/4-run.py` script automates execution across all combinations of channels, center-of-mass energies, and selections, avoiding repeated manual execution. This uses the **FCCAnalysis framework** (part of FCCWorkspace). The xsec analysis is located at `FCCWorkspace/analyses/ZH/xsec/`.

For more details on the run scripts and workflow automation, see [0-Run/README.md](../0-Run/README.md).

For detailed parameter descriptions and path management, see [package/userConfig.py](../package/README.md#userconfig--path-management--global-parameters).

For physics process configuration, see [package/config.py](../package/README.md#configpy--physics-configuration).

For histogram processing utilities and the `get_hist()` and `concat()` functions, see [package/tools/process.py](../package/tools/README.md).

## Integration with Analysis Pipeline

This stage bridges measurement and statistical fitting:

1. **Input source:** BDT-separated histograms from 3-Measurement (high BDT score = signal-like, low BDT score = background-like)
2. **Processing:** Control region combination and histogram consolidation
3. **Output target:** COMBINE datacards for RooFit statistical framework
4. **Next stage:** Statistical fitting (5-Fit) to extract ZH cross-section

## Key Dependencies

- **[package/userConfig.py](../package/README.md)** — Path templates and global configuration
- **[package/config.py](../package/README.md)** — Physics processes, decay modes, and constants
- **[package/tools/process.py](../package/tools/README.md)** — Histogram retrieval with scaling and caching
- **[package/tools/utils.py](../package/tools/README.md)** — File I/O and utility functions
- **ROOT** — Histogram I/O and physics data format
- **numpy** — Numerical operations (implicit through ROOT)

## Notes

- Histograms are processed per final state (`ee`, `mumu`) and center-of-mass energy (`240`, `365` GeV)
- Control region splitting (high/low BDT scores) enables enhanced signal/background separation in statistical fit
- Background normalization uncertainties (1%) are uncorrelated between different background processes (ZZ, WW, Zgamma, Rare)
- Integrated luminosity is set to 1.0 because histograms are already weighted by luminosity at the final-selection step in 3-Measurement
- Cross-section scaling (polarization, ILC) is optional in `process_histogram.py` and applied before histogram writing
- FCCAnalysis framework setup required: run `setup_FCCAnalyses.sh` in FCCWorkspace before executing `fccanalysis combine` commands
