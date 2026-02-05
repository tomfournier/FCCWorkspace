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
```

Alternatively, automate execution across all combinations:
```bash
python run/4-run.py
```

The `run/4-run.py` script processes histograms for all combinations of channels, center-of-mass energies, and selections, avoiding repeated manual execution. For more details on the run scripts, see [run/README.md](../run/README.md).

**Command-line options:**
- `--cat` — Final state: `ee`, `mumu`, or `ee-mumu` (default: `ee-mumu`)
- `--ecm` — Center-of-mass energy: `240` or `365` GeV (default: `240`)
- `--polL` — Scale to left polarization cross-sections
- `--polR` — Scale to right polarization cross-sections
- `--ILC` — Scale to ILC machine conditions

Note: `process_histogram.py` is run standalone with command-line arguments, not through the FCCAnalysis framework.

**Configuration:**
- Selection strategies: `Baseline`, `Baseline_miss`, `Baseline_sep`
- Processed histogram: `zll_recoil_m` (Higgs recoil mass computed from $Z\rightarrow\ell^+\ell^-$)

### 2. `combine.py` — COMBINE Datacard Generation

Generates COMBINE-compatible datacards for statistical fitting of the ZH signal and backgrounds.

**Key operations:**
- Defines signal processes: $e^+e^-\rightarrow Z(f\bar{f})H$ with all Higgs decay modes ($b\bar{b}$, $c\bar{c}$, $s\bar{s}$, $gg$, $\mu^+\mu^-$, $\tau^+\tau^-$, $ZZ^*$, $WW^*$, $Z\gamma$, $\gamma\gamma$, $\textrm{Inv}$)
    - Also take into account the $Z$ decays from quarks, $\tau$ and neutrinos
- Defines background processes: $ZZ$, $W^+W^-$, $Z/\gamma$, and rare processes
- Configures systematic uncertainties (1% log-normal normalization uncertainties for backgrounds)
- Prepares datacard structure for RooFit/COMBINE statistical framework
- Supports multiple analysis categories based on Higgs production/decay modes

**Input:** Configuration from command-line or automated workflow
```
cat:  ee or mumu (final state channel)
ecm:  240 or 365 (center-of-mass energy)
sel:  Baseline, Baseline_miss, Baseline_sep (selection strategy)
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
- **Signal:** $e^+e^-\rightarrow Z(f\bar{f})H$ with 10 Higgs decay modes + $ZZ_\textrm{noInv}$ variant
- **Backgrounds:**
  - `ZZ` — Pair production via $e^+e^- \rightarrow ZZ$
  - `WW` — Pair production via $e^+e^- \rightarrow W^+W^-$ (all decay channels: hadronic, electron, muon)
  - `Zgamma` — Drell-Yan and photon production ($e^+e^- \rightarrow e^+e^-$, $\mu^+\mu^-$, $\tau^+\tau^-$)
  - `Rare` — Rare processes ($e^\pm\gamma \rightarrow e^\pm Z$, $\gamma\gamma \rightarrow \ell^+\ell^-$, invisible $Z$ decays)

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
   - Loads configuration from `output/tmp/config_json/{run-name}/4-run.json`
   - Automatically selected when running via `run/4-run.py`

**Execution:**

*Standalone (interactive mode):*
```bash
fccanalysis combine 4-Combine/combine.py
```

*Automated (via run script):*
```bash
python run/4-run.py --run 2
```

The `run/4-run.py` script automates execution across all combinations of channels, center-of-mass energies, and selections, avoiding repeated manual execution. This uses the **FCCAnalysis framework** (part of FCCWorkspace). The xsec analysis is located at `FCCWorkspace/analyses/ZH/xsec/`.

For more details on the run scripts and workflow automation, see [run/README.md](../run/README.md).

For detailed parameter descriptions and path management, see [package/userConfig.py](../package/README.md#userconfig--path-management--global-parameters).

For physics process configuration, see [package/config.py](../package/README.md#configpy--physics-configuration).

For histogram processing utilities, see [package/tools/process.py](../package/tools/README.md).

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
