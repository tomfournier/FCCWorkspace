# 1-MVAInputs: MVA Input Generation

Event selection and histogram generation for Boosted Decision Tree (BDT) training in the FCC-ee ZH cross-section analysis.

## Overview

This module prepares kinematic features and histograms for BDT training to discriminate ZH signal from background processes. The workflow consists of three scripts executed sequentially:

| Script | Purpose |
|--------|---------|
| **`pre-selection.py`** | Apply initial kinematic cuts and compute kinematic variables |
| **`final-selection.py`** | Generate MVA input histograms for all processes |
| **`plots.py`** | Visualize kinematic distributions for validation |

## Workflow

### 1. `pre-selection.py` — Event Pre-Selection & Variable Computation

Applies initial kinematic cuts to raw FCC-ee simulation and computes kinematic variables needed for BDT training.

**Key operations:**
- Selects dilepton events ($e^+e^- \to Z(\ell^+\ell^-)+X$) from raw EDM4Hep events
- Reconstructs $Z \to \ell^+\ell^-$ candidates using mass ($m_Z \approx 91.2$ GeV) and recoil mass ($m_{\text{recoil}} \approx 125$ GeV) constraints
- Applies lepton identification: momentum cut ($p > 20$ GeV), isolation ($I_{\text{rel}} < 0.25$)
- Vetoes $H\to\ell^+\ell^-$ candidates to avoid bias in final selection
- Computes kinematic variables for each event:
  - **Lepton kinematics:** momentum, transverse momentum, polar angle for leading/subleading leptons
  - **Dilepton system:** mass, momentum, transverse momentum, polar angle
  - **Topology:** acolinearity, acoplanarity, $\Delta R$ between leptons
  - **Recoil:** Higgs candidate recoil mass
  - **Other:** visible energy, missing energy, Higgsstrahlungness discriminant

**Input:** Raw EDM4Hep events from FCC centralized productions
```
FCCee/winter2023_training/IDEA/ (via procDict)
```

**Output:** Pre-selected event trees with computed variables
```
output/data/events/{ecm}/{cat}/full/training/
```

**Usage:**
```bash
cd xsec/
fccanalysis run 1-MVAInputs/pre-selection.py
```

Or automate through the pipeline runner:
```bash
python run/1-run.py --run 1
```

**Configuration:**
- Channels: `ee` (electron) or `mumu` (muon) final state
- Center-of-mass energies: 240 GeV or 365 GeV
- Parallel processing: 20 CPUs (adjustable via `nCPUS` parameter)
- Process dictionary: `FCCee_procDict_winter2023_training_IDEA.json`

**Samples processed:**
- **Signal:** $e^+e^- \to ZH$ with $Z \to \ell^+\ell^-$ ($\ell=e,\mu$ depending on `cat`)
- **Backgrounds:**
  - Diboson: $ZZ$, $WW$
  - $Z+\text{jets}$: $Z/\gamma \to \ell^+\ell^-$
  - Rare: radiative Z ($e^\pm\gamma \to e^\pm Z$), diphoton ($\gamma\gamma \to \ell^+\ell^-$)

---

### 2. `final-selection.py` — MVA Input Histogram Generation

Produces analysis histograms and TTree from pre-selected events for BDT training and feature study.

**Key operations:**
- Loads pre-selected event TTree for all processes (signal + backgrounds)
- Applies baseline kinematic selection cuts:
  - Dilepton mass window: $86 < m_{\ell^+\ell^-} < 96\text{ GeV}$
  - Momentum range (energy-dependent):
    - 240 GeV: $20 < p_{\ell^+\ell^-} < 70\text{ GeV}$
    - 365 GeV: $50 < p_{\ell^+\ell^-} < 150\text{ GeV}$ (+ recoil window $100 < m_{\text{recoil}} < 150\text{ GeV}$)
- Fills histograms for 19 kinematic variables
- Scales yields to integrated luminosity:
  - 240 GeV: 10.8 ab⁻¹
  - 365 GeV: 3.12 ab⁻¹
- Outputs ROOT histograms for all processes and selection cuts

**Input:** Pre-selected event trees from `pre-selection.py`
```
output/data/events/{ecm}/{cat}/full/training/
```

**Output:** MVA input histograms
```
output/data/histograms/MVAInputs/{ecm}/{cat}/{sample}_{sel}_histo.root  # For histogram files (for plotting)
output/data/histograms/MVAInputs/{ecm}/{cat}/{sample}_{sel}.root        # For TTree files (for BDT training)
```

**Usage:**
```bash
cd xsec/
fccanalysis final 1-MVAInputs/final-selection.py
```

Or automate:
```bash
python run/1-run.py --run 2
```

Note: Must be run after `pre-selection.py` to have input TTree available.

**Configuration:**
- Output formats: ROOT TTrees and histograms (`doTree = True` to make TTree files)
- Process samples: 7 processes per channel (1 signal + 6 backgrounds)
- Integration: Luminosity scaling enabled if `doScale = True`

**Key variables in histograms:**
- Lepton kinematics (8 variables): $p$, $p_T$, $\theta$ for leading/subleading leptons
- Dilepton system (5 variables): $m_{\ell^+\ell^-}$, $p_{\ell^+\ell^-}$, $p_{T,\ell^+\ell^-}$, $\theta_{\ell^+\ell^-}$, $\phi_{\ell^+\ell^-}$
- Topology (3 variables): acolinearity, acoplanarity, $\Delta R$
- Other (3 variables): recoil mass, visible energy, Higgsstrahlungness

For detailed variable definitions, see [package/config.py](../package/README.md).

---

### 3. `plots.py` — Validation Plots

Generates plots of kinematic distributions for signal and backgrounds to validate event selection.

**Key operations:**
- Loads MVA input histograms from all processes
- Stacks background histograms and overlays signal
- Produces linear and logarithmic scale plots for each variable
- Outputs PNG format for visualization (can choose other formats e.g. PDF)

**Input:** MVA input histograms from `final-selection.py`
```
output/data/histograms/MVAInputs/{ecm}/{cat}/
```

**Output:** Validation plots
```
output/plots/MVAInputs/{ecm}/{cat}/{sel}/
```

**Usage:**
```bash
fccanalysis plots 1-MVAInputs/plots.py
```

Note: Must be run after `final-selection.py` to have input histograms available.

---

## Configuration & Customization

All scripts import analysis configuration from the [package](../package/README.md) module:

- **`package.userConfig`**: File paths, luminosity, channel/energy parameters
- **`package.config`**: Process definitions, variable labels, color schemes

Key parameters in scripts:

```python
# Channel and energy selection
cat  = 'ee' or 'mumu'          # Final state
ecm  = 240 or 365              # Center-of-mass energy (GeV)
lumi = 10.8 or 3.12            # Integrated luminosity (ab⁻¹)

# Processing configuration
nCPUS    = 10                  # Number of parallel processors
frac, nb = 1, 10               # Data fraction and number of chunks
doTree   = True                # Save output event trees
doScale  = True                # Scale to integrated luminosity
```

For advanced configuration (batch processing, custom cuts, HTCondor submission), see individual script headers.

---

## Workflow Integration

Execute all three scripts in sequence through the automated pipeline:

```bash
python run/1-run.py
```

This runner processes both channels (`ee`, `mumu`) and both energies (240, 365 GeV), generating all MVA input histograms needed for the downstream BDT training stage ([2-BDT](../2-BDT/README.md)).

---

## Output Structure

```
output/data/
├── events/{ecm}/{cat}/full/training/          # Pre-selected event trees (ROOT files)
├── histograms/
│   ├── MVAInputs/{ecm}/{cat}/                 # MVA input histograms (ROOT files)
│   └── ...                                    # Other histograms
└── plots/MVAInputs/{ecm}/{cat}/{sel}/         # Validation plots (PNG/PDF)
```

---

## Next Steps

After completing the MVA input stage:

1. **BDT Training** → [2-BDT](../2-BDT/README.md): Train machine learning models using generated histograms
2. **Physics Measurement** → [3-Measurement](../3-Measurement/README.md): Apply BDT to measurement data
3. **Statistical Analysis** → [4-Combine](../4-Combine/README.md) and [5-Fit](../5-Fit/README.md): Extract cross-section results and do bias test

See [package/README.md](../package/README.md) for details on analysis configuration and utility functions.
