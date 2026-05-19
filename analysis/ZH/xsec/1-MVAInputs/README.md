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

**Configuration & Usage:**

The script is configured through the `1-run.json` file automatically. Key parameters:

```python
# Channel and energy selection (loaded from 1-run.json)
cat  = 'ee', 'mumu', or 'qq'      # Final state (leptonic or hadronic)
ecm  = 240 or 365                 # Center-of-mass energy (GeV)
test = False or True              # Test mode flag

# Processing configuration
nCPUS    = 20                     # Number of parallel processors
prodTag  = 'FCCee/winter2023_training/IDEA/'
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# HTCondor batch processing (optional)
runBatch  = True/False            # Enable batch submission (from RUN_BATCH env var)
batchQueue = 'espresso'           # HTCondor queue name
compGroup = 'group_u_FCC.local_gen'  # Computing account
```

**Usage:**
```bash
cd xsec/
# Automated pipeline (recommended)
python 0-Run/1-run.py --run 1

# Manual execution with HTCondor batch
RUN_BATCH=1 fccanalysis run 1-MVAInputs/pre-selection.py
```

**Process samples:**
- **Signal:** $e^+e^- \to ZH$ with $Z \to \ell^+\ell^-$ 
- **Backgrounds:**
  - Diboson: $ZZ$, $WW$
  - $Z+\text{jets}$: $Z/\gamma \to \ell^+\ell^-$
  - Rare: radiative Z ($e^\pm\gamma \to e^\pm Z$), diphoton ($\gamma\gamma \to \ell^+\ell^-$)

**Notes:**
- Process list is dynamically loaded using `get_process_list()` based on channel, energy, and batch settings
- Test mode outputs to `output/data/events/{ecm}/{cat}/full/test/` instead of training directory
- Custom C++ analysis functions are included from `functions.h` and `functions_hadronic.h`

---

### 2. `final-selection.py` — MVA Input Histogram Generation

Produces analysis histograms and ROOT TTrees from pre-selected events for BDT training and feature study.

**Key operations:**
- Loads pre-selected event TTree for all processes (signal + backgrounds)
- Applies multiple selection cuts defined in `cutList` dictionary:
  - `sel0`: No cut (baseline acceptance)
  - `Baseline`: Main analysis cuts
  - `test`: Same as Baseline (for testing)
- **Baseline selection cuts** (energy-dependent):
  - Dilepton mass window: $86 < m_{\ell^+\ell^-} < 96\text{ GeV}$
  - Momentum range:
    - 240 GeV: $20 < p_{\ell^+\ell^-} < 70\text{ GeV}$
    - 365 GeV: $50 < p_{\ell^+\ell^-} < 150\text{ GeV}$ + recoil $100 < m_{\text{recoil}} < 150\text{ GeV}$
- Fills histograms for kinematic variables via `histos_ll` configuration
- Scales yields to integrated luminosity:
  - 240 GeV: 10.8 ab⁻¹
  - 365 GeV: 3.12 ab⁻¹
- Outputs ROOT histograms and TTrees for all processes and selection cuts

**Input:** Pre-selected event trees from `pre-selection.py`
```
output/data/events/{ecm}/{cat}/full/training/
```

**Output:** MVA input histograms and TTrees
```
output/data/histograms/MVAInputs/{ecm}/{cat}/{sample}_{sel}_histo.root  # Histogram files (for plotting)
output/data/histograms/MVAInputs/{ecm}/{cat}/{sample}_{sel}.root        # TTree files (for BDT training)
```

**Configuration:**

```python
# Channel and energy selection (from 1-run.json)
cat      = 'ee' or 'mumu'         # Final state
ecm      = 240 or 365             # Center-of-mass energy (GeV)
lumi     = 10.8 or 3.12           # Integrated luminosity (ab⁻¹)
test     = False or True          # Test mode flag

# Processing configuration
nCPUS   = 10                      # Number of parallel processors
doTree  = True                    # Save output event TTrees
doScale = True                    # Scale to integrated luminosity

# Process samples (7 per channel: 1 signal + 6 backgrounds)
processList = [
    f'wzp6_ee_{cat}H_ecm{ecm}',           # Signal: ZH
    f'p8_ee_ZZ_ecm{ecm}',                 # Background: ZZ
    f'p8_ee_WW_{cat}_ecm{ecm}',           # Background: WW
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',    # Background: Z→ee (or Z→μμ)
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',    # Background: Radiative Z (egamma)
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',    # Background: Radiative Z (gammae)
    f'wzp6_gaga_{cat}_60_ecm{ecm}'        # Background: Photon fusion
]
```

**Selection cuts dictionary:**

```python
cutList = {
    'sel0':     'return true;',           # No cut
    'Baseline':  Baseline_Cut,            # Main analysis cuts (energy-dependent)
    'test':      Baseline_Cut             # Test selection
}
```

**Usage:**
```bash
cd xsec/
# Automated pipeline (recommended)
python 0-Run/1-run.py --run 2

# Manual execution
fccanalysis final 1-MVAInputs/final-selection.py
```

**Notes:**
- Must be run after `pre-selection.py` to have input TTrees available
- Histogram definitions are imported from `sel.final.leptonic.histos_ll`
- Energy-dependent momentum cuts ensure optimal signal/background separation at each energy
- Recoil mass cut only applied at 365 GeV (where Higgs mass is better resolved)

---

### 3. `plots.py` — Validation Plots

Generates plots of kinematic distributions for signal and backgrounds to validate event selection and BDT training inputs.

**Key operations:**
- Loads MVA input histograms from all processes via histogram files (`*_histo.root`)
- Configures plots by analysis type (`ZH`) and selections (`sel0`, `Baseline`, `test`)
- Stacks background histograms and overlays signal for comparison
- Produces linear and logarithmic scale plots for each kinematic variable
- Applies consistent color scheme per process and ROOT TLatex labels
- Outputs plots in format(s) specified by `plot_file` configuration

**Input:** MVA input histograms from `final-selection.py`
```
output/data/histograms/MVAInputs/{ecm}/{cat}/{sample}_{sel}_histo.root
```

**Output:** Validation plots
```
output/plots/MVAInputs/{ecm}/{cat}/{sel}/
```

**Configuration:**

```python
# Channel and energy selection (from 1-run.json)
cat      = 'ee' or 'mumu'          # Final state
ecm      = 240 or 365              # Center-of-mass energy (GeV)
lumi     = 10.8 or 3.12            # Integrated luminosity (ab⁻¹)

# Plot display settings
intLumi  = lumi * 1e6              # Integrated luminosity in pb⁻¹
intLumiLabel = f'L = {lumi} ab⁻¹'  # Label for plots
ana_tex  = Process string for plots (e.g., 'e⁺e⁻ → ZH → e⁺e⁻ + X')
customLabel  = 'Training sample'

# Plot output configuration
yaxis    = ['lin', 'log']          # Linear and logarithmic scales
stacksig = ['nostack']             # Don't stack signal with background
formats  = plot_file               # Output formats (PNG, PDF, etc.)
setGrid  = True                    # Draw grid on plots
strictRange = True                 # Use strict axis ranges

# Scaling factors
scaleSig = 1.0                     # Signal scale factor (for visibility)
scaleBkg = 1.0                     # Background scale factor

# Selection cuts to plot
selections['ZH'] = ['sel0', 'Baseline', 'test']

# Per-selection plot labels
extralabel['sel0']     = 'No cut'
extralabel['Baseline'] = 'Baseline'
extralabel['test']     = 'test'
```

**Process configuration:**

```python
# Signal and background samples with ROOT process labels
plots['ZH'] = {
    'signal': {
        'eeH' or 'mumuH': [sample_name]
    },
    'backgrounds': {
        'WWee/mumu': [sample_name],
        'ZZ': [sample_name],
        'Zee/mumu': [sample_name],
        'eeZ': [sample_names...],     # Radiative Z backgrounds
        'gagaee/mumu': [sample_name]  # Photon fusion
    }
}

# ROOT color scheme
colors[process] = ROOT.kColor      # Color for each process type
legend[process] = 'TLatex string'  # ROOT legend label with TLatex formatting
```

**Plotted variables:**

The script plots all kinematic variables computed in `pre-selection.py`:
- **Leptons:** `leading_p`, `leading_pT`, `leading_theta`, `subleading_p`, `subleading_pT`, `subleading_theta`
- **Z boson:** `zll_m`, `zll_p`, `zll_pT`, `zll_theta`, `zll_costheta`, `zll_category`
- **Topology:** `acolinearity`, `acoplanarity`, `acopolarity`, `deltaR`
- **Recoil:** `zll_recoil_m`, `zll_recoil_p`
- **Energy:** `e_long`, `e_trans`, `e_tan`, `visibleEnergy`, `visibleEnergy_tot`, `cosTheta_miss`, `missingMass`, `missingEnergy`
- **Discriminant:** `H` (Higgsstrahlungness)

**Usage:**
```bash
cd xsec/
# Automated pipeline (recommended)
python 0-Run/1-run.py --run 3

# Manual execution
fccanalysis plots 1-MVAInputs/plots.py
```

**Notes:**
- Must be run after `final-selection.py` to have input histograms available
- Variables are automatically sorted alphabetically in the output
- Supports multiple selections per analysis for comprehensive validation
- Linear and logarithmic scales help identify both small and large contributions
- For each energy, appropriately scaled visualization is automatically applied

---

## Configuration & Customization

All scripts are configured automatically through the `1-run.json` file, which is managed by the pipeline runner and populated with parameters from the main analysis configuration in the [package](../package/README.md) module:

- **`package.userConfig`**: File paths, luminosity scaling, channel/energy parameters
- **`package.config`**: Process definitions, variable labels, color schemes

### Configuration Parameters

**Channel and Energy Selection:**
```python
cat  = 'ee', 'mumu', or 'qq'       # Final state (leptonic or hadronic)
ecm  = 240 or 365                  # Center-of-mass energy (GeV)
test = False or True               # Test mode (outputs to /test/ subdirectory)
```

**`pre-selection.py` parameters:**
```python
nCPUS       = 20                   # Number of parallel processors
prodTag     = 'FCCee/winter2023_training/IDEA/'
procDict    = 'FCCee_procDict_winter2023_training_IDEA.json'
runBatch    = True/False           # Enable HTCondor batch submission
batchQueue  = 'espresso'           # HTCondor queue (espresso, longlunch, workday)
compGroup   = 'group_u_FCC.local_gen'  # Computing account
```

**`final-selection.py` parameters:**
```python
nCPUS       = 10                   # Number of parallel processors
doTree      = True                 # Save ROOT TTrees for BDT training
doScale     = True                 # Scale histograms to integrated luminosity
intLumi     = lumi * 1e6           # Luminosity in pb⁻¹
```

**`plots.py` parameters:**
```python
yaxis       = ['lin', 'log']       # Produce both linear and log scale plots
stacksig    = ['nostack']          # Don't stack signal with backgrounds
formats     = plot_file            # Output image formats
customLabel = 'Training sample'    # Label to add to all plots
```

### Adding Custom Selections

To add a new selection cut, edit the `cutList` in `final-selection.py`:

```python
cutList = {
    'sel0':      'return true;',
    'Baseline':  Baseline_Cut,
    'custom':    'your_custom_cut_expression'  # Add here
}
```

Then add corresponding plot configuration in `plots.py`:

```python
selections['ZH'].append('custom')
extralabel['custom'] = 'Custom selection label'
```

### Batch Processing

To submit the pre-selection step to HTCondor:

```bash
cd xsec/
python 0-Run/1-run.py --run 1 --batch
```

The `batchQueue` and `compGroup` can be adjusted in `pre-selection.py` for different resource requirements.

---

## Workflow Integration

Execute the complete MVA input pipeline through the automated runner, which processes all three scripts in the correct sequence:

```bash
cd xsec/

# Run entire pipeline (all channels and energies)
python 0-Run/1-run.py

# Or run individual stages
python 0-Run/1-run.py --run 1  # Pre-selection only
python 0-Run/1-run.py --run 2  # Final-selection only  
python 0-Run/1-run.py --run 3  # Plots only

# With HTCondor batch for heavy processing
python 0-Run/1-run.py --run 1 --batch
```

The runner automatically:
1. Loads configuration from `1-run.json` 
2. Processes both channels (`ee`, `mumu`) and both energies (240, 365 GeV)
3. Generates all MVA input histograms needed for downstream BDT training
4. Manages dependencies between stages (plot generation requires histograms, etc.)

### Manual Execution

For development or debugging, scripts can also be run individually:

```bash
# Pre-selection
fccanalysis run 1-MVAInputs/pre-selection.py

# Final-selection (after pre-selection is complete)
fccanalysis final 1-MVAInputs/final-selection.py

# Plots (after final-selection is complete)
fccanalysis plots 1-MVAInputs/plots.py
```

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
