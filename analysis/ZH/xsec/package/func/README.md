# `func/` Module: Physics Analysis Utilities

The `func/` module provides core functionality for multivariate analysis (BDT training and evaluation) and statistical bias testing in the ZH cross-section measurement framework.

## Overview

This module is organized into two specialized submodules:

- **`bdt.py`**: Boosted Decision Tree (BDT) utilities for training, evaluation, and ROOT integration
- **`bias.py`**: Pseudo-data generation and statistical analysis tools for bias testing

## Module: `bdt.py`

Comprehensive utilities for XGBoost-based BDT training and evaluation with seamless ROOT/TMVA integration for physics analysis.

### Data Preparation & Loading

- **`counts_and_effs(files, vars, only_eff=False)`**
  - Load events from ROOT files and compute selection efficiency
  - Returns: efficiency only (when `only_eff=True`) or tuple `(dataframe, efficiency, N_events)`
  - Handles lazy loading with uproot and automatic file concatenation

- **`additional_info(df, mode, sig)`**
  - Add metadata columns to dataframe: `'sample'` (process name) and `'isSignal'` (binary classification)
  - Used to mark signal vs background for BDT classification

### Sample Balancing & Weighting

- **`BDT_input_numbers(df, modes, sig, eff, xsec, frac)`**
  - Compute balanced event counts per process for BDT training
  - Accounts for cross-section differences and selection efficiency
  - Returns dictionary: `{process_name: num_events}`

- **`df_split_data(df, N_BDT_inputs, eff, xsec, N_events, mode, lumi=10.8)`**
  - Split samples into 50/50 training/validation sets with luminosity weights
  - Adds columns: `'valid'` (boolean), `'norm_weight'`, and `'weights'`
  - Weights scaled by efficiency, cross-section, and integrated luminosity (pb→ab⁻¹ conversion)

- **`print_stats(df, modes)`**
  - Display training and validation event counts per process
  - Formatted output for debugging sample composition

### Model Training & Evaluation

- **`split_data(df, vars)`**
  - Extract features and labels into numpy arrays
  - Returns: `(X_train, y_train, X_valid, y_valid)` as float32/int8 arrays
  - Training/validation split based on `'valid'` column

- **`train_model(X_train, y_train, X_valid, y_valid, config, early)`**
  - Train XGBoost classifier with early stopping
  - `config`: hyperparameters dict (learning rate, max depth, etc.)
  - `early`: number of rounds for early stopping
  - Multi-metric evaluation: error, logloss, AUC
  - Returns: trained XGBClassifier

- **`evaluate_bdt(df, bdt, vars)`**
  - Compute BDT scores for all events
  - Adds `'BDTscore'` column (probability of signal class, range 0–1)
  - Returns: modified dataframe with scores

- **`get_metrics(bdt)`**
  - Extract training curves and optimal iteration
  - Returns: `(results_dict, num_epochs, epoch_axis, best_epoch)`
  - Used for visualization of training performance

### Model Persistence

- **`save_model(bdt, vars, path)`**
  - Export trained model in two formats:
    - ROOT/TMVA format (`xgb_bdt.root`) for ROOT-based analysis
    - Joblib format (`xgb_bdt.joblib`) for Python
  - Stores feature variable names as ROOT TList in `xgb_bdt.root`
  - BDT cut threshold saved separately as `BDT_cut.txt`

- **`load_model(inDir)`**
  - Load previously trained XGBoost model from `inDir/xgb_bdt.joblib`
  - Returns: XGBClassifier ready for evaluation

### ROOT Integration & Analysis Selections

- **`def_bdt(vars, loc_bdt, MVAVec='MVAVec', score='BDTscore', defineList={}, suffix='')`**
  - Define BDT computation in ROOT RDataFrame with TMVA integration
  - Loads TMVA model from `loc_bdt/xgb_bdt.root`
  - Creates feature vector column and computes BDT scores
  - Loads BDT cut threshold from `loc_bdt/BDT_cut.txt`
  - Returns: `(updated defineList, BDT_cut_value)` for ROOT selection building

- **`make_high_low(cutList, bdt_cut, sels, score='BDTscore')`**
  - Create signal-like and background-like selection regions
  - For each selection in `sels`, generates two variants:
    - `{selection}_high`: BDT score > threshold (signal-enriched)
    - `{selection}_low`: BDT score < threshold (background-enriched)
  - Returns: updated cutList dictionary

### Key Conventions

- **Signal/Background**: Encoded as binary labels (1 = signal, 0 = background)
- **Event Weights**: Scaled by efficiency × cross-section × luminosity
- **BDT Scores**: Probability of signal class (0 to 1)
- **Training Strategy**: 50/50 train/validation split per process after balanced sampling
- **Tree Construction**: XGBoost histogram tree method with early stopping
- **Output Format**: ROOT TMVA + Joblib for maximum ecosystem compatibility

---

## Module: `bias.py`

Tools for generating pseudo-data with controlled signal variations and creating Combine-compatible datacards for statistical fits and bias testing.

### Helper Functions

- **`_signal_lists(cat, z_decays, h_decays, target, ecm=240, tot=True)`**
  - Generate nested lists of signal process names for each Higgs decay channel
  - Naming convention: `wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}`
  - For invisible Higgs decays: replaces 'ZZ' with 'ZZ_noInv' to avoid double-counting
  - Returns: `list[list[str]]` where each sublist corresponds to a Higgs decay

- **`_scaling(sigs, h_decays, target, variation, verbose=True)`**
  - Compute scaling factors for signal channel variations preserving total cross-section
  - `variation`: factor for total signal (e.g., 1.05 = +5%)
  - Scale for target channel computed as: `scale_target = 1 + (variation - 1) × xsec_tot / xsec_target`
  - Returns: `(scale_target, xsec_tot, xsec_tot_new)`

### Pseudo-Data & Datacard Generation

- **`make_pseudodata(hName, inDir, procs, processes, cat, z_decays, h_decays, target, ecm=240, variation=1.05, suffix='', proc_scales=None, tot=True)`**
  - Create pseudo-data histogram combining backgrounds and perturbed signal
  - `hName`: histogram name to retrieve
  - `procs`: list of process names (signal first, then backgrounds)
  - `processes`: dict mapping process names to sample lists
  - `target`: Higgs decay channel to perturb
  - `variation`: total signal scaling factor
  - `proc_scales`: optional per-process scale factors (e.g., for polarization or luminosity corrections)
  - Returns: ROOT histogram (TH1) containing pseudo-data

- **`make_datacard(outDir, procs, target, bkg_unc, categories, freezeBkgs=False, floatBkgs=False, plot_dc=False)`**
  - Generate Combine-compatible statistical analysis datacard
  - Creates text file: `datacard_{target}.txt`
  - `bkg_unc`: background uncertainty (log-normal nuisance parameter)
  - `categories`: list of analysis channels/categories
  - `freezeBkgs`: freeze background normalizations (fixed rates)
  - `floatBkgs`: allow backgrounds to float (negative process indices)
  - Process indices: signal=0, background1=1, background2=2, etc.
  - Returns: None (writes file)

- **`pseudo_datacard(inDir, outDir, cat, ecm, target, pert, z_decays, h_decays, processes, tot=False, scales='', freeze=False, float_bkg=False, plot_dc=False)`**
  - Full pipeline: generate pseudo-data histogram and Combine datacard
  - Creates ROOT file: `datacard_{target}.root` with histograms
  - Creates text datacard: `datacard_{target}.txt`
  - Useful for bias test workflows
  - Returns: None

- **`hist_from_datacard(inDir, target, cat, procs)`**
  - Retrieve histograms from generated datacard
  - Constructs background histogram by summing all background processes
  - Subtracts background from pseudo-data to extract signal
  - Returns: `(signal_hist, pseudo_data_hist)` as `hist.Hist` objects

### Key Conventions

- **Process Naming**: Follows FCC pattern `wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}`
- **Invisible Channel**: 'ZZ' replaced with 'ZZ_noInv' for invisible Higgs decays
- **Cross-Section Preservation**: Total signal unchanged by channel variation
- **Pseudo-Data**: Combines unscaled backgrounds + scaled signal
- **Datacard Format**: Combine-compatible with shape templates and nuisance parameters
- **Process Indices**: Signal=0, backgrounds=1,2,3,... (or negative if floating)

---

## Usage Examples

### BDT Training & Evaluation

```python
from package.func import bdt

# 1. Load data and compute efficiency
df, eff, N_events = bdt.counts_and_effs(
    files=['signal.root', 'background.root'],
    vars=['pt', 'eta', 'energy']
)

# 2. Prepare data with weights
df = bdt.additional_info(df, mode='signal', sig='signal')
balanced_counts = bdt.BDT_input_numbers(df, modes=['signal', 'bkg'], sig='signal', ...)
df_train = bdt.df_split_data(df, N_BDT_inputs=balanced_counts, ...)

# 3. Extract and train
X_train, y_train, X_valid, y_valid = bdt.split_data(df_train, vars=['pt', 'eta', 'energy'])
config = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
bdt_model = bdt.train_model(X_train, y_train, X_valid, y_valid, config=config, early=20)

# 4. Evaluate and save
df_eval = bdt.evaluate_bdt(df, bdt_model, vars=['pt', 'eta', 'energy'])
bdt.save_model(bdt_model, vars=['pt', 'eta', 'energy'], path='./models')

# 5. Use in ROOT
defineList, bdt_cut = bdt.def_bdt('pt, eta, energy', './models')
cutList = bdt.make_high_low(cutList, bdt_cut, ['signal_region'])
```

### Bias Testing & Pseudo-Data

```python
from package.func import bias

# Generate pseudo-data with signal variation
hist_pseudo = bias.make_pseudodata(
    hName='mass_plot',
    inDir='./histograms',
    procs=['ZH', 'WW', 'ZZ'],
    processes={'ZH': ['wzp6_...'], 'WW': ['ww_...'], 'ZZ': ['zz_...']},
    cat='ee',
    z_decays=['ee', 'mumu'],
    h_decays=['bb', 'inv', 'cc'],
    target='inv',
    ecm=240,
    variation=1.05  # +5% signal variation
)

# Create Combine datacard
bias.pseudo_datacard(
    inDir='./histograms',
    outDir='./datacards',
    cat='ee',
    ecm=240,
    target='inv',
    pert=1.05,
    z_decays=['ee', 'mumu'],
    h_decays=['bb', 'inv', 'cc'],
    processes={...}
)
```

---

## Dependencies

- **Core**: numpy, pandas, xgboost, scikit-learn
- **ROOT Integration**: ROOT (with TMVA), uproot
- **Utilities**: joblib, hist (for bias tests)

Note: Heavy dependencies are lazy-loaded to minimize overhead during module import.

---

## Module Author & Context

These utilities are designed for the FCC-ee ZH cross-section analysis workflow, supporting:
- Multivariate analysis with BDT classifiers
- Signal/background discrimination
- Systematic bias testing through pseudo-experiments
- Seamless integration with ROOT TMVA and Combine frameworks
