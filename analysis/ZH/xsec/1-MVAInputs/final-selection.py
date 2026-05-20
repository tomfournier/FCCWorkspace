##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load analysis configuration and predefined histogram config
from package.userConfig import loc, get_params
from sel.final.leptonic import histos_ll

# Load analysis parameters: decay category, CoM energy, luminosity, test flag
cat, ecm, lumi, test = get_params(os.environ.copy(), '1-run.json', is_final=True)



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

# Input: Pre-selection ROOT trees and histograms
if test: inputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)  # Test subset
else:    inputDir = loc.get('EVENTS_TRAINING',   cat, ecm)  # Full training sample

# Output: Directory for MVA input histograms
outputDir = loc.get('HIST_MVA', cat, ecm)

# Process dictionary with cross-section and sample metadata
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# Parallel processing configuration (default 4)
nCPUS = 10  # Number of CPUs for parallel histogram filling

# ROOT output options
doTree  = True   # Save ROOT TTrees in addition to histograms (for validation/debugging)
doScale = True   # Scale histograms to integrated luminosity
intLumi = lumi * 1e6  # Integrated luminosity in pb^-1



##########################
### DEFINE SAMPLE LIST ###
##########################

# Samples to process: ZH signal and main background processes
# These are processed through final selection cuts and histogram filling
processList = [
    # Signal: ZH production
    f'wzp6_ee_{cat}H_ecm{ecm}',

    # Main backgrounds: diboson and Drell-Yan
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_{cat}_ecm{ecm}',
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' else f'wzp6_ee_mumu_ecm{ecm}',

    # Rare backgrounds: radiative and diphoton processes
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]



########################
### DEFINE SELECTION ###
########################

# CoM-dependent kinematic bounds
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)  # Upper momentum cut [GeV]
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)    # Lower momentum cut [GeV]

# Baseline selection cuts (dilepton mass and momentum requirements)
m_cut = 'zll_m > 86 && zll_m < 96'           # Z boson mass window [GeV]
p_cut = f'zll_p > {p_dw} && zll_p < {p_up}'  # Dilepton momentum window
rec_cut = ' && zll_recoil_m > 100 && zll_recoil_m < 150' if ecm==365 else ''  # Recoil mass (365 GeV only)
Baseline_Cut = m_cut + ' && ' + p_cut + rec_cut

# Selection cuts dictionary for ROOT filtering
# Keys: selection names appearing in output file names and histograms
cutList = {
    # 'sel0':     'return true;',        # No cuts (diagnostics)
    'Baseline':  Baseline_Cut,         # Baseline selection
    'test':      Baseline_Cut,         # Test selection (same cuts as baseline)
    'test1':     Baseline_Cut
}
doTree = False if 'sel0' in cutList else doTree  # Do not write TTree if sel0 is in cutList



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
histoList = histos_ll
