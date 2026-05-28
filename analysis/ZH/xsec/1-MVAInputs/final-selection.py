##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load analysis configuration and predefined histogram config
from package.userConfig import loc, get_params
from sel.final.leptonic import Baseline_cut_ll, histos_ll
from sel.final.hadronic import Baseline_cut_qq, histos_qq

# Load analysis parameters: decay category, CoM energy, luminosity, test flag
cat, ecm, lumi, test = get_params(os.environ.copy(), '1-run.json', is_final=True, qq_allowed=True)



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
    f'p8_ee_WW_ecm{ecm}' if cat=='qq' else f'p8_ee_WW_{cat}_ecm{ecm}',
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' else f'wzp6_ee_{cat}_ecm{ecm}',

    # Rare backgrounds: radiative and diphoton processes
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
]
if (cat=='qq') and (ecm == 365):
    processList.extend(['wzp6_ee_bbH_ecm365', 'wzp6_ee_ccH_ecm365', 'wzp6_ee_ssH_ecm365'])
if cat in ['ee', 'mumu']:
    processList.append(f'wzp6_gaga_{cat}_60_ecm{ecm}')



########################
### DEFINE SELECTION ###
########################

# Selection cuts dictionary for ROOT filtering
# Keys: selection names appearing in output file names and histograms
cutList: dict[str, str] = {}
# cutList['sel0'] = 'return true;'  # No cuts
if cat in ['ee', 'mumu']:
    cutList['Baseline'] = Baseline_cut_ll(ecm)   # Baseline selection (leptonic channel)
elif cat == 'qq':
    cutList['Baseline'] = Baseline_cut_qq(ecm)   # Baseline selection (hadronic channel)
doTree = False if 'sel0' in cutList else doTree  # Do not write TTree if sel0 is in cutList



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
if cat in ['ee', 'mumu']: histoList = histos_ll
elif cat == 'qq':         histoList = histos_qq
else: raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
