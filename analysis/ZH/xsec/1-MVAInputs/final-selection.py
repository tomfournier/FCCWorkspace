##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load analysis configuration and predefined histogram config
from package.userConfig import loc, get_params
from package.config import get_process_list
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
processList = get_process_list(cat, ecm, train=True).keys()



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
    cutList['Baseline'] = Baseline_cut_qq(ecm, True, True)   # Baseline selection (hadronic channel)
doTree = False if 'sel0' in cutList else doTree  # Do not write TTree if sel0 is in cutList



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
if cat in ['ee', 'mumu']: histoList = histos_ll
elif cat == 'qq':         histoList = histos_qq
else: raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
