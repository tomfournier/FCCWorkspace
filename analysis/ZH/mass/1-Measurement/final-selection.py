##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load configuration and measurement selection functions
from package.userConfig import (
    loc, event, get_params
)
from sel.final import Baseline_cut, histo_list    # Histogram definitions
from package.config import get_process_list

# Load analysis parameters: decay category, CoM energy, luminosity, test flag
cat, ecm, lumi, test = get_params(os.environ.copy(), '1-run.json', is_final=True)



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

# Input: Preprocessed ROOT trees and events from pre-selection
if test: inputDir  = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    inputDir  = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Output: Directory for measurement histograms (used by measurement/fit stages)
outputDir = loc.get('HIST_PREPROCESSED', cat, ecm)

# Process dictionary with cross-section and sample metadata
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# Parallel processing configuration
nCPUS = 10  # Number of CPUs for parallel histogram filling

# ROOT output options
doTree  = False       # Optionally save ROOT TTrees (default is False)
doScale = True        # Scale histograms to integrated luminosity
intLumi = lumi * 1e6  # Integrated luminosity in pb^-1

# Optional outputs (commented out by default)
# saveJSON = True    # Export results to JSON format
# saveTabular = True # Generate LaTeX tables



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Samples processed during pre-selection
samples = get_process_list(cat, ecm).keys()

# Load event samples with events TTree
processList = event(samples, inputDir)



#######################
### DEFINE CUT LIST ###
#######################

# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {}
cutList['sel0'] = 'return true;'
if cat in ['ee', 'mumu']:
    Baseline = Baseline_cut(ecm)
    cutList['Baseline']      = Baseline
    cutList['Baseline_miss'] = Baseline + ' && cosTheta_miss < 0.98'
    if ecm == 240:
        cutList['Baseline_vis'] = Baseline + ' && visibleEnergy > 100'
        cutList['Baseline_inv'] = Baseline + ' && visibleEnergy < 100'
        cutList['Baseline_sep'] = Baseline + ' && ((visibleEnergy > 100) || (visibleEnergy < 100 && cosTheta_miss < 0.99))'
    elif ecm == 365:
        cutList['Baseline_vis'] = Baseline + ' && visibleEnergy > 171'
        cutList['Baseline_inv'] = Baseline + ' && visibleEnergy < 171'
        cutList['Baseline_sep'] = Baseline + ' && ((visibleEnergy > 171) || (visibleEnergy < 171 && cosTheta_miss < 0.99))'
else:
    raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

customHists: dict[str, dict[str, str | int | float]] = {}
if cat in ['ee', 'mumu']:
    customHists['leps_iso']    = {'name':'ConeIsolation', 'title':'I_{rel}'}
    customHists['leps_iso_no'] = {'name':'n_leptons', 'title':'Isolated leptons'}

histoList = histo_list
