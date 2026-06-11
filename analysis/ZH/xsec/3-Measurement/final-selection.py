##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load configuration and measurement selection functions
from package.userConfig import (
    loc, event, get_params
)
from package.func.bdt import def_bdt, make_high_low          # BDT score binning utilities
from sel.final.leptonic import Baseline_cut_ll, histos_ll    # Histogram definitions (leptonic channel)
from sel.final.hadronic import Baseline_cut_qq, histos_qq    # Histogram definitions (hadronic channel)
from package.config import (
    input_vars_ll,    # BDT input variables for classification (leptonic channel)
    input_vars_qq,    # BDT input variables for classification (hadronic channel)
    get_process_list
)

# Load analysis parameters: decay category, CoM energy, luminosity, test flag
cat, ecm, lumi, test = get_params(os.environ.copy(), '3-run.json', is_final=True)
input_vars = input_vars_ll if cat in ['ee', 'mumu'] else input_vars_qq



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

# Define BDT score from trained model and apply BDT cut
loc_BDT = loc.get('BDT', cat, ecm, 'Baseline')
defineList, bdt_cut = def_bdt(input_vars, loc_BDT)



#######################
### DEFINE CUT LIST ###
#######################

# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {}
cutList['sel0'] = 'return true;'
if cat in ['ee', 'mumu']:
    Baseline = Baseline_cut_ll(ecm)
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
elif cat == 'qq':
    Baseline = Baseline_cut_qq(ecm)
    cutList['Baseline'] = Baseline
else:
    raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')

# List of selections to split into high/low BDT score regions
sels = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']
# Split each selection into high and low BDT score regions
cutList = make_high_low(cutList, bdt_cut, sels)



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

customHists: dict[str, dict[str, str | int | float]] = {}
if cat in ['ee', 'mumu']:
    customHists['leps_iso']    = {'name':'ConeIsolation', 'title':'I_{rel}'}
    customHists['leps_iso_no'] = {'name':'n_leptons',     'title':'Isolated leptons'}
else:
    customHists['best_cluster_idx'] = {'name':'best_cluster_idx', 'title':'Best clustering algorithm'}
    customHists['njets_inclusive']  = {'name':'njets_inclusive',  'title':'Number of jets (inclusive)'}
    customHists['njets_incl']       = {'name':'njets_incl',       'title':'Number of jets (inclusive)'}

# Output histogram definitions (name, title, binning)
histoList = histos_ll if cat in ['ee', 'mumu'] else histos_qq
histoList['BDTscore'] = {'name':'BDTscore',
                         'title':'BDT score',
                         'bin':1000,'xmin':0,'xmax':1}
