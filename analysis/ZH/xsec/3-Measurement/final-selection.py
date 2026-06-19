##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load configuration and measurement selection functions
from package.userConfig import (
    loc, event, get_params
)
from package.func.bdt import def_bdt, make_high_low  # BDT score binning utilities
from sel.final.leptonic import (
    Baseline_cut_ll,  # Baseline cut definition      (leptonic channel)
    histos_ll,        # Histogram definitions        (leptonic channel)
    custom_hists_ll   # Custom histogram definitions (leptonic channel)
)
from sel.final.hadronic import (
    Baseline_cut_qq,  # Baseline cut definition      (hadronic channel)
    histos_qq,        # Histogram definitions        (hadronic channel)
    custom_hists_qq   # Custom histogram definitions (hadronic channel)
)
from package.config import get_process_list

# Load analysis parameters: decay category, CoM energy, luminosity, test flag
cat, ecm, lumi, test = get_params(os.environ.copy(), '3-run.json', is_final=True)



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
if test:
    loc_BDT = loc.get('BDT', cat, ecm, 'test')
else:
    loc_BDT = loc.get('BDT', cat, ecm, 'Baseline')
defineList, bdt_cut = def_bdt(loc_BDT)



#######################
### DEFINE CUT LIST ###
#######################

# Selection cut dictionary (key = selection name used in outputs)
cutList: dict[str, str] = {}
if not test: cutList['sel0'] = 'return true;'
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
    if test:
        cutList['test'] = Baseline
    else:
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

# Custom histogram made at the pre-selection
customHists = custom_hists_ll if cat in ['ee', 'mumu'] else custom_hists_qq

# Output histogram definitions (name, title, binning)
histoList = histos_ll if cat in ['ee', 'mumu'] else histos_qq
histoList['BDTscore'] = {'name':'BDTscore',
                         'title':'BDT score',
                         'bin':1000,'xmin':0,'xmax':1}
if cat == 'qq':
    if ecm == 240:
        histoList['zqq_m_recoil_m_mva'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                           'bins':[(50, 100, 150), (100, 40, 140), (100, 0, 1)]}
        histoList['zqq_m_recoil_m_mva_high'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                'bins':[(50, 100, 150), (100, 40, 140), (1, bdt_cut, 1)]}
        histoList['zqq_m_recoil_m_mva_low'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                               'bins':[(50, 100, 150), (100, 40, 140), (1, 0, bdt_cut)]}
        histoList['zqq_m_recoil_m_mva_jan_high'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                    'bins':[(50, 100, 150), (100, 40, 140), (1, 0.75, 1)]}
        histoList['zqq_m_recoil_m_mva_jan_low'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                   'bins':[(50, 100, 150), (100, 40, 140), (1, 0, 0.75)]}
    elif ecm == 365:
        histoList['zqq_m_recoil_m_mva'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                           'bins':[(100, 100, 200), (100, 40, 140), (100, 0, 1)]}
        histoList['zqq_m_recoil_m_mva_high'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                'bins':[(100, 100, 200), (100, 40, 140), (1, bdt_cut, 1)]}
        histoList['zqq_m_recoil_m_mva_low'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                               'bins':[(100, 100, 200), (100, 40, 140), (1, 0, bdt_cut)]}
        histoList['zqq_m_recoil_m_mva_jan_high'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                    'bins':[(100, 100, 200), (100, 40, 140), (1, 0.95, 1)]}
        histoList['zqq_m_recoil_m_mva_jan_low'] = {'cols':['zqq_recoil_m', 'zqq_m', 'BDTscore'],
                                                   'bins':[(100, 100, 200), (100, 40, 140), (1, 0, 0.95)]}
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')
