##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Load configuration and measurement selection functions
from package.userConfig import (
    loc, event, get_params
)
from package.func.bdt import def_bdt, make_high_low  # BDT score binning utilities
from sel.final.leptonic import histos_ll             # Histogram definitions
from package.config import (
    z_decays,      # Z boson decay modes
    H_decays,      # Higgs boson decay modes
    input_vars     # BDT input variables for classification
)

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

# Background samples:
samples_bkg = [
    # Diboson:  ee -> VV
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_ecm{ecm}',
    f'p8_ee_WW_ee_ecm{ecm}',
    f'p8_ee_WW_mumu_ecm{ecm}',

    # ee -> Z+jets
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}',
    f'wzp6_ee_tautau_ecm{ecm}',

    # Radiative: ey -> eZ(ll)
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',

    # Diphoton: yy -> ll
    f'wzp6_gaga_ee_60_ecm{ecm}',
    f'wzp6_gaga_mumu_60_ecm{ecm}',
    f'wzp6_gaga_tautau_60_ecm{ecm}',

    # Invisible: ee -> nunuZ
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]

# Signal samples: ee -> Z(ll)H with all Higgs decay modes
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays + ('ZZ_noInv',)]

# Load event samples with events TTree
processList = event(samples_sig + samples_bkg, inputDir)

# Define BDT score from trained model and apply BDT cut
sel_BDT = 'Baseline'
loc_BDT = loc.get('BDT', cat, ecm, sel_BDT)
defineList, bdt_cut = def_bdt(input_vars, loc_BDT)



#######################
### DEFINE CUT LIST ###
#######################

# Define range for CoM dependent variables
p_up = 70 if ecm==240 else (150 if ecm==365 else 200)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)

# Define baseline selection cuts
vis_cut = 100 if ecm==240 else (171 if ecm==365 else 0)
m_cut, p_cut = 'zll_m > 86 && zll_m < 96', f'zll_p > {p_dw} && zll_p < {p_up}',
rec_cut = ' && zll_recoil_m > 100 && zll_recoil_m < 150' if ecm==365 else ''

Baseline_Cut = m_cut + ' && ' + p_cut + rec_cut
vis, inv = Baseline_Cut + f' && visibleEnergy > {vis_cut}', Baseline_Cut + f' && visibleEnergy < {vis_cut}'

# Selection cut dictionary (key = selection name used in outputs)
cutList = {
    'Baseline':          Baseline_Cut,
    # 'Baseline_vis':      vis,
    # 'Baseline_inv':      inv,
    # 'Baseline_miss':     Baseline_Cut + ' && cosTheta_miss < 0.98',
    # 'Baseline_sep':      '(('+vis+') || ('+inv+' && cosTheta_miss < 0.99))',
    # 'test':              Baseline_Cut,
    # 'test1':             Baseline_Cut
}

# List of selections to split into high/low BDT score regions
sels = [
    'Baseline', 'Baseline_miss', 'Baseline_sep', 'test', 'test1'
]
# Split each selection into high and low BDT score regions
cutList = make_high_low(cutList, bdt_cut, sels)



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

customHists: dict[str, dict[str, str | int | float]] = {
    'leps_iso':     {'name':'ConeIsolation', 'title':'I_{rel}'},
    'leps_iso_no':  {'name':'n_leptons', 'title':'N_{leptons}'}
}

# Output histogram definitions (name, title, binning)
histoList: dict[str, dict[str, str | int | float]] = histos_ll
histoList['BDTscore'] = {'name':'BDTscore',
                         'title':'BDT score',
                         'bin':1000,'xmin':0,'xmax':1}
