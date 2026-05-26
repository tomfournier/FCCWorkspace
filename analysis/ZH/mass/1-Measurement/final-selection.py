##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Analysis configuration and paths
from package.userConfig import (
    loc, event, get_params,
)
from sel.final.leptonic import histos_ll

cat, ecm, lumi = get_params(os.environ.copy(), '3-run.json', is_final=True)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Input directory for pre-selection outputs
inputDir  = loc.get('EVENTS_TEST', cat, ecm)
# inputDir  = loc.get('EVENTS', cat, ecm)

# Output directory for final-selection histograms
outputDir = loc.get('HIST_PREPROCESSED', cat, ecm)

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# If procDict is incomplete, can use procDictAdd to add information on the missing samples

# Parallel processing configuration (default nCPUS=4)
nCPUS = 10

# Produces ROOT TTrees, default is False
# doTree = True

# Scale yields to integrated luminosity
doScale = True
intLumi = lumi * 1e6  # in pb-1

# Save results in a .json file
# saveJSON = True

# Save result in LaTeX tables
# saveTabular = True



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Background samples:
processList = event([
    # Signal: ee -> ZH
    f'wzp6_ee_{cat}H_ecm{ecm}',

    # Diboson:  ee -> VV
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_ecm{ecm}',
    f'p8_ee_WW_{cat}_ecm{ecm}',

    # ee -> Z+jets
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}' if cat=='mumu'
    else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_ee_tautau_ecm{ecm}',

    # Radiative: ey -> eZ(ll)
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',

    # Diphoton: yy -> ll
    f'wzp6_gaga_{cat}_60_ecm{ecm}',
    f'wzp6_gaga_tautau_60_ecm{ecm}',

    # Invisible: ee -> nunuZ
    f'wzp6_ee_nuenueZ_ecm{ecm}'
], inputDir)



#######################
### DEFINE CUT LIST ###
#######################

if ecm == 240:
    Baseline_cut = 'zll_m > 86 && zll_m < 96 && zll_p > 20 && zll_p < 70  && zll_recoil_m > 100 && zll_recoil_m < 150 && cosTheta_miss < 0.98'
elif ecm == 365:
    Baseline_cut = 'zll_m > 86 && zll_m < 96 && zll_p > 50 && zll_p < 150 && zll_recoil_m > 100 && zll_recoil_m < 150 && cosTheta_miss < 0.98'

# Selection cut dictionary (key = selection name used in outputs)
cutList = {
    'Baseline':          Baseline_cut,
    # 'test':              Baseline_cut
}



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

customHists: dict[str, dict[str, str | int | float]] = {
    'leps_iso': {'name':'ConeIsolation', 'title':'I_{rel}'},
    'leps_no':  {'name':'n_leptons', 'title':'N_{leptons}'}
}

# Output histogram definitions (name, title, binning)
histoList: dict[str, dict[str, str | int | float]] = histos_ll
