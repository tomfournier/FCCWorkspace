##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

# Analysis configuration and paths
from package.userConfig import loc, get_params
from sel.final.leptonic import histos_ll

cat, ecm, lumi = get_params(os.environ.copy(), '1-run.json', is_final=True)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Input directory for pre-selection outputs
inputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)
# inputDir = loc.get('EVENTS_TRAINING', cat, ecm)
# Output directory for final-selection histograms
outputDir = loc.get('HIST_MVA', cat, ecm)

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'
# If procDict is incomplete, can use procDictAdd to add information on the missing samples

# Parallel processing configuration (default nCPUS=4)
nCPUS = 10

# Produce ROOT TTrees in addition to histograms (default is False)
doTree = True
# Scale yields to integrated luminosity
doScale = True
intLumi = lumi * 1e6  # in pb-1



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Process samples for BDT (signal + backgrounds)
processList = [
    # Signal: ZH production
    f'wzp6_ee_{cat}H_ecm{ecm}',

    # Main backgrounds: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_{cat}_ecm{ecm}',
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}' if cat=='ee' else f'wzp6_ee_mumu_ecm{ecm}',

    # Rare backgrounds: radiative and diphton
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]



#######################
### DEFINE CUT LIST ###
#######################

# Define range for CoM dependent variables
p_up = 70 if ecm==240 else (150 if ecm==365 else 240)
p_dw = 20 if ecm==240 else (50 if ecm==365 else 0)

# Define baseline selection cuts
m_cut, p_cut = 'zll_m > 86 && zll_m < 96', f'zll_p > {p_dw} && zll_p < {p_up}'
rec_cut = ' && zll_recoil_m > 100 && zll_recoil_m < 150' if ecm==365 else ''
Baseline_Cut = m_cut + ' && ' + p_cut + rec_cut

# Selection cuts dictionary (key = selection name used in outputs)
cutList = {
    # 'sel0':     'return true;',
    # 'Baseline': Baseline_Cut,
    'test': Baseline_Cut
}



#################################
### DEFINE HISTOGRAM SETTINGS ###
#################################

# Output histogram definitions (name, title, binning)
histoList = histos_ll
