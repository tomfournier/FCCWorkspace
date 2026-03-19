###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import os

# Import user configuration paths and parameters
from package.config import z_decays, H_decays
from package.sel.presel.leptonic import presel_ll
from package.userConfig import (
    loc, get_params
)

cat, ecm = get_params(os.environ.copy(), '3-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = loc.get('EVENTS_TEST', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# Optional: Number of CPUs for parallel processing
# (default is 4,  -1 uses all cores available)
nCPUS = 20

# Run on HTCondor batch system (default is False)
runBatch = False

# Batch queue name for HTCondor (default is workday)
batchQueue = 'longlunch'

# Computing account for HTCondor (default is group_u_FCC.local_gen)
compGroup = 'group_u_FCC.local_gen'



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

# Combine all samples (override with WW only if ww flag is set)
samples = samples_sig + samples_bkg

# Large samples requiring chunked processing
big_sample = [
    f'p8_ee_WW_ecm{ecm}',
    'wzp6_ee_mumu_ecm240' if cat=='mumu' else 'wzp6_ee_ee_Mee_30_150_ecm240'
]
middle_sample = [
    f'p8_ee_WW_{cat}_ecm{ecm}',
    f'p8_ee_ZZ_ecm{ecm}',
    'wzp6_ee_mumu_ecm365' if cat=='mumu' else 'wzp6_ee_ee_Mee_30_150_ecm365',
    f'wzp6_egamma_eZ_Z{cat}_ecm240',
    f'wzp6_gammae_eZ_Z{cat}_ecm240',
    f'wzp6_gaga_{cat}_60_ecm240'
]
# Configure processing fraction and chunks for each sample
processList = {i:{'fraction': 1, 'chunks': 10 if i in big_sample else
                  (5 if i in middle_sample else 1)} for i in samples}



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFgraph():
    '''RDataFrame analysis class for pre-selection stage.'''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df, dataset):
        '''Apply analysis graph construction to the dataframe.'''
        df, params = presel_ll(df, cat, ecm, dataset)
        return df, params

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        '''Define output branches to save.'''
        branchList = sorted([
            # Lepton kinematics (leading and subleading)
            'leading_p',    'leading_pT',    'leading_theta',    'leading_phi',
            'subleading_p', 'subleading_pT', 'subleading_theta', 'subleading_phi',

            # Angular correlation
            'acolinearity', 'acopolarity', 'acoplanarity', 'deltaR',

            # Z boson kinematics
            'zll_m', 'zll_p', 'zll_pT', 'zll_theta', 'zll_costheta', 'zll_phi',

            # Recoil mass (Higgs candidate)
            'zll_recoil_m',

            # Missing energy variables
            'visibleEnergy', 'cosTheta_miss', 'missingMass', 'missingEnergy',

            # Higgsstrahlungness discriminant
            'H'
        ])
        return branchList
