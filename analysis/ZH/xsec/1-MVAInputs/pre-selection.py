##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

from package.userConfig import loc, get_params
from sel.presel.leptonic import training_ll

# Load config from temporary JSON if running automated, else prompt
cat, ecm = get_params(os.environ.copy(), '1-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)
# outputDir = loc.get('EVENTS_TRAINING', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023_training/IDEA/'
# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# Optional: Number of CPUs for parallel processing
# (default is 4,  -1 uses all cores available)
nCPUS = 20

# Run on HTCondor batch system (default is False)
# runBatch = True
# Batch queue name for HTCondor (default is workday)
batchQueue = 'longlunch'
# Computing account for HTCondor (default is group_u_FCC.local_gen)
compGroup = 'group_u_FCC.local_gen'



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Process samples for BDT training (electron channel)
samples_ee = [
    # Signal: ZH production with H->ee
    f'wzp6_ee_eeH_ecm{ecm}',

    # Main backgrounds: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_ee_ecm{ecm}',
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',

    # Rare backgrounds: radiative and diphoton
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f'wzp6_gaga_ee_60_ecm{ecm}'
]

# Process samples for BDT training (muon channel)
samples_mumu = [
    # Signal: ZH production with H->mumu
    f'wzp6_ee_mumuH_ecm{ecm}',

    # Background: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_mumu_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}',

    # Rare backgrounds: radiative and diphoton
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gaga_mumu_60_ecm{ecm}'
]

# Select samples based on final state
if   cat=='ee':   samples_BDT = samples_ee
elif cat=='mumu': samples_BDT = samples_mumu
else: raise ValueError(f'cat {cat} not supported')

very_big_sample = [f'wzp6_ee_mumu_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}', f'wzp6_ee_ee_Mee_30_150_ecm{ecm}']
big_sample      = [f'p8_ee_WW_mumu_ecm{ecm}', 'wzp6_gaga_mumu_60_ecm365']

# Process list with parameters for RDataFrame analysis
processList = {i:{'fraction': 1, 'chunks': 20 if i in very_big_sample else
                  (10 if i in big_sample else 1)} for i in samples_BDT}



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    """New_RDataFrame analysis class for pre-selection."""

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        """Apply analysis graph construction to the dataframe."""
        df = training_ll(df, cat, ecm)
        return df

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        """Define output branches to save."""
        branchList = sorted([
            # Lepton kinematics (leading and subleading)
            'leading_p',    'leading_pT',    'leading_theta',    'leading_phi',
            'subleading_p', 'subleading_pT', 'subleading_theta', 'subleading_phi',

            # Angular correlation
            'acolinearity', 'acoplanarity', 'acopolarity', 'deltaR',

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
