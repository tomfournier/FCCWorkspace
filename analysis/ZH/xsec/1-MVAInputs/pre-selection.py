##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

from package.userConfig import loc, get_params
from package.config import get_process_list
from sel.presel.leptonic import training_ll, branch_list_ll
from sel.presel.hadronic import training_qq, branch_list_qq

# Load config from temporary JSON if running automated, else prompt
cat, ecm = get_params(os.environ.copy(), '1-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)
# outputDir = loc.get('EVENTS_TRAINING', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h']

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

processList = get_process_list(cat, ecm, train=True)
processList = {f'wzp6_ee_qqH_ecm{ecm}': {'fraction': 1, 'chunks': 1}}



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    """New_RDataFrame analysis class for pre-selection."""

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        """Apply analysis graph construction to the dataframe."""
        if cat in ['ee', 'mumu']:
            df = training_ll(df, cat, ecm)
        elif cat == 'qq':
            df = training_qq(df, cat, ecm)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        """Define output branches to save."""
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll)
        elif cat == 'qq':
            return sorted(branch_list_qq)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
