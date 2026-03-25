##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os, sys

# Add parent directory to path so package and sel modules are found
# This is necessary for HTCondor batch jobs to find local modules
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from package.userConfig import loc, get_params
from package.config import get_process_list
from sel.presel.leptonic import training_ll, branch_list_ll
from sel.presel.hadronic import training_qq, branch_list_qq

# Load config from temporary JSON if running automated, else prompt
env = os.environ.copy()
cat, ecm = get_params(env, '1-run.json')



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
# nCPUS = 20

# Run on HTCondor batch system (default is False)
runBatch = True if env.get('RUN_BATCH') else False
# Batch queue name for HTCondor (default is workday)
batchQueue = 'espresso'  # 'longlunch'
# Computing account for HTCondor (default is group_u_FCC.local_gen)
compGroup = 'group_u_FCC.local_gen'



################################
### SETUP SAMPLES TO PROCESS ###
################################

processList = get_process_list(cat, ecm, train=True, batch=runBatch)



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
