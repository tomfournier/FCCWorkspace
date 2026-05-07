###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import os

# Import user configuration paths and parameters
from package.config import get_process_list, quarks
from sel.presel.chi2 import (
    fsr_recovery,
    branch_list_fsr
)
from package.userConfig import (
    loc, get_params
)

cat, ecm = get_params(os.environ.copy(), '3a-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = loc.get('FSR_TREE', cat, ecm)

# Include custom C++ analysis functions
includePaths = [
    '../../../../functions/functions.h',
    '../../../../functions/utils.h',
    '../../../../functions/FSR_recovery.h'
]

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# Optional: Number of CPUs for parallel processing
# (default is 4, -1 uses all cores available)
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

decays = [cat] if cat in ['ee', 'mumu'] else quarks
processList = get_process_list(
    cat, ecm, onlysig=True, batch=True, z_decays=decays,
    include={
        'sig': {
            f'wzp6_ee_{cat}H_ecm{ecm}': {'fraction': 1, 'chunks': 5}
        }
    }
)



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    '''RDataFrame analysis class for pre-selection stage.'''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        '''Apply analysis graph construction to the dataframe.'''
        df = fsr_recovery(df, cat)
        return df

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define output branches to save.'''
        return sorted(branch_list_fsr)
