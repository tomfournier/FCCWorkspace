###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import os

# Import user configuration paths and parameters
from package.config import get_process_list, quarks
from sel.presel.chi2 import (
    optimize_ll, optimize_qq,
    branch_list_ll, branch_list_qq
)
from package.userConfig import (
    loc, get_params
)

cat, ecm = get_params(os.environ.copy(), '3-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = loc.get('OPTIMISATION_TEST', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h',
                '../../../../functions/utils.h',
                '../../../../functions/optimisation.h']

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

Z_decays = (cat,) if cat in ['ee', 'mumu'] else quarks
processList = get_process_list(
    cat, ecm, onlysig=True, z_decays=Z_decays, batch=True
)



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFgraph():
    '''RDataFrame analysis class for pre-selection stage.'''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df, dataset):
        '''Apply analysis graph construction to the dataframe.'''
        if cat in ['ee', 'mumu']:
            df = optimize_ll(df, cat, ecm, dataset)
        elif cat == 'qq':
            df = optimize_qq(df, cat, ecm, dataset)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df, []

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define output branches to save.'''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll)
        elif cat == 'qq':
            return sorted(branch_list_qq)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
