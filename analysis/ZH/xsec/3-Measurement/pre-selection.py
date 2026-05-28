###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import os, sys

# Add parent directory to path so package and sel modules are found
# This is necessary for HTCondor batch jobs to find local modules
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Load analysis configuration and preselection functions
from package.userConfig import loc, get_params
from package.config import get_process_list
from sel.presel.leptonic import presel_ll, branch_list_ll
from sel.presel.hadronic import presel_qq, branch_list_qq

# Load analysis parameters: decay category, CoM energy, test flag
env = os.environ.copy()
cat, ecm, test = get_params(env, '3-run.json')



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

# Output: Preprocessed events for measurement and fit stages
if test: outputDir = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    outputDir = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Custom C++ analysis functions for particle selection and calculations
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h']

# Production tag for accessing centrally produced EDM4Hep event samples
# Points to YAML files containing sample statistics from /cvmfs/fcc.cern.ch
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary with cross-section and normalization information
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# HTCondor batch system configuration (disabled by default)
runBatch = True if env.get('RUN_BATCH') else False
batchQueue = 'longlunch'  # Queue for batch submission (alternatives: 'espresso')
compGroup = 'group_u_FCC.local_gen'  # Computing account for resource allocation

# Parallel processing configuration
nCPUS = 8 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)


##########################
### DEFINE SAMPLE LIST ###
##########################

# Retrieve all samples for this channel and energy from central configuration
processList = get_process_list(cat, ecm)



#####################################################
### RDF ANALYSIS CLASS FOR PRE-SELECTION WORKFLOW ###
#####################################################

class RDFgraph():
    '''RDataFrame analysis class for pre-selection stage.'''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df, dataset):
        '''Apply analysis graph construction to the dataframe.'''
        if cat in ['ee', 'mumu']:
            df, params = presel_ll(df, cat, ecm, dataset, test)
        elif cat == 'qq':
            df, params = presel_qq(df, cat, ecm, dataset, test)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df, params

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        '''Define output branches to save.'''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll)
        elif cat == 'qq':
            return sorted(branch_list_qq)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
