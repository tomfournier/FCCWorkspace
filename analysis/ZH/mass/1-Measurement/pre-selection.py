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
from sel.presel import presel, branch_list

# Load environment to know which configuration to use
env = os.environ.copy()

# Get UUID from environment (set by 1-run.py), fallback to default if UUID not set
run_uuid = env.get('RUN_UUID')
config_name = f'1-run-{run_uuid}.json' if run_uuid else '1-run.json'

# Load analysis configuration from JSON or environment variables
# cat: decay category (ee, mumu, qq)
# ecm: center of mass energy (e.g., 240, 365 GeV)
# test: whether to apply kinematic cuts or not
cat, ecm, test = get_params(env, config_name, qq_allowed=True)



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

# Output: Preprocessed events for measurement and fit stages
if test: outputDir = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    outputDir = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Custom C++ analysis functions for particle selection and calculations
includePaths = ['../../../../functions/functions.h']

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

# User batch configuration: only set in batch mode to export RUN_UUID
userBatchConfig = env.get('RUN_USER_BATCH_CONFIG', '')

# Parallel processing configuration
nCPUS = 4 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)


##########################
### DEFINE SAMPLE LIST ###
##########################

# Retrieve all samples for this channel and energy from central configuration
processList = get_process_list(cat, ecm, batch=runBatch)
process = {
    f'wzp6_ee_mumuH_ecm{ecm}':                  {'fraction':1},
    f'wzp6_ee_mumuH_mH-lower-100MeV_ecm{ecm}':  {'fraction':1},
    f'wzp6_ee_mumuH_mH-lower-50MeV_ecm{ecm}':   {'fraction':1},
    f'wzp6_ee_mumuH_mH-higher-100MeV_ecm{ecm}': {'fraction':1},
    f'wzp6_ee_mumuH_mH-higher-50MeV_ecm{ecm}':  {'fraction':1},
    f'wzp6_ee_mumuH_mH-higher-50MeV_ecm{ecm}':  {'fraction':1},
    f'wzp6_ee_mumuH_BES-lower-1pc_ecm{ecm}':    {'fraction':1},
    f'wzp6_ee_mumuH_BES-higher-1pc_ecm{ecm}':   {'fraction':1},
    f'wzp6_ee_mumuH_BES-lower-6pc_ecm{ecm}':    {'fraction':1},
    f'wzp6_ee_mumuH_BES-higher-6pc_ecm{ecm}':   {'fraction':1},
}



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
            df, params = presel(df, cat, ecm, dataset, test)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df, params

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        '''Define output branches to save.'''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
