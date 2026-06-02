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



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
if test: outputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)
else:    outputDir = loc.get('EVENTS_TRAINING',   cat, ecm)

# Custom C++ analysis functions for particle selection and kinematic calculations
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h']

# Production tag for accessing centrally produced EDM4Hep event samples
# Points to YAML files containing sample statistics from /cvmfs/fcc.cern.ch
prodTag = 'FCCee/winter2023_training/IDEA/'

# Process dictionary with cross-section and normalization information
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

# HTCondor batch system configuration (disabled by default)
runBatch = True if env.get('RUN_BATCH') else False
batchQueue = 'longlunch'             # Queue for batch submission (alternatives: 'workday')
compGroup = 'group_u_FCC.local_gen'  # Computing account for resource allocation

# User batch configuration: only set in batch mode to export RUN_UUID
userBatchConfig = env.get('RUN_USER_BATCH_CONFIG', '')

# Parallel processing configuration (default 4)
nCPUS = 4 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)


################################
### SETUP SAMPLES TO PROCESS ###
################################

processList = get_process_list(
    cat, ecm, train=True, batch=runBatch,
    chunks={'wzp6_gaga_ee_60_ecm365': 1}
)



#####################################################
### RDF ANALYSIS CLASS FOR PRE-SELECTION WORKFLOW ###
#####################################################

class RDFanalysis():
    """RDataFrame analysis class for applying pre-selection cuts and computing kinematic variables.

    This class defines the analysis pipeline using FCC's ROOT analysis framework,
    applying particle selection, kinematic calculations, and producing output branches
    for downstream BDT training.
    """

    # Define analysis graph construction and variable computation
    def analysers(df):
        """Apply analysis cuts and compute kinematic variables for the dataframe.

        Args:
            df: Input RDataFrame from EDM4Hep events

        Returns:
            df: Modified RDataFrame with new kinematic variables and applied selections
        """
        if cat in ['ee', 'mumu']:
            df = training_ll(df, cat, ecm, test)
        elif cat == 'qq':
            df = training_qq(df, cat, ecm, test)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df

    # Define output branches to save from processed events
    def output():
        """Return list of output branches to save from processed events.

        Returns:
            list: Names of kinematic variables and event properties to output as ROOT branches
        """
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll)
        elif cat == 'qq':
            return sorted(branch_list_qq)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
