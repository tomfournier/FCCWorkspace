################################
### STANDARD LIBRARY IMPORTS ###
################################

from pathlib import Path
import sys, logging

# Add parent directory to path so zh_xsec and sel modules are found
# This is necessary for HTCondor batch jobs to find local modules
# script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if script_dir not in sys.path: sys.path.insert(0, script_dir)



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser
parser = create_parser('1-MVAInputs', 'pre-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.pre-selection')



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_xsec.userConfig import loc
from samples import (
    get_process_list,
    parse_sample_selection,
    parse_sample_exclusion,
)
from sel_xsec.presel.leptonic import training_ll, branch_list_ll
from sel_xsec.presel.hadronic import training_qq, branch_list_qq



#############################
### SETUP CONFIG SETTINGS ###
#############################

cat, ecm, test = arg.cat, arg.ecm, arg.do_test
LOGGER.info(f'Running the pre-selection for {cat = } | {ecm = } | {test = }')

# Output directory for training events (default is local directory)
if   test:    outputDir = loc.get('EVENTS_TRAIN_TEST', cat, ecm)  # Test subset
elif arg.jan: outputDir = loc.get('EVENTS_TRAIN_JAN',  cat, ecm)  # Jan's samples
else:         outputDir = loc.get('EVENTS_TRAINING',   cat, ecm)  # Full event samples

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
runBatch   = arg.run_batch            # Submit the job to HTCondor
batchQueue = arg.job_flavor           # Queue for batch submission
compGroup  = 'group_u_FCC.local_gen'  # Computing account for resource allocation

# To make HTCondor execute userBatchConfig.sh
# This will make HTCondor detect FCCWorkspace/python and custom modules in analysis/**
userBatchConfig = str(Path(__file__).resolve().parents[4] / 'userBatchConfig.sh')

# Parallel processing configuration (default 4)
nCPUS = 4 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)

if arg.run_batch:
    LOGGER.info(f'Running script on HTCondor with {nCPUS} CPUs and using {batchQueue} job flavor')



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Retrieve all samples for this channel and energy from central configuration
processList = get_process_list(
    cat, ecm, train=True, batch=runBatch,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=parse_sample_selection(arg.include),
    exclude=parse_sample_exclusion(arg.exclude),
    chunks={'wzp6_gaga_ee_60_ecm365': 1},
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
    def output() -> list[str]:
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
