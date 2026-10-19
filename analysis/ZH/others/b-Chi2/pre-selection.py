################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging

from pathlib import Path



########################
### ARGUMENT PARSING ###
########################

from zh_others.parsing import create_parser
parser = create_parser('b-Chi2', 'pre-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.pre-selection')



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from samples import get_process_list, get_include, get_exclude
from zh_others.sel.chi2 import (
    optimize_ll, branch_list_ll,
    optimize_qq, branch_list_qq
)
from zh_others.userConfig import loc



#############################
### SETUP CONFIG SETTINGS ###
#############################

cat, ecm = arg.cat, arg.ecm

# Output directory for chi2 optimization input events (default is local directory)
outputDir = loc.get('EVENTS_CHI2', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h',
                '../../../../functions/utils.h',
                '../../../../functions/FSR_recovery.h',
                '../../../../functions/optimization.h']

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

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

processList = get_process_list(
    cat, ecm, onlysig=True, train=True, batch=runBatch,
    include=get_include(arg.include),
    exclude=get_exclude(arg.exclude)
)



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    '''RDataFrame analysis class for applying pre-selection cuts and computing kinematic variables.

    This class defines the analysis pipeline using FCC's ROOT analysis framework,
    applying particle selection, kinematic calculations, and producing output branches
    for downstream chi2 optimization.
    '''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        '''Apply chi2 optimization analysis to the dataframe.

        Processes events through channel-specific optimization algorithms:
        - optimize_ll: For leptonic channels (ee, mumu) - computes chi2 distances
          between reconstructed lepton pairs and true Z boson momentum
        - optimize_qq: For hadronic channels (qq) - applies similar optimization
          for jet pair reconstruction

        Args:
            df: Input RDataFrame

        Returns:
            RDataFrame with optimization variables applied

        Raises:
            ValueError: If category is not supported
        '''
        if cat in ['ee', 'mumu']:
            df = optimize_ll(df, cat, ecm)
            return df
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define optimization output branches to save.

        Returns channel-specific branches containing chi2 distance metrics,
        pair variables, MC truth information, and reconstruction quality indicators.

        Returns:
            Sorted list of branch names to persist in output

        Raises:
            ValueError: If category is not supported
        '''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll)
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
