################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging

from pathlib import Path



########################
### ARGUMENT PARSING ###
########################

from zh_others.parsing import create_parser
parser = create_parser('a-FSR', 'pre-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknwon'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.pre-selection')



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from zh_others.userConfig import loc
from samples import get_process_list, get_include, get_exclude
from zh_others.sel.fsr import fsr_recovery, branch_list



#############################
### SETUP CONFIG SETTINGS ###
#############################

cat, ecm = arg.cat, arg.ecm

# Output directory for FSR recovery events (default is local directory)
outputDir = loc.get('EVENTS_FSR', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/utils.h',
                '../../../../functions/FSR_recovery.h']

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
    for downstream FSR recovery optimization.
    '''

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        '''Apply FSR recovery analysis to the dataframe.

        Processes events through FSR recovery algorithms to extract photon
        kinematic information and correlations.

        Args:
            df: Input RDataFrame

        Returns:
            RDataFrame with FSR recovery variables applied
        '''
        if cat in ['ee', 'mumu']:
            df = fsr_recovery(df, cat, ecm)
            return df
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define FSR recovery output branches to save.

        Returns a list of all branches related to FSR analysis including
        photon variables (momentum, angles, origin) and lepton-photon correlations.

        Returns:
            Sorted list of branch names to persist in output
        '''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list)
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')
