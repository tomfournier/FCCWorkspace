################################
### STANDARD LIBRARY IMPORTS ###
################################

import sys, logging

from pathlib import Path



########################
### ARGUMENT PARSIGN ###
########################

from zh_xsec.parsing import create_parser
parser = create_parser('1-Selection', 'pre-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.pre-selection')



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

from zh_mass.userConfig import loc
from samples import get_process_list_mass, get_include, get_exclude
from zh_mass.sel.presel import presel, branch_list



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, test = arg.cat, arg.ecm, arg.do_test

# Output: Preprocessed events for measurement and fit stages
if test: outputDir = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    outputDir = loc.get('EVENTS',      cat, ecm)  # Full event sample
LOGGER.info(f'Running the pre-selection for {cat = } | {ecm = } | {test = }')

# Custom C++ analysis functions for particle selection and calculations
includePaths = ['../../../../functions/functions.h']

# Production tag for accessing centrally produced EDM4Hep event samples
# Points to YAML files containing sample statistics from /cvmfs/fcc.cern.ch
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary with cross-section and normalization information
# Source: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# HTCondor batch system configuration (disabled by default)
runBatch   = arg.run_batch           # Submit the job to HTCondor
batchQueue = arg.job_flavor          # Queue for batch submission
compGroup = 'group_u_FCC.local_gen'  # Computing account for resource allocation

# To make HTCondor execute userBatchConfig.sh
# This will make HTCondor detect FCCWorkspace/python and custom modules in analysis/**
userBatchConfig = str(Path(__file__).resolve().parents[4] / 'userBatchConfig.sh')

# Parallel processing configuration
nCPUS = 4 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)

if arg.run_batch:
    LOGGER.info(f'Running script on HTCondor with {nCPUS} CPUs and using {batchQueue} job flavor')



##########################
### DEFINE SAMPLE LIST ###
##########################

# Retrieve all samples for this channel and energy from central configuration
processList = get_process_list_mass(
    cat, ecm, batch=runBatch,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=get_include(arg.include),
    exclude=get_exclude(arg.exclude)
)



#####################################################
### RDF ANALYSIS CLASS FOR PRE-SELECTION WORKFLOW ###
#####################################################

class RDFgraph():
    '''RDataFrame analysis class for pre-selection stage.'''

    @staticmethod
    def dataset_name(dataset) -> str:
        '''Recover the process name for direct-file batch jobs.'''
        if dataset: return dataset

        import os, re
        match = re.fullmatch(r'job_(.+)_chunk_\d+', os.path.basename(os.getcwd()))
        return match.group(1) if match else dataset

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df, dataset):
        '''Apply analysis graph construction to the dataframe.'''
        if cat in ['ee', 'mumu']:
            df, params = presel(df, cat, ecm, dataset, test)
            return df, params
        raise ValueError(f'{cat = } not supported, choose between [ee, mumu]')

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define output branches to save.'''
        if cat in ['ee', 'mumu']:
            return sorted(branch_list)
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')
