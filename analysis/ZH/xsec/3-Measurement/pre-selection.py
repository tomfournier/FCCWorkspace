################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, re, sys, logging

# Add parent directory to path so package and sel modules are found
# This is necessary for HTCondor batch jobs to find local modules
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path: sys.path.insert(0, script_dir)



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser
parser = create_parser('3-Measurement', 'pre-selection')
cmd_args = globals().get('cmdline_args')
arguments = cmd_args['unknown'] if cmd_args is not None else sys.argv[1:]
arg, _ = parser.parse_known_args(arguments)

LOGGER = logging.getLogger('FCCAnalyses.pre-selection')



###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################



# Load analysis configuration and preselection functions
from package.userConfig import loc
from package.config import (
    get_process_list,
    parse_sample_selection,
    parse_sample_exclusion
)
from sel.presel.leptonic import get_systs_list, presel_ll, branch_list_ll
from sel.presel.hadronic import presel_qq, branch_list_qq



##############################
### CONFIGURE INPUT/OUTPUT ###
##############################

cat, ecm, test = arg.cat, arg.ecm, arg.test

# Output directory for analysis events (default is local directory)
if test: outputDir = loc.get('EVENTS_TEST', cat, ecm)  # Test subset
else:    outputDir = loc.get('EVENTS',      cat, ecm)  # Full event sample

# Custom C++ analysis functions for particle selection and kinematic calculations
includePaths = ['../../../../functions/functions.h',
                '../../../../functions/functions_hadronic.h']

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

# Parallel processing configuration
nCPUS = 4 if runBatch else 20  # Number of CPUs for parallel processing (-1 uses all available)



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Retrieve all samples for this channel and energy from central configuration
processList = get_process_list(
    cat, ecm, batch=runBatch,
    onlysig=arg.only_sig, onlybkg=arg.only_bkg,
    include=parse_sample_selection(arg.include),
    exclude=parse_sample_exclusion(arg.exclude)
)



#####################################################
### RDF ANALYSIS CLASS FOR PRE-SELECTION WORKFLOW ###
#####################################################

class RDFgraph():
    '''RDataFrame analysis class for pre-selection stage.'''

    @staticmethod
    def dataset_name(dataset):
        '''Recover the process name for direct-file batch jobs.'''
        if dataset:
            return dataset

        match = re.fullmatch(r'job_(.+)_chunk_\d+', os.path.basename(os.getcwd()))
        return match.group(1) if match else dataset

    # _________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df, dataset: str):
        """Apply analysis cuts and compute kinematic variables for the dataframe.

        Args:
            df: Input RDataFrame from EDM4Hep events
            dataset: Name of the sample used

        Returns:
            df: Modified RDataFrame with new kinematic variables and applied selections
            params: A list of histograms, TParameter or other object writable in a root file
        """
        dataset = RDFgraph.dataset_name(dataset)
        if cat in ['ee', 'mumu']:
            df, params = presel_ll(df, cat, ecm, dataset, test)
        elif cat == 'qq':
            df, params = presel_qq(df, cat, ecm, dataset, test)
        else:
            raise ValueError(f'{cat = } not supported, choose between [ee, mumu, qq]')
        return df, params

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        """Return list of output branches to save from processed events.

        Returns:
            list: Names of kinematic variables and event properties to output as ROOT branches
        """
        if cat in ['ee', 'mumu']:
            return sorted(branch_list_ll + get_systs_list(cat))
        elif cat == 'qq':
            return sorted(branch_list_qq)
        else:
            raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')
