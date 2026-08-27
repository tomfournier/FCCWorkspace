###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

import os

# Import user configuration paths and parameters
from package.config import get_process_list, quarks
from sel.presel.optimization.fsr import (
    fsr_recovery,
    branch_list
)
from package.userConfig import (
    loc, get_params
)

env = os.environ.copy()
cat, ecm, _ = get_params(env, 'a-run.json')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for FSR recovery events (default is local directory)
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
runBatch = True if env.get('RUN_BATCH') else False

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
    '''RDataFrame analysis class for FSR recovery stage.

    Applies FSR (Final State Radiation) recovery algorithms to reconstruct
    true photon kinematics from the detector response and calculates FSR-related
    kinematic variables.
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
        df = fsr_recovery(df, cat, ecm)
        return df

    # _____________________________________________________
    # Mandatory: output function defining branches to save
    def output() -> list[str]:
        '''Define FSR recovery output branches to save.

        Returns a list of all branches related to FSR analysis including
        photon variables (momentum, angles, origin) and lepton-photon correlations.

        Returns:
            Sorted list of branch names to persist in output
        '''
        return sorted(branch_list)
