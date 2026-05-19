#################################
### IMPORT STANDARD LIBRARIES ###
#################################

from time import time
import pandas as pd

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sels=True,
    allow_qq=False,
    description='BDT Training Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc  # Directory management utilities
from package.config import (
    timer,                          # Performance timing utility
    input_vars                      # List of training variables
)
from package.func.bdt import (
    print_stats,                    # Display event counts per process
    split_data,                     # Create train/validation split
    train_model,                    # Train XGBoost classifier
    save_model                      # Serialize trained model and metadata
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm = arg.cat, arg.ecm  # Decay category and center-of-mass energy

# Selection strategies for BDT training (from command-line or defaults)
if arg.sels=='':
    sels = ['Baseline', 'test']  # Default selections if not specified
else:
    sels = arg.sels.split('-')   # Parse selection names from command-line

# Process modes for training (signal and major backgrounds)
# Must match modes defined in process_input.py
modes = [
    f'Z{cat}H',                        # Signal
    'ZZ', f'WW{cat}', f'Z{cat}',       # Major backgrounds: diboson and Drell-Yan
    f'egamma_{cat}', f'gammae_{cat}',  # Photon-induced backgrounds
    f'gaga_{cat}'                      # Diphoton background
]

# XGBoost hyperparameter configuration
# These parameters control the BDT learning and regularization
config = {
    'n_estimators': 350,       # Number of boosting rounds (trees to grow)
    'learning_rate': 0.20,     # Step size shrinkage (lower = more conservative)
    'max_depth': 3,            # Maximum tree depth (3 = shallow trees, reduces overfitting)
    'subsample': 0.5,          # Subsample ratio of training instances per tree
    'gamma': 3,                # Minimum loss reduction required for tree split
    'min_child_weight': 10,    # Minimum sum of instance weight in leaf node
    'max_delta_step': 0,       # Maximum delta step for weight update (0 = no limit)
    'colsample_bytree': 0.5,   # Subsample ratio of columns when building each tree
}



##########################
### EXECUTION FUNCTION ###
##########################

def run(sels: list[str],
        modes: list[str],
        vars: list[str],
        config: dict[str, str],
        early: int = 25
        ) -> None:
    """Train XGBoost BDT models for each selection strategy.

    Loads preprocessed training data, trains BDT classifiers with early stopping,
    and saves trained models along with feature maps for evaluation.

    Args:
        sels: List of selection strategies to train models for
        modes: List of process modes (signal and backgrounds)
        vars: List of input variables for BDT training
        config: XGBoost hyperparameter dictionary
        early: Early stopping patience (rounds without improvement before stopping)

    Returns:
        None (saves trained models and feature maps to BDT directories)
    """

    # Log the input variables being used for training
    LOGGER.info('Training variable used for the training')
    LOGGER.info(', '.join(var for var in vars) + '\n')

    for sel in sels:
        # Input: Preprocessed training data from process_input.py
        inDir  = loc.get('MVA_INPUTS', cat, ecm, sel)
        # Output: Trained BDT models and metadata
        outDir = loc.get('BDT',        cat, ecm, sel)

        # Skip diphoton process at 365 GeV (efficiency too low, not enough events)
        if cat=='ee' and ecm==365:
            if 'gaga_ee' in modes:
                Modes = [m for m in modes if m!='gaga_ee']
        else:
            Modes = modes.copy()

        # Load preprocessed training dataframe
        LOGGER.debug('Loading preprocessed training data')
        df = pd.read_pickle(f'{inDir}/preprocessed.pkl')
        print_stats(df, Modes)

        # Create training and validation datasets (50% training, 50% validation)
        LOGGER.debug('Splitting data into training and validation sample')
        X_train, y_train, X_valid, y_valid = split_data(df, vars)

        # Train XGBoost classifier with early stopping
        # Monitor validation loss and stop if no improvement for 'early' rounds
        bdt = train_model(
            X_train, y_train,
            X_valid, y_valid,
            config, early
        )

        # Serialize trained model to disk (joblib and root format)
        save_model(bdt, vars, outDir)

        # Write feature map file for XGBoost tree visualization
        # Maps tree split indices to human-readable variable names
        fmap = pd.DataFrame({'vars':vars, 'Q':list('q' * len(vars))})
        fmap.to_csv(f'{outDir}/feature.txt', sep='\t', header=False)
        LOGGER.info(f'Wrote variable input in {outDir}/feature.txt')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run BDT training pipeline
        run(sels, modes, input_vars, config)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
