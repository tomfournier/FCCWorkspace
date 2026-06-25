#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import time

import pandas as pd

from package.tools.utils import load_data

# Start timer for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sels=True,
    allow_qq=True,
    description='BDT Training Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc  # Directory management utilities
from package.config import timer    # Performance timing utility
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
    sels = ['Baseline']          # Default selections if not specified
else:
    sels = arg.sels.split('-')   # Parse selection names from command-line

# Process modes for training (signal and major backgrounds)
# Must match modes defined in process_input.py
modes = [
    f'Z{cat}H',                        # Signal
    'ZZ', f'WW{cat}', f'Z{cat}',       # Major backgrounds: diboson and Drell-Yan
    f'egamma_{cat}', f'gammae_{cat}',  # Photon-induced backgrounds
]
if (cat != 'qq') and not ((cat == 'ee') and ecm == 365):
    modes.append(f'gaga_{cat}')        # Diphoton background

# XGBoost hyperparameter configuration
# These parameters control the BDT learning and regularization
configs = {
    'lep': {
        'n_estimators': 350,                           # Number of boosting rounds (trees to grow)
        'learning_rate': 0.20,                         # Step size shrinkage (lower = more conservative)
        'max_depth': 3,                                # Maximum tree depth (3 = shallow trees, reduces overfitting)
        'subsample': 0.5,                              # Subsample ratio of training instances per tree
        'gamma': 3,                                    # Minimum loss reduction required for tree split
        'min_child_weight': 10,                        # Minimum sum of instance weight in leaf node
        'max_delta_step': 0,                           # Maximum delta step for weight update (0 = no limit)
        'colsample_bytree': 0.5,                       # Subsample ratio of columns when building each tree
        'early_stopping_rounds': 25,                   # Validation metric needs to improve at least once every early stopping round
        'eval_metric': ['error', 'logloss', 'auc']     # Metrics to use for monitoring the training
    },
    'had': {
        'objective': 'binary:logistic',                # Learning task and the correspondinf learning objective to be used
        'eval_metric': ['error', 'logloss', 'auc'],    # Metrics to use for monitoring the training
        'n_estimators': 350,                           # Number of boosting round (tree to grow)
        'max_depth': 5,                                # Maximum tree depth
        'early_stopping_rounds': 1                     # Validation metric need to improve at least once every early stoppinf round
    }
}
config = configs['lep'] if cat in ['ee', 'mumu'] else configs['had']



##########################
### EXECUTION FUNCTION ###
##########################

def run(sels: list[str],
        modes: list[str],
        config: dict[str, str],
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

    for sel in sels:
        # Input: Preprocessed training data from process_input.py
        inDir  = loc.get('MVA_INPUTS', cat, ecm, sel)
        # Output: Trained BDT models and metadata
        outDir = loc.get('BDT',        cat, ecm, sel)

        # Load preprocessed training dataframe
        LOGGER.debug('Loading preprocessed training data and input variables')
        df, vars = load_data(inDir)

        print_stats(df, modes)

        # Create training and validation datasets (50% training, 50% validation)
        LOGGER.debug('Splitting data into training and validation sample')
        X_train, y_train, X_valid, y_valid, train_weight, valid_weight = split_data(df, vars, 'norm_weight')

        # Train XGBoost classifier with early stopping
        # Monitor validation loss and stop if no improvement for 'early' rounds
        bdt = train_model(
            X_train, y_train,
            X_valid, y_valid,
            train_weight,
            valid_weight,
            config,
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
        run(sels, modes, config)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
