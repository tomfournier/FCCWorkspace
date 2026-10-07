#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import time

import pandas as pd

from zh_xsec.tools.utils import load_data

# Start timer for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser, parse_args, set_log
from zh_xsec.logger import get_logger
parser = create_parser('2-BDT', 'train_bdt')
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_xsec.userConfig import loc  # Directory management utilities
from zh_xsec.config import (
    timer,          # Performance timing utility
    get_bdt_modes   # Build BDT signal and background samples
)
from zh_xsec.func.bdt import (
    print_stats,    # Display event counts per process
    split_data,     # Create train/validation split
    train_model,    # Train XGBoost classifier
    save_model      # Serialize trained model and metadata
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Analysis parameters from command-line arguments
cat, ecm, sels = arg.cat, arg.ecm, arg.sels.split('-')

# Process modes for training (signal and major backgrounds)
modes = list(get_bdt_modes(cat, ecm).keys())

# XGBoost hyperparameter configuration
# These parameters control the BDT learning and regularization
common_config = {
    'n_estimators': 350,                         # Number of boosting rounds (trees to grow)
    'subsample': 0.5,                            # Subsample ratio of training instances per tree
    'min_child_weight': 10,                      # Minimum sum of instance weight in leaf node
    'colsample_bytree': 0.5,                     # Subsample ratio of columns when building each tree
    'eval_metric': ['error', 'logloss', 'auc'],  # Metrics to use for monitoring the training
    'n_jobs': -1,                                # Use all available CPU cores
}
configs = {
    'lep': {
        **common_config,
        'learning_rate': 0.20,         # Step size shrinkage (lower = more conservative)
        'max_depth': 3,                # Maximum tree depth (3 = shallow trees, reduces overfitting)
        'gamma': 3,                    # Minimum loss reduction required for tree split
        'early_stopping_rounds': 25,   # Validation metric need to improve at least once every stopping round, stop training otherwise
    },
    'had': {
        **common_config,
        'objective': 'binary:logistic',   # Learning task and the corresponding learning objective to be used
        'max_depth': 5,                   # Maximum tree depth (5 = medium trees, need to verify overfitting)
        'early_stopping_rounds': 5,       # Validation metric need to improve at least once every stopping round, stop training otherwise
    },
}
config = configs['lep'] if cat in ['ee', 'mumu'] else configs['had']



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    """Train XGBoost BDT models for each selection strategy.

    Loads preprocessed training data, trains BDT classifiers with early stopping,
    and saves trained models along with feature maps for evaluation."""

    for sel in sels:
        # Input: Preprocessed training data from process_input.py
        inputdir  = loc.get('MVA_INPUTS', cat, ecm, sel)
        # Output: Trained BDT models and metadata
        outputdir = loc.get('BDT',        cat, ecm, sel)

        # Load preprocessed training dataframe
        LOGGER.debug('Loading preprocessed training data and input variables')
        df, vars = load_data(inputdir)

        print_stats(df, modes)

        # Create training and validation datasets
        LOGGER.debug('Splitting data into training and validation sample')
        X_train, y_train, X_valid, y_valid, train_weight, valid_weight = split_data(df, vars, arg.weight_name)
        train_weight = train_weight if arg.use_weights else None
        valid_weight = valid_weight if arg.use_weights else None

        msg = f"Using '{arg.weight_name}' as training weight" if arg.use_weights \
            else 'Not using weights for the BDT training'
        LOGGER.info(msg)

        # Train XGBoost classifier
        bdt = train_model(X_train, y_train, X_valid, y_valid,
                          train_weight, valid_weight, config)

        # Serialize trained model to disk (joblib and root format)
        save_model(bdt, vars, outputdir)

        # Write feature map file for XGBoost tree visualization
        # Maps tree split indices to human-readable variable names
        fmap = pd.DataFrame({'vars':vars, 'Q':list('q' * len(vars))})
        fmap.to_csv(f'{outputdir}/feature.txt', sep='\t', header=False)
        LOGGER.info(f'Wrote variable input in {outputdir}/feature.txt')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run BDT training pipeline
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
