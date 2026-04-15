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
    description='BDT Training Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from package.config import (
    timer,
    input_vars
)
from package.func.bdt import (
    print_stats,
    split_data,
    train_model,
    save_model
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

cat, ecm = arg.cat, arg.ecm
# Selection strategies for BDT training
if arg.sels=='':
    sels = ['Baseline', 'test']
else:
    sels = arg.sels.split('-')

# Process modes for training (signal vs various backgrounds)
modes = [
    f'Z{cat}H',                        # Signal
    'ZZ', f'WW{cat}', f'Z{cat}',       # Major backgrounds
    f'egamma_{cat}', f'gammae_{cat}',  # Photon backgrounds
    f'gaga_{cat}'                      # Diphoton background
]

# XGBoost hyperparameter configuration
config = {
    'n_estimators': 350,       # Number of boosting rounds
    'learning_rate': 0.20,     # Step size shrinkage
    'max_depth': 3,            # Maximum tree depth
    'subsample': 0.5,          # Subsample ratio of training instances
    'gamma': 3,                # Minimum loss reduction for split
    'min_child_weight': 10,    # Minimum sum of instance weight in child
    'max_delta_step': 0,       # Maximum delta step for weight update
    'colsample_bytree': 0.5,   # Subsample ratio of columns per tree
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
    """Train BDT models for each selection strategy."""

    # Display training variables
    LOGGER.info('Training variable used for the training')
    LOGGER.info(', '.join(var for var in vars))

    for sel in sels:
        # Define input and output directories
        inDir  = loc.get('MVA_INPUTS', cat, ecm, sel)
        outDir = loc.get('BDT',        cat, ecm, sel)

        if cat=='ee' and ecm==365:
            if 'gaga_ee' in modes:
                Modes = [m for m in modes if m!='gaga_ee']
        else:
            Modes = modes.copy()

        # Load preprocessed training data
        df = pd.read_pickle(f'{inDir}/preprocessed.pkl')
        print_stats(df, Modes)

        # Split data into training and validation sets
        LOGGER.info('Spltting data into training and validation sample')
        X_train, y_train, X_valid, y_valid = split_data(df, vars)

        # Train XGBoost model with early stopping
        bdt = train_model(
            X_train, y_train,
            X_valid, y_valid,
            config, early
        )

        # Save trained model
        save_model(bdt, vars, outDir)

        # Write feature map file for XGBoost evaluation
        fmap = pd.DataFrame({'vars':vars, 'Q':list('q' * len(vars))})
        fmap.to_csv(f'{outDir}/feature.txt', sep='\t', header=False)
        LOGGER.info(f'Wrote variable input in {outDir}/feature.txt')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run BDT training pipeline
    run(sels, modes, input_vars, config)
    # Print execution time
    timer(t)
