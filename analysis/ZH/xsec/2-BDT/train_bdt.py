##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

print('\n----->[Info] Loading modules')

from time import time
from argparse import ArgumentParser

import pandas as pd

# Start timer for performance tracking
t = time()

print('----->[Info] Loading custom modules\n')

from package.userConfig import loc
from package.config import (
    timer, 
    warning, 
    input_vars
)
from package.func.bdt import (
    print_stats, 
    split_data, 
    train_model, 
    save_model
)



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee or mumu
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

# Validate that final state is selected
if arg.cat=='':
    warning(log_msg='Final state was not selected, please select one to run this script')



#############################
### SETUP CONFIG SETTINGS ###
#############################

cat, ecm = arg.cat, arg.ecm
# Selection strategies for BDT training
sels = [
    'Baseline',
    # 'test'
]

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
    print('----->[Info] Training variable used for the training')
    print('\t'+', '.join(var for var in vars)+'\n')

    for sel in sels:
        # Define input and output directories
        inDir  = loc.get('MVA_INPUTS', cat, ecm, sel)
        outDir = loc.get('BDT',        cat, ecm, sel)

        if 'Baseline' in sel and cat=='ee' and ecm==365:
            if 'gaga_ee' in modes:
                Modes = [m for m in modes if m!='gaga_ee']
        else:
            Modes = modes.copy()

        # Load preprocessed training data
        df = pd.read_pickle(f'{inDir}/preprocessed.pkl')
        print_stats(df, Modes)

        # Split data into training and validation sets
        print('\n----->[Info] Spltting data into training and validation sample')
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
        print(f'----->[Info] Wrote variable input in {outDir}/feature.txt')



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run BDT training pipeline
    run(sels, modes, input_vars, config)
    # Print execution time
    timer(t)
