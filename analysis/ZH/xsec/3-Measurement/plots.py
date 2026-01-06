##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from time import time
from argparse import ArgumentParser

# Start timer for performance tracking
t = time()

from package.userConfig import loc, get_loc

from package.config import (
    timer, mk_processes, 
    z_decays, H_decays, 
    colors, labels
)
from package.plots.plotting import (
    get_args, args_decay, 
    significance, 
    AAAyields,
    makePlot, 
    PlotDecays, 
)



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee, mumu, or both
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='ee-mumu')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)

# Flags to control which plot types to skip (inverted logic: flag skips the plot except for --scan)
parser.add_argument('--yields', help='Do not make yields plots',            action='store_true')
parser.add_argument('--decay',  help='Do not make Higgs decays only plots', action='store_true')
parser.add_argument('--make',   help='Do not make distribution plots',      action='store_true')
parser.add_argument('--scan',   help='Make significance scan plots',        action='store_true')

# Include all Z decay modes in plots
parser.add_argument('--tot', help='Include all the Z decays in the plots', action='store_true')
arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.1 if ecm==365 else -1)

# Selection strategies to plot
sels = [
    # 'Baseline', 'Baseline_high', 'Baseline_low',
    # 'Baseline_miss', 'Baseline_miss_high', 'Baseline_miss_low', 
    
    # 'Baseline_vis', 'Baseline_inv',
    'Baseline_sep', 'Baseline_sep_high', 'Baseline_sep_low',
    'Baseline_sep1', 'Baseline_sep1_high', 'Baseline_sep1_low',
    'Baseline_sep2', 'Baseline_sep2_high', 'Baseline_sep2_low',
    'Baseline_sep3', 'Baseline_sep3_high', 'Baseline_sep3_low',
]

# Define physics processes
processes = mk_processes(procs=['ZH', 'ZeeH', 'ZmumuH', 
                                'WW', 'ZZ', 'Zgamma', 'Rare'], 
                         ecm=ecm)

# Variables to plot
variables = [
    # Leptons kinematics
    'leading_p', 'leading_theta', 'subleading_p', 'subleading_theta',
    # 'leading_phi', 'subleading_phi',

    # Z boson properties
    'zll_m', 'zll_p', 'zll_theta', # 'zll_phi',

    # Angular separation variables
    'acolinearity', 'acoplanarity', 'deltaR',

    # Recoil mass (Higgs candidate)
    'zll_recoil_m',
    
    # Missing energy and mass
    'visibleEnergy', 'cosTheta_miss', 'missingMass',
    
    # Higgsstrahlungness
    'H',
    
    # BDT score
    'BDTscore'
]

# Define signal and background samples for AAAyields
plots = {
    'signal':      {proc: processes[proc] for proc in ['ZH']},
    'backgrounds': {proc: processes[proc] for proc in ['WW', 'ZZ', 'Zgamma', 'Rare']}
}

# Custom plot arguments for specific variables
args = {
    'cosTheta_miss': {'xmin':0.9},
    'BDTscore': {'ymin':1, 'ymax':1e5, 'rebin':2, 'which':'make'},
    'zll_recoil_m': {'xmin':120, 'xmax':140, 'sel':'*_high'}
}    



##########################
### EXECUTION FUNCTION ###
##########################

def run(cats, sels, vars, processes, colors, legend):
    '''Generate distribution plots for all channels, selections, and variables.'''
    for cat in cats:
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        # Define process names (signal must be first)
        procs = ['', 'WW', 'ZZ', 'Zgamma', 'Rare']
        procs[0] = f'Z{cat}H' if not arg.tot else 'ZH'

        # Define input and output directories
        inDir  = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')
        outDir = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, '')

        for sel in sels:
            if not arg.yields and not arg.make and not arg.decay and not arg.scan:
                print(f'\n----->[Info] Making plots for {sel} selection\n')

            # Generate yields plots unless skipped
            if not arg.yields: 
                AAAyields('zll_p', inDir, outDir, plots, legend, colors, cat, sel, ecm=ecm, lumi=lumi)
            
            # Generate distribution and decay plots unless all skipped
            if not arg.make or not arg.decay or arg.scan:
                for var in vars:
                    print(f'\n----->[Info] Making plots for {var}')

                    # Generate significance scan plots if requested
                    if arg.scan:
                        significance(var, inDir, outDir, sel, procs, processes, reverse=True)
                        significance(var, inDir, outDir, sel, procs, processes, reverse=False)

                    # Generate Higgs decay mode plots unless skipped
                    if not arg.decay: 
                        kwarg_decay = args_decay(var, sel, ecm, lumi, args)
                        # Channel-specific decay plots (linear and log scale)
                        PlotDecays(var, inDir, outDir, sel, [cat], H_decays, logY=False, tot=False, **kwarg_decay)
                        PlotDecays(var, inDir, outDir, sel, [cat], H_decays, logY=True,  tot=False, **kwarg_decay)

                        # All Z decay plots (linear and log scale)
                        PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=False, tot=True, **kwarg_decay)
                        PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=True,  tot=True, **kwarg_decay)
                    
                    # Generate standard distribution plots unless skipped
                    if not arg.make:  
                        kwarg = get_args(var, sel, ecm, lumi, args)
                        # Signal vs background plots (linear and log scale)
                        makePlot(var, inDir, outDir, sel, procs, processes, colors, legend, logY=False, **kwarg)
                        makePlot(var, inDir, outDir, sel, procs, processes, colors, legend, logY=True,  **kwarg)



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run plotting for all categories, selections and variables
    run(cats, sels, variables, processes, colors, labels)
    # Print execution time
    timer(t)
