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
from package.tools.utils import high_low_sels
from package.tools.process import (
    preload_histograms, clear_histogram_cache
)



########################
### ARGUMENT PARSING ###
########################

parser = ArgumentParser()
# Define final state: ee, mumu, or both
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu', 'ee-mumu'], type=str, default='ee-mumu')
# Define center of mass energy
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)

# Flags to control which plot types to skip (inverted logic: flag skips the plot except for --scan)
parser.add_argument('--yields', help='Do not make yields plots',            action='store_true')
parser.add_argument('--decay',  help='Do not make Higgs decays only plots', action='store_true')
parser.add_argument('--make',   help='Do not make distribution plots',      action='store_true')
parser.add_argument('--scan',   help='Make significance scan plots',        action='store_true')

arg = parser.parse_args()



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Selection strategies to plot
sels = [
    'Baseline',
    'Baseline_miss',
    'Baseline_sep',
    
    'Baseline_vis', 'Baseline_inv',
]
hl = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']
sels = high_low_sels(sels, hl)


# Define physics processes
processes = mk_processes(
    procs=[
        'ZH', 'ZeeH', 'ZmumuH', 
        'WW', 'ZZ', 'Zgamma', 
        'Rare'], ecm=ecm
)

# Variables to plot
variables = [
    'leading_p', 'leading_theta', #'leading_phi',          # leading lepton variables
    'subleading_p', 'subleading_theta', #'subleading_phi', # subleading lepton variables
    'zll_m', 'zll_p', 'zll_theta', # 'zll_phi',            # Z boson properties
    'acolinearity', 'acoplanarity', 'deltaR',              # Angular separation variables
    'zll_recoil_m',                                        # Recoil mass (Higgs candidate)
    'visibleEnergy', 'cosTheta_miss', 'missingMass',       # Missing energy and mass
    'H',                                                   # Higgsstrahlungness
    'BDTscore'                                             # BDT score
]

# Define signal and background samples for AAAyields
plots = {
    'signal':      {proc: processes[proc] for proc in ['ZH']},
    'backgrounds': {proc: processes[proc] for proc in ['WW', 'ZZ', 'Zgamma', 'Rare']}
}

# Custom plot arguments for specific variables
args = {
    'cosTheta_miss': {'xmin': 0.9},
    'BDTscore': {
        240: {'ymin':1,    'ymax':1e5, 'rebin':2, 'which':'make'},
        365: {'ymin':1e-1, 'ymax':1e5, 'rebin':2, 'which':'make'}
    },
    'zll_recoil_m': {
        240: {
            '*_high': {'xmin':120, 'xmax':140, 'strict': False}
        }
    }
}



##########################
### EXECUTION FUNCTION ###
##########################

def run(cats, sels, vars, processes, colors, legend):
    '''Generate distribution plots for all channels, selections, and variables.'''
    for cat in cats:
        print(f'\n----->[Info] Making plots for {cat} channel\n')
        # Define process names (signal must be first)
        procs = ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare']

        # Define input and output directories
        inDir  = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')
        outDir = get_loc(loc.PLOTS_MEASUREMENT, cat, ecm, '')

        for sel in sels:
            if not arg.yields or not arg.make or not arg.decay or arg.scan:
                # Preload all histograms for this selection to avoid repeated file I/O
                all_procs = [p for proc in processes.values() for p in proc]
                preload_histograms(all_procs, inDir, suffix=f'_{sel}_histo', hNames=vars)
                print(f'\n----->[Info] Making plots for {sel} selection\n')

            # Generate yields plots unless skipped
            if not arg.yields: 
                from package.plots.plotting import AAAyields
                AAAyields('zll_p', inDir, outDir, plots, legend, colors, cat, sel, ecm=ecm, lumi=lumi)
            
            # Generate distribution and decay plots unless all skipped
            if not arg.make or not arg.decay or arg.scan:
                for var in vars:
                    print(f'\n----->[Info] Making plots for {var}')

                    # Generate significance scan plots if requested
                    if arg.scan:
                        from package.plots.plotting import significance
                        for reverse in [True, False]:
                            significance(var, inDir, outDir, sel, procs, processes, reverse=reverse)

                    # Generate Higgs decay mode plots unless skipped
                    if not arg.decay: 
                        from package.plots.plotting import args_decay, PlotDecays
                        kwarg_decay = args_decay(var, sel, ecm, lumi, args)
                        # Channel-specific decay plots (linear and log scale)
                        for logY in [False, True]:
                            PlotDecays(var, inDir, outDir, sel, [cat],    H_decays, logY=logY, tot=False, **kwarg_decay)
                            PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=logY, tot=True,  **kwarg_decay)
                    
                    # Generate standard distribution plots unless skipped
                    if not arg.make:
                        from package.plots.plotting import get_args, makePlot
                        kwarg = get_args(var, sel, ecm, lumi, args)
                        # Signal vs background plots (linear and log scale)
                        for logY in [False, True]:
                            makePlot(var, inDir, outDir, sel, procs, processes, colors, legend, logY=logY, **kwarg)
            
            # Clear cache after finishing this selection to free memory
            clear_histogram_cache()



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run plotting for all categories, selections and variables
    run(cats, sels, variables, processes, colors, labels)
    # Print execution time
    timer(t)
