#################################
### IMPORT STANDARD LIBRARIES ###
#################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,
    include_sels=True,
    plots=True,
    description='Measurement plots Script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
from package.config import (
    timer, mk_processes,
    z_decays, H_decays,
    colors, labels
)
from package.tools.utils import high_low_sels
from package.tools.process import (
    preload_histograms, clear_histogram_cache
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Selection strategies to plot
if arg.sels=='':
    sels = [
        'Baseline',
        'Baseline_miss',
        'Baseline_sep',
        'Baseline_vis', 'Baseline_inv',
        'test'
    ]
else:
    sels = arg.sels.split('-')
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
    'leading_p', 'leading_pT', 'leading_theta',                               # leading lepton variables
    'subleading_p', 'subleading_pT', 'subleading_theta',                      # subleading lepton variables
    'zll_m', 'zll_p', 'zll_pT', 'zll_theta', 'zll_costheta',                  # Z boson properties
    # 'zll_phi', 'leading_phi', 'subleading_phi',                               # Azimutal angles
    'acolinearity', 'acoplanarity', 'deltaR',                                 # Angular separation variables
    'zll_recoil_m',                                                           # Recoil mass (Higgs candidate)
    'visibleEnergy', 'cosTheta_miss', 'missingMass',  # missingEnergy         # Missing energy and mass
    'H',                                                                      # Higgsstrahlungness
    'BDTscore',                                                               # BDT score
    'ConeIsolation', 'n_leptons'
]

# Define signal and background samples for AAAyields
plots = {
    'signal':      {proc: processes[proc] for proc in ['ZH']},
    'backgrounds': {proc: processes[proc] for proc in ['WW', 'ZZ', 'Zgamma', 'Rare']}
}

# Custom plot arguments for specific variables
args = {
    'cosTheta_miss': {'xmin': 0.9},
    'zll_costheta':  {'rebin': 2},
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
        LOGGER.info(f'Making plots for {cat} channel')
        # Define process names (signal must be first)
        procs = ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare']

        # Define input and output directories
        inDir  = loc.get('HIST_PREPROCESSED', cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT', cat, ecm)

        for sel in sels:
            if not arg.yields or not arg.make or not arg.decay or arg.scan:
                # Preload all histograms for this selection to avoid repeated file I/O
                all_procs = [p for proc in processes.values() for p in proc]
                LOGGER.info(f'Making plots for {sel} selection')
                preload_histograms(all_procs, inDir, suffix=f'_{sel}_histo', hNames=vars)

            # Generate yields plots unless skipped
            if not arg.yields:
                from package.plots.plotting import AAAyields
                AAAyields('zll_p', inDir, outDir, plots, legend, colors, cat, sel, ecm=ecm, lumi=lumi)

            # Generate distribution and decay plots unless all skipped
            if not arg.make or not arg.decay or arg.scan:
                for var in vars:
                    LOGGER.info(f'Making plots for {var}')

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
    try:
        run(cats, sels, variables, processes, colors, labels)
    except Exception as e:
        LOGGER.error(f'Error during plotting: {e}')
    finally:
        # Print execution time
        timer(t)
