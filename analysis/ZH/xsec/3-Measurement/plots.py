################################
### STANDARD LIBRARY IMPORTS ###
################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, set_log
from package.logger import get_logger
parser = create_parser(
    cat_multi=True,        # Support multiple decay categories
    include_sels=True,     # Include selection strategy options
    plots=True,            # Include plot output options
    description='Measurement Plots Script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory paths and process configurations
from package.userConfig import loc
from package.config import (
    timer,              # Timing utility
    mk_processes,       # Build process definitions
    z_decays,           # Z boson decay modes
    H_decays,           # Higgs decay modes
    colors, labels,      # Plot styling
    vars_label_ll,
    vars_label_qq,
    vars_xlabel_ll,
    vars_xlabel_qq
)
from package.tools.utils import high_low_sels  # High/low control region helpers
from sel.final.leptonic import histos_ll, custom_hists_ll
from sel.final.hadronic import histos_qq, custom_hists_qq
from package.tools.process import (
    preload_histograms,     # Preload histogram cache for performance
    clear_histogram_cache   # Clear cache when done
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

# Selection strategies to plot
if arg.sels=='':
    sels = ['Baseline']
else:
    sels = arg.sels.split('-')
hl = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']
sels = high_low_sels(sels, hl)


# Define physics processes
procs_bkg = ['WW', 'ZZ', 'Zgamma', 'Rare']
if ecm == 365: procs_bkg.append('tt')
processes = mk_processes(
    procs=[
        'ZH', 'ZeeH', 'ZmumuH', 'ZqqH'
    ] + procs_bkg,
    ecm=ecm
)

# Define signal and background samples for AAAyields
plots_tot = {
    'signal':      {proc: processes[proc] for proc in ['ZH']},
    'backgrounds': {proc: processes[proc] for proc in procs_bkg}
}

# Custom plot arguments for specific variables
args = {
    'cosTheta_miss': {'xmin': 0.9},
    'zll_recoil_m': {
        240: {
            '*_high': {'xmin':120, 'xmax':140, 'strict': False}
        }
    }
}



##########################
### EXECUTION FUNCTION ###
##########################

def run(
        cats: str,
        sels: list[str],
        processes: dict[str, list[str]],
        colors: dict[str, list],
        legend: dict[str, list[str]]
         ) -> None:
    '''Generate distribution plots for all channels, selections, and variables.'''
    # Define process names for all Z decays (signal must be first)
    procs_tot = ['ZH'] + procs_bkg
    for cat in cats:
        LOGGER.info(f'Making plots for {cat} channel')
        # Define process names for specific channel (signal must be first)
        procs = [f'Z{cat}H'] + procs_bkg
        if cat in ['ee', 'mumu']:
            vars = list(histos_ll) + list(custom_hists_ll)
        elif cat == 'qq':
            vars = [v for v in histos_qq if v!='zqq_m_recoil_m'] + list(custom_hists_qq)
        vars.append('BDTscore')

        vars_label  = vars_label_ll  if cat in ['ee', 'mumu'] else vars_label_qq
        vars_xlabel = vars_xlabel_ll if cat in ['ee', 'mumu'] else vars_xlabel_qq

        plots = {
            'signal':      {proc: processes[proc] for proc in [f'Z{cat}H']},
            'backgrounds': {proc: processes[proc] for proc in procs_bkg}
        }

        # Define input and output directories
        inDir  = loc.get('HIST_PREPROCESSED', cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT', cat, ecm)

        for sel in sels:
            if arg.yields or arg.make or arg.decay or arg.scan:
                # Preload all histograms for this selection to avoid repeated file I/O
                all_procs = [p for proc in processes.values() for p in proc]
                LOGGER.info(f'Making plots for {sel} selection')
                preload_histograms(all_procs, inDir, suffix=f'_{sel}_histo', hNames=vars)

            # Generate yields plots unless skipped
            if arg.yields:
                from package.plots.plotting import AAAyields
                histo = 'zll_p' if cat in ['ee', 'mumu'] else 'zqq_p'
                AAAyields(histo, inDir, outDir, plots,     legend, colors, cat, sel, ecm=ecm, lumi=lumi, tot=False)
                AAAyields(histo, inDir, outDir, plots_tot, legend, colors, cat, sel, ecm=ecm, lumi=lumi, tot=True)

            # Generate distribution and decay plots unless all skipped
            if arg.make or arg.decay or arg.scan:
                for var in vars:
                    LOGGER.info(f'Making plots for {var}')

                    # Generate significance scan plots if requested
                    if arg.scan:
                        from package.plots.plotting import significance
                        for reverse in [True, False]:
                            significance(
                                var, inDir, outDir,
                                sel, procs, processes,
                                vars_label, vars_xlabel,
                                reverse=reverse
                            )

                    # Generate Higgs decay mode plots unless skipped
                    if arg.decay:
                        from package.plots.plotting import args_decay, PlotDecays
                        kwarg_decay = args_decay(var, sel, ecm, lumi, args)
                        # Channel-specific decay plots (linear and log scale)
                        for logY in [False, True]:
                            PlotDecays(var, inDir, outDir, sel, [cat],    H_decays, logY=logY, tot=False, **kwarg_decay)
                            PlotDecays(var, inDir, outDir, sel, z_decays, H_decays, logY=logY, tot=True,  **kwarg_decay)

                    # Generate standard distribution plots unless skipped
                    if arg.make:
                        from package.plots.plotting import get_args, makePlot
                        kwarg = get_args(var, sel, cat, ecm, lumi, args)
                        # Signal vs background plots (linear and log scale)
                        for logY in [False, True]:
                            makePlot(var, inDir, outDir, sel, procs,     processes, colors, legend, logY=logY, **kwarg)
                            makePlot(var, inDir, outDir, sel, procs_tot, processes, colors, legend, logY=logY, **kwarg)

            # Clear cache after finishing this selection to free memory
            clear_histogram_cache()


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run plotting for all categories, selections and variables
    try:
        run(cats, sels, processes, colors, labels)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
