################################
### STANDARD LIBRARY IMPORTS ###
################################

from time import time
from typing import Any

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_mass.parsing import create_parser
from logger import get_logger
arg = create_parser('1-Selection', 'plots').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_mass.userConfig import loc
from tools.utils import timer
from constants import vars_label_ll as var_labels
from zh_mass.sel.final import histo_list, custom_hists
from tools.process import preload_hists, clear_hists
from plotting.plotting import plot_configs_mass, get_args
from samples import get_process_dict_mass



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm, sels = arg.cat.split('-'), arg.ecm, arg.sels.split('-')
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

quiet = not arg.verbose

# Custom plot arguments for specific variables
args = {'cosTheta_miss': {'xmin': 0.9}}

VARIABLES = list(histo_list) + list(h['name'] for h in custom_hists.values())
variables = VARIABLES if 'all' in arg.variables else arg.variables.split('-')


##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    '''Generate distribution plots for all channels, selections, and variables.'''
    # Define process names for all Z decays (signal must be first)
    for cat in cats:
        LOGGER.info(f'Making plots for {cat} channel')

        plots = plot_configs_mass(ecm, cat)
        procs = [proc for proc_group in plots['total'].values() for procs in proc_group.values() for proc in procs]

        # Define input and output directories
        inputdir  = loc.get('HIST_RAW',   cat, ecm)
        outputdir = loc.get('PLOT_MEAS', cat, ecm)

        for sel in sels:
            # Shared context is filtered by get_args for each target function.
            context = {'selection': sel, 'cat': cat, 'ecm': ecm, 'lumi': lumi,
                        'inputdir': inputdir, 'outputdir': outputdir, 'quiet': quiet}

            def plot_kwargs(config_var: str, function, **overrides) -> dict[str, Any]:
                '''Resolve one function's defaults and selection-specific options.'''
                return get_args(config_var, function, args, context, **overrides)
            
            if arg.yields or arg.make or arg.scan:
                # Avoid repeated ROOT I/O when several plot families are enabled.
                LOGGER.info(f'Making plots for {sel} selection')
                if (arg.make or arg.scan) and not (len(variables)==1 or len(sels)==1):
                    preload_hists(procs, inputdir, f'_{sel}_histo', variables)

            # Generate yields plots unless skipped
            if arg.yields:
                from plotting.plotting import AAAyields
                yield_args = plot_kwargs('acolinearity', AAAyields, hName='acolinearity', plots=plots['procs'])
                AAAyields(**yield_args)

            # Generate distribution and decay plots unless all skipped
            if arg.make or arg.scan:
                for variable in variables:
                    LOGGER.info(f'Making plots for {variable}')

                    # Generate significance scan plots if requested
                    if arg.scan:
                        from plotting.plotting import significance
                        scan_args = plot_kwargs(variable, significance,
                                                plots=plots['category'], var_labels=var_labels)
                        for reverse in [True, False]:
                            scan_args.update(reverse=reverse)
                            significance(**scan_args)

                    # Generate standard distribution plots unless skipped
                    if arg.make:
                        from plotting.plotting import get_args, makePlot
                        make_args = plot_kwargs(variable, makePlot, plots=plots['procs'])
                        # Signal vs background plots (linear and log scale)
                        for logY in [False, True]:
                            make_args.update(logY=logY)
                            makePlot(**make_args)

            # Clear cache after finishing this selection to free memory
            clear_hists()


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    # Run plotting for all categories, selections and variables
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
