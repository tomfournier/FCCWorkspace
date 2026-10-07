################################
### STANDARD LIBRARY IMPORTS ###
################################

from time import time

# Start timer for performance tracking
t = time()



########################
### ARGUMENT PARSING ###
########################

from zh_xsec.parsing import create_parser, set_log
from zh_xsec.logger import get_logger
parser = create_parser('3-Measurement', 'plots')
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory paths and process configurations
from zh_xsec.userConfig import loc
from zh_xsec.config import (
    timer,              # Timing utility
    z_decays,           # Z boson decay modes
    H_decays,           # Higgs decay modes
    vars_label_ll,
    vars_label_qq
)
from zh_xsec.tools.utils import high_low_sels  # High/low control region helpers
from sel_xsec.final.leptonic import histos_ll, custom_hists_ll
from sel_xsec.final.hadronic import histos_qq, custom_hists_qq
from zh_xsec.plots.plotting import (
    AAAyields, Efficiency, PlotDecays, get_args, makePlot,
    plot_configs, significance
)
from zh_xsec.tools.process import (
    preload_histograms,     # Preload histogram cache for performance
    clear_histogram_cache   # Clear cache when done
)



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm, sels = arg.cat.split('-'), arg.ecm, arg.sels.split('-')
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

if arg.hl: sels = high_low_sels(sels, arg.hlsel.split('-') +
                                arg.hl_include.split('-'))
quiet = not arg.verbose

# Custom plot arguments for specific variables
args = {'cosTheta_miss': {'xmin': 0.9},
        'zll_recoil_m': {240: {'*_high': {'xmin':120, 'xmax':140, 'strict': False}}}}


VARIABLES = {
    'lep': list(histos_ll) + list(h['name'] for h in custom_hists_ll.values()) + ['BDTscore'],
    'had': list(histos_qq) + list(h['name'] for h in custom_hists_qq.values()) + ['BDTscore'],
}



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    '''Generate yields, distributions, decay, and scan plots for the analysis.'''
    for cat in cats:
        LOGGER.info(f'Making plots for {cat} channel')
        plots = plot_configs(ecm, cat)

        all_procs = [process for process_group in plots['total'].values()
                     for processes in process_group.values()
                     for process in processes]

        variables = VARIABLES['lep' if cat in ['ee', 'mumu'] else 'had']
        variables = variables if 'all' in arg.variables else arg.variables.split('-')

        var_labels = vars_label_ll if cat in ['ee', 'mumu'] else vars_label_qq

        # Define input and output directories
        inDir  = loc.get('HIST_PREPROCESSED', cat, ecm)
        outDir = loc.get('PLOTS_MEASUREMENT', cat, ecm)

        for sel in sels:
            # Shared context is filtered by get_args for each target function.
            plot_context = {'sel': sel, 'cat': cat, 'ecm': ecm, 'lumi': lumi,
                            'inDir': inDir, 'outDir': outDir, 'quiet': quiet}

            def plot_kwargs(config_var: str, function, **overrides):
                '''Resolve one function's defaults and selection-specific options.'''
                return get_args(config_var, function, args, plot_context, **overrides)

            if arg.yields or arg.make or arg.decay or arg.scan:
                # Avoid repeated ROOT I/O when several plot families are enabled.
                LOGGER.info(f'Making plots for {sel} selection')
                if (arg.make or arg.decay or arg.scan) and not (len(variables)==1 or len(sels)==1):
                    preload_histograms(all_procs, inDir, f'_{sel}_histo', variables)

            # Generate yields plots unless skipped
            if arg.yields:
                from zh_xsec.config import quarks, z_labels, h_labels

                yield_args = plot_kwargs('acolinearity', AAAyields, hName='acolinearity')
                for total, config in [(False, plots['category']), (True, plots['total'])]:
                    # Category and total yields share every option except plots/tot.
                    yield_args.update(plots=config, tot=total)
                    AAAyields(**yield_args)

                Z_decays = [cat] if cat in ['ee', 'mumu'] else quarks
                efficiencies = [(Z_decays, h_labels, 'selection_efficiency',     False),
                                (z_decays, h_labels, 'selection_efficiency_tot', False),
                                (z_decays, z_labels, 'efficiency_selection',     True)]

                for z_modes, decay_labels, name, invert in efficiencies:
                    efficiency_args = plot_kwargs('acolinearity', Efficiency,
                                                  hName='acolinearity',
                                                  z_decays=z_modes, h_decays=H_decays,
                                                  h_labels=decay_labels, outName=name, invert=invert)
                    Efficiency(**efficiency_args)


            # Generate distribution and decay plots unless all skipped
            if arg.make or arg.decay or arg.scan:
                for variable in variables:
                    LOGGER.info(f'Making plots for {variable}')

                    # Generate significance scan plots if requested
                    if arg.scan:
                        # One configuration is reused for both cumulative directions.
                        scan_args = plot_kwargs(variable, significance,
                                                plots=plots['category'], var_labels=var_labels)
                        for reverse in [True, False]:
                            scan_args.update(reverse=reverse)
                            significance(**scan_args)

                    # Generate Higgs decay mode plots unless skipped
                    if arg.decay:
                        # Reuse invariant decay options across scales and categories.
                        decay_args = plot_kwargs(variable, PlotDecays)
                        for logY in [False, True]:
                            decay_args['logY'] = logY
                            for total, config in [(False, plots['decay']), (True, plots['total_decay'])]:
                                decay_args.update(plots=config, tot=total)
                                PlotDecays(**decay_args)

                    # Generate standard distribution plots unless skipped
                    if arg.make:
                        # Reuse invariant options for category and total distributions.
                        make_args = plot_kwargs(variable, makePlot)
                        for logY in [False, True]:
                            make_args['logY'] = logY
                            make_args['sig_scale'] = 1 if logY else make_args['sig_scale']
                            for total, config in [(False, plots['category']), (True, plots['total'])]:
                                make_args.update(plots=config, tot=total)
                                makePlot(**make_args)

            # Clear cache after finishing this selection to free memory
            clear_histogram_cache()


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
