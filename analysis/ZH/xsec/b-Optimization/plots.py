#!/usr/bin/env python
"""
Unified plotting script for chi2 methods optimization results.

Combines plots_ll.py and plots_pll.py to generate plots for both mll and pll methods.
This script reads JSON results from optimize.py and generates comprehensive plots
to visualize pairing efficiency as a function of the chi2 parameter.

Usage:
    python3 plots.py --cat mumu --ecm 240 [--method mll-pll] [--procs process1-process2]
"""

#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library and scientific computing imports
import sys, time

import numpy as np

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    optimize=True,
    is_plot=True,
    description='Optimisation Plots Script'
)
arg = parse_args(parser, validate_cat=True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Import configuration paths and plot settings
from package.userConfig import loc, plot_file
loc.set_default_type('Path')
# Import utilities and plotting configurations
from package.config import timer, process_label
from package.plots.optimization import (
    load_results,
    extract_arrays,
    efficiency,
    pairing_composition,
    event_counts,
    compare_dists
)



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main execution function.

    Orchestrates plotting workflow:
    1. Parse command-line arguments and setup logging
    2. Load process list and iterate over processes and chi2 methods
    3. Generate optimization scan plots (efficiency, composition, event counts)
    4. Generate kinematic distribution comparison plots
    5. Color-code distributions by pairing match quality
    """
    cat, ecm = arg.cat, arg.ecm

    # Get input and output directories
    inDir  = loc.get('OPTIMISATION_RES',   cat, ecm)
    outDir = loc.get('PLOTS_OPTIMISATION', cat, ecm)

    if not inDir.exists():
        LOGGER.error(f'Input directory not found: {inDir}')
        sys.exit(1)

    # Get processes to plot
    if arg.procs == '':
        procs: list[str] = [p.name for p in inDir.iterdir() if p.is_dir()]
    else:
        processes: list[str] = arg.procs.split('-')
        procs = []
        for x in processes:
            if x.startswith('wzp6_ee_'):
                # Full process name already provided
                procs.append(x)
            elif x == 'all':
                # Special case: base process name without H{x} suffix
                procs.append(f'wzp6_ee_{cat}H_ecm{ecm}')
            else:
                # Short name: just the 'x' part, convert to full name
                procs.append(f'wzp6_ee_{cat}H_H{x}_ecm{ecm}')

    if not procs:
        LOGGER.error(f'No process found in {inDir}')
        sys.exit(1)

    # Get chi2 methods to plot
    chi2_methods: list[str] = arg.method.split('-')

    # Set LaTeX label for final state
    if cat == 'mumu':
        label = r'$Z(\rightarrow\mu^+\mu^-)H'
    elif cat == 'ee':
        label = r'$Z(\rightarrow e^+e^-)H'
    else:
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # Plot results for each chi2 method
    for chi2_method in chi2_methods:
        LOGGER.info(f'{"="*34}\n' + f'Generating plots for chi2_{chi2_method}'.center(34) + f'\n{"="*34}\n')

        # Plot results for each process
        for proc in procs:
            proc_dir = inDir / proc / chi2_method
            results_file = proc_dir / 'results.json'
            out_dir = outDir / proc / chi2_method

            proc_label = proc.replace(f'wzp6_ee_{cat}H', '').replace(f'_ecm{ecm}', '').replace('_H', '')
            if proc_label:
                Label = label + r'(\rightarrow ' + process_label[proc_label] + ')$'
            else:
                Label = label + r'$'

            if not results_file.exists():
                LOGGER.warning(f'Results file not found: {results_file}')
                continue

            LOGGER.info(f'Processing {proc}')

            # Load results
            results = load_results(results_file)

            if not results:
                LOGGER.warning(f'No results loaded for {proc}')
                continue

            # Create output directory
            out_dir.mkdir(exist_ok=True, parents=True)

            # Generate plots
            if arg.metrics:

                # Define categories and their output folder names
                categories = {
                    'overall':    'overall',
                    'one_pair':   'one',
                    'multi_pair': 'several'
                }

                # Generate optimization plots for each category
                for category, folder_name in categories.items():
                    cat_outDir = out_dir / folder_name

                    # Extract arrays for overall category
                    fracs, effs, n_correct, n_partial, n_incorrect, n_total = extract_arrays(results, category)

                    # Skip if no data for this category
                    if len(fracs) == 0 or np.all(n_total == 0):
                        LOGGER.warning(f'No data for {category} category in {proc}, skipping')
                        continue

                    cat_outDir.mkdir(exist_ok=True, parents=True)

                    # Generate optimization plots
                    LOGGER.info(f'Generating {category} plots for {proc}')
                    efficiency(
                        fracs, effs,
                        cat_outDir, Label, chi2_method, format=plot_file
                    )
                    pairing_composition(
                        fracs,
                        n_correct, n_partial, n_incorrect,
                        n_total, effs,
                        cat_outDir, Label, chi2_method, plot_file
                    )
                    event_counts(
                        fracs,
                        n_correct, n_partial, n_incorrect,
                        cat_outDir, Label, chi2_method, plot_file
                    )

            if arg.dist:
                LOGGER.debug(f'Generating distribution comparison plots for {proc}')
                baseline_file = proc_dir / 'results_baseline.root'
                optimal_file  = proc_dir / 'results_optimal.root'
                dist_outDir   = out_dir / 'distributions'
                origin_outDir = out_dir / 'origins'

                if baseline_file.exists() and optimal_file.exists():
                    compare_dists(
                        baseline_file, optimal_file,
                        dist_outDir, origin_outDir,
                        Label, chi2_method, plot_file
                    )
                else:
                    missing = ''
                    if not baseline_file.exists(): missing += f'Missing: {baseline_file}\n'
                    if not optimal_file.exists():  missing += f'Missing: {optimal_file}'
                    LOGGER.warning(f'Distribution file not found for {proc}\n{missing}')


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
