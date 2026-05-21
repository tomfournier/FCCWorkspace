#!/usr/bin/env python3
"""
Unified optimization script for chi2 pairing methods in e+e- -> ZH analysis.

Optimizes chi2-based pairing efficiency by varying the weighting parameter (chi2_frac)
for both leptonic channels (mll: mass+recoil) and leptonic+momentum channels (pll).
Compares reconstructed pairings with MC truth to find optimal parameter values.

This script uses uproot and awkward arrays for efficient, type-safe data handling
and avoids ROOT's automatic type inference limitations.

Usage:
    python3 optimize.py --cat ee/mumu [--method mll-pll] [--incr 0.01]
"""

#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import sys, time

import numpy as np

t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    optimize=True,
    description='Optmisation Script'
)
arg = parse_args(parser, True)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
loc.set_default_type('Path')
from package.config import timer
from package.func.optimization import Optimizer



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main execution function.

    Orchestrates optimization workflow:
    1. Parse command-line arguments and setup logging
    2. Load process list and iterate over processes
    3. For each chi2 method (mll and/or pll):
       - Initialize optimizer and load data
       - Scan chi2 parameter space
       - Save optimization results and optimal distributions
    4. Generate summary report comparing methods
    """
    cat, ecm, nevents = arg.cat, arg.ecm, arg.nevents

    inDir  = loc.get('OPTIMISATION',     cat, ecm)
    outDir = loc.get('OPTIMISATION_RES', cat, ecm)

    if not inDir.exists():
        LOGGER.error(f'Input directory not found at {inDir}')
        sys.exit(1)

    if arg.procs == '':
        procs = sorted([d for d in inDir.iterdir() if d.is_dir()])
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

    # Calculate rounding precision from arg.incr
    incr_str = f"{arg.incr:.10f}".rstrip('0').rstrip('.')
    precision = len(incr_str.split('.')[-1]) if '.' in incr_str else 0

    # Store results for both methods
    chi2_methods = arg.method.split('-')
    all_results = {method: {} for method in chi2_methods}

    # Run optimization for both chi2 methods
    for chi2_method in chi2_methods:
        LOGGER.info(f'{"="*38}\n'+f'Running optimization for chi2_{chi2_method}'.center(38)+f'\n{"="*38}\n')

        baseline_value = 0.6 if chi2_method == 'mll' else 0

        for proc in procs:
            optimizer = Optimizer(proc, inDir, outDir, ecm, nevents=nevents, chi2_method=chi2_method)
            optimizer.optimize(np.arange(0, 1+arg.incr, arg.incr), precision=precision)
            optimizer.save_results()

            # Find optimal chi2_frac
            best_frac = max(optimizer.results.items(), key=lambda x: x[1]['multi_pair']['efficiency'])[0]
            baseline_result = optimizer.results.get(round(baseline_value, precision), None)
            optimal_result  = optimizer.results[best_frac]

            # Store results for comparison
            all_results[chi2_method][proc] = {
                'baseline_value':      baseline_value,
                'baseline_efficiency': baseline_result['multi_pair']['efficiency'],
                'optimal_value':       best_frac,
                'optimal_efficiency':  optimal_result['multi_pair']['efficiency']
            }

            LOGGER.info(f'Process: {proc}\n'
                        f'  Baseline (chi2_{chi2_method} = {baseline_value:<{precision+2}}): {baseline_result["multi_pair"]["efficiency"]*100:.2f} %\n'
                        f'  Optimal  (chi2_{chi2_method} = {best_frac:<{precision+2}}): {optimal_result["multi_pair"]["efficiency"]*100:.2f} %')

            # Save distributions for baseline
            LOGGER.debug(f'Saving variables distribution for chi2_{chi2_method} = {baseline_value}')
            optimizer.save_distributions(round(baseline_value, precision), 'results_baseline.root')

            # Save distributions for optimal chi2_frac
            LOGGER.debug(f'Saving variables distribution for optimal chi2_{chi2_method} = {best_frac}')
            optimizer.save_distributions(best_frac, 'results_optimal.root')

    # Print summary comparison
    LOGGER.info(f'{"="*26}\n'+'OPTIMIZATION SUMMARY'.center(26)+f'\n{"="*26}\n')

    for proc in procs:
        LOGGER.info(f'Process: {proc}')
        for chi2_method in chi2_methods:
            if proc in all_results[chi2_method]:
                res = all_results[chi2_method][proc]
                if res['baseline_efficiency'] == 0:
                    improvement = 0
                else:
                    improvement = (res["optimal_efficiency"] - res["baseline_efficiency"]) / res["baseline_efficiency"] * 100
                LOGGER.info(f'chi2_{chi2_method}:\n'
                            f'  Baseline (value = {res["baseline_value"]:<{precision+2}}): efficiency = {res["baseline_efficiency"]*100:.2f} %\n'
                            f'  Optimal  (value = {res["optimal_value"]:<{precision+2}}): efficiency = {res["optimal_efficiency"]*100:.2f} %\n'
                            f'  Improvement: {improvement:+.2f}%')


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution:', exc_info=True)
    finally:
        # Print execution time
        timer(t)
