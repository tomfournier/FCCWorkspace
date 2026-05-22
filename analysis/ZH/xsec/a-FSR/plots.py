#!/usr/bin/env python3
"""
Plotting script for FSR recovery analysis.

This script reads FSR recovery results and generates comprehensive plots to visualize:
- Photon distributions split by ISR/FSR origin
- Lepton distributions comparing reconstructed, true, and FSR-recovered quantities
- Correlation distributions split by same_parent criterion
- Distribution of number of radiated photons

Usage:
    python3 plots.py --cat mumu --ecm 240 [--procs process1-process2]
"""

#################################
### IMPORT STANDARD LIBRARIES ###
#################################

# Standard library and scientific computing imports
import sys, time

# Start execution timer
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger

parser = create_parser(
    cat_single=True,
    allow_qq=False,
    is_plot=True,
    optimize=True,
    only_procs=True,
    description='FSR Recovery Plots Script'
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
from package.plots.fsr import (
    load_data,
    n_radiated,
    leptons_origin,
    photon_distributions,
    leptons_fsr_comparison,
    correlation_scan,
    correlation_distributions,
    significance
)



#######################
### HELPER FUNCTION ###
#######################

# Helper function to find process input (either .root file or directory with chunk*.root pattern)
def find_process_path(inDir, process_name):
    """Find process input: first check for .root file, then check for directory."""
    root_file = inDir / f'{process_name}.root'
    if root_file.exists():
        return root_file
    proc_dir = inDir / process_name
    if proc_dir.exists() and proc_dir.is_dir():
        return proc_dir
    return None



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    """Main execution function.

    Orchestrates FSR recovery plotting workflow:
    1. Parse command-line arguments and setup logging
    2. Load FSR recovery analysis output from ROOT trees
    3. Generate plots for each process:
       - Photon distributions split by ISR/FSR origin
       - Lepton distributions split by decay origin
       - Lepton-photon correlation distributions
       - Significance optimization vs isolation cut
       - Photon multiplicity distribution
    4. Support both linear and logarithmic axis scales
    """
    cat, ecm = arg.cat, arg.ecm

    # Get input and output directories
    inDir  = loc.get('FSR_TREE',  cat, ecm)
    outDir = loc.get('PLOTS_FSR', cat, ecm)

    if not inDir.exists():
        LOGGER.error(f'Input directory not found: {inDir}')
        sys.exit(1)

    outDir.mkdir(exist_ok=True, parents=True)

    # Set LaTeX label
    if cat == 'mumu':
        label = r'$Z(\rightarrow\mu^+\mu^-)H'
    elif cat == 'ee':
        label = r'$Z(\rightarrow e^+e^-)H'
    else:
        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu]')

    # Get process directories
    if arg.procs == '':
        # Find both .root files and directories
        proc_dirs = []
        for item in sorted(inDir.iterdir()):
            if item.is_file() and item.suffix == '.root':
                proc_dirs.append(item)
            elif item.is_dir():
                proc_dirs.append(item)
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
        proc_dirs = [find_process_path(inDir, p) for p in procs]
        proc_dirs = [p for p in proc_dirs if p is not None]  # Filter out None values

    if not proc_dirs:
        LOGGER.error(f'No process directories found in {inDir}')
        sys.exit(1)

    LOGGER.info(f'Found {len(proc_dirs)} process(es) to process')

    # Process each directory
    for proc_dir in proc_dirs:
        if not proc_dir.exists():
            LOGGER.warning(f'Process directory not found: {proc_dir}, skipping')
            continue

        proc = proc_dir.name.replace('.root', '')
        LOGGER.info(f'Processing {proc}')

        proc_label = proc.replace(f'wzp6_ee_{cat}H', '').replace(f'_ecm{ecm}', '').replace('_H', '')
        if proc_label:
            Label = label + r'(\rightarrow ' + process_label[proc_label] + ')$'
        else:
            Label = label + r')$'

        # Load data from all chunk files (only required branches)
        data = load_data(proc_dir)

        if not data:
            LOGGER.warning(f'Could not load data from {proc_dir}, skipping')
            continue

        proc_outDir = outDir / proc_dir.name.replace('.root', '')
        proc_outDir.mkdir(exist_ok=True, parents=True)

        # Generate significance plot with optimal cut
        cut = 1.80 if ecm == 240 else (0.95 if ecm == 365 else 1e10)
        significance(data, proc_outDir, Label, format=plot_file)
        for iso_name in [f'LEPS_iso{x}_pair' for x in ['', '_ch', '_ne', '_ph', '_PH']]:
            correlation_scan(data, proc_outDir, Label, iso_name, cut, format=plot_file)

        # Generate all plots (linear and log scales)
        LOGGER.info(f'Generating plots for {proc_dir.name}')
        for scale in [False, True]:
            n_radiated(data, proc_outDir, Label, scale=scale, format=plot_file)
            photon_distributions(data, proc_outDir, Label, scale=scale, format=plot_file)
            leptons_fsr_comparison(data, proc_outDir, Label, scale=scale, format=plot_file)
            for iso_name in ['', '_ch', '_ne', '_ph', '_PH']:
                leptons_origin(data, proc_outDir, Label, scale=scale, iso_name=iso_name, iso_cut=cut, format=plot_file)
                correlation_distributions(data, proc_outDir, Label, scale=scale, iso_name=iso_name, iso_cut=cut, format=plot_file)

        LOGGER.info(f'Plots for {proc_dir.name} saved to {proc_outDir}\n')


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
