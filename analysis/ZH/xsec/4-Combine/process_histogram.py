################################
### STANDARD LIBRARY IMPORTS ###
################################

import os

from time import time
from ROOT import TFile

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
    polarization=True,     # Include polarization/scale options
    description='Histogram Processing Script'
)
arg = parser.parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory paths and histogram processing utilities
from package.userConfig import loc
from package.config import (
    timer,              # Timing utility
    mk_processes,       # Build process definitions
    get_process_list,
    H_decays            # Higgs decay modes
)
from package.tools.utils import mkdir                       # Directory creation
from package.tools.process import get_hist, concat, unroll  # Histogram utilities



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Decay categories to process (from command-line: cat1-cat2 format)
cats, ecm = arg.cat.split('-'), arg.ecm

# Selection strategies to process (from command-line or defaults)
if arg.sels=='':
    sels = ['Baseline']  # Default selections
else:
    sels = arg.sels.split('-')  # Parse from command-line

# Define cross-section scaling factors based on polarization or luminosity
# Used to rescale histograms to match different beam configurations
if arg.ILC:  # ILC configuration (note: change fit to ASIMOV mode -t -1)
    procs_scales = {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919}
elif arg.polL:  # Left-handed polarization
    procs_scales = {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263}
elif arg.polR:  # Right-handed polarization
    procs_scales = {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018}
else:
    procs_scales  = {}  # No scaling
if procs_scales!={}:
    LOGGER.info('Rescaling histograms to alternative cross-sections')



# Define physics processes and their Higgs decay modes
processes = mk_processes(
    ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare'] + ['tt'] if ecm == 365 else [],
    H_decays=H_decays+('ZZ_noInv',), ecm=ecm
)



##########################
### EXECUTION FUNCTION ###
##########################

def run(cats: str,
        sels: str,
        ) -> None:
    """Process measurement histograms: split by BDT score and combine channels.

    For each category and selection, this function:
    1. Loads preprocessed histograms from measurement stage
    2. Splits histograms into high/low BDT score control regions (signal vs background enriched)
    3. Combines channels and applies cross-section scaling
    4. Saves processed histograms for combine/fit stages

    Args:
        cats: List of decay categories to process
        sels: List of selection strategies

    Returns:
        None (writes processed histograms to HIST_PROCESSED directories)
    """

    # Process each final state category
    for cat in cats:
        LOGGER.info(f'Processing histograms for {cat}')
        inDir = loc.get('HIST_PREPROCESSED', cat, ecm)

        # Histograms to process for BDT training/measurement
        hNames = ['zll_recoil_m'] if cat in ['ee', 'mumu'] else ['zqq_m_recoil_m']
        samples = get_process_list(cat, ecm).keys()

        # Process each selection strategy
        for sel in sels:
            LOGGER.info(f'Processing histograms for {sel}')

            outDir = loc.get('HIST_PROCESSED', cat, ecm, sel)
            mkdir(outDir)

            # Define histogram name suffixes for control regions and combined
            suffix_high = f'_{sel}_high_histo'
            suffix_low  = f'_{sel}_low_histo'
            suffix_base = f'_{sel}_histo'
            # Filter samples to only those that exist for this selection
            existing_samples = []
            for sample in samples:
                if os.path.exists(f'{inDir}/{sample}{suffix_base}.root'):
                    existing_samples.append(sample)

            if not existing_samples:
                LOGGER.warning(f'No sample found for {sel} in {cat}')
                continue

            # Process each sample
            for sample in existing_samples:
                LOGGER.debug(f'Processing {sample}')
                hists = []
                has_valid_hist = False
                for hName in hNames:

                    # Retrieve high and low region histograms with optional scaling
                    h_high = get_hist(
                        hName, sample, processes, inDir,
                        suffix=suffix_high,
                        rebin=1,
                        proc_scales=procs_scales
                    )
                    h_low  = get_hist(
                        hName, sample, processes, inDir,
                        suffix=suffix_low,
                        rebin=1,
                        proc_scales=procs_scales
                    )

                    if h_high is None or h_low is None:
                        continue

                    # Rename histograms to denote region
                    h_high.SetName(f'z{cat}_fit_high')
                    h_low.SetName(f'z{cat}_fit_low')

                    has_valid_hist = True
                    # Concatenate high and low region histograms
                    if cat in ['ee', 'mumu']:
                        h = concat([h_low, h_high], hName, f'z{cat}_fit')
                        hists.extend([h, h_high, h_low])
                    elif cat == 'qq':
                        h_high_1D = unroll(h_high, f'z{cat}_fit_high')
                        h_low_1D  = unroll(h_low,  f'z{cat}_fit_low')
                        h = concat([h_high_1D, h_low_1D], f'z{cat}_fit')

                        hists.extend([h, h_high, h_low, h_high_1D, h_low_1D])
                    else:
                        raise ValueError(f'{cat = } is not supported, choose between [ee, mumu, qq]')

                # Skip sample if no valid histograms found
                if not has_valid_hist:
                    LOGGER.warning(f'No valid histogram found for {sample}, skipping file creation')
                    continue

                # Write histograms to ROOT file
                fOut = os.path.join(outDir, sample+'.root')
                f = TFile(fOut, 'RECREATE')
                for hist in hists:
                    hist.Write()
                f.Close()
                LOGGER.info(f'Saved histograms in {fOut}')


######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        # Run histogram processing pipeline
        run(cats, sels)
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
