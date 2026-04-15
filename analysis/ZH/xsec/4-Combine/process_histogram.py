#################################
### IMPORT STANDARD LIBRARIES ###
#################################

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
    cat_multi=True,
    include_sels=True,
    polarization=True,
    description='Histogram Processing Script'
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
    z_decays, H_decays
)
from package.tools.utils import mkdir
from package.tools.process import get_hist, concat



#############################
### SETUP CONFIG SETTINGS ###
#############################

cats, ecm = arg.cat.split('-'), arg.ecm
# Histograms to process
hNames = ['zll_recoil_m']

# Selection strategies to apply
if arg.sels=='':
    sels = ['Baseline', 'Baseline_miss', 'Baseline_sep', 'test']
else:
    sels = arg.sels.split('-')

# Define cross-section scaling factors based on polarization or luminosity
if arg.ILC:  # change fit to ASIMOV -t -1 !!!
    procs_scales = {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919}
elif arg.polL:
    procs_scales = {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263}
elif arg.polR:
    procs_scales = {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018}
else:
    procs_scales  = {}
if procs_scales!={}:
    LOGGER.info('Will scale histograms to ILC cross section')



# Define physics processes and their Higgs decay modes
processes = mk_processes(
    ['ZH', 'WW', 'ZZ', 'Zgamma', 'Rare'],
    H_decays=H_decays+('ZZ_noInv',), ecm=ecm
)

# Background samples: WW, ZZ, Z/gamma -> ll, photon processes, invisible decays
samples_bkg = [
    # ee -> WW
    f'p8_ee_WW_ecm{ecm}',
    f'p8_ee_WW_ee_ecm{ecm}',
    f'p8_ee_WW_mumu_ecm{ecm}',

    # ee -> ZZ
    f'p8_ee_ZZ_ecm{ecm}',

    # ee -> Z/ga -> ll
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}',
    f'wzp6_ee_tautau_ecm{ecm}',

    # e ga -> e Z(ll)
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',

    # ga ga -> ll
    f'wzp6_gaga_ee_60_ecm{ecm}',
    f'wzp6_gaga_mumu_60_ecm{ecm}',
    f'wzp6_gaga_tautau_60_ecm{ecm}',

    # ee -> nu nu Z
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]

# Signal samples: ee -> Z(ll)H(decay) with all Higgs decay modes
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays + ('ZZ_noInv',)]
# Add additional signal modes: ee -> eeH, mumuH, and Hinv
samples_sig.extend([f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}', f'wzp6_ee_ZH_Hinv_ecm{ecm}'])

# Combine signal and background samples
samples = samples_sig + samples_bkg



##########################
### EXECUTION FUNCTION ###
##########################

def run(cats: str,
        sels: str,
        hNames: list[str],
        samples: list[str]
        ) -> None:
    """Process histograms by splitting into high/low control regions and combining."""

    # Process each final state category
    for cat in cats:
        LOGGER.info(f'Processing histograms for {cat}')
        inDir = loc.get('HIST_PREPROCESSED', cat, ecm)

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
                LOGGER.info(f'Processing {sample}')
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
                    h_high.SetName(h_high.GetName()+'_high')
                    h_low.SetName(h_low.GetName()+'_low')

                    has_valid_hist = True
                    # Concatenate high and low region histograms
                    h = concat([h_low, h_high], hName)

                    hists.extend([h, h_high, h_low])

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
    # Run histogram processing pipeline
    run(cats, sels, hNames, samples)
    # Print execution time
    timer(t)
