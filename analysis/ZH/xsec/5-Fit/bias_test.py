################################
### STANDARD LIBRARY IMPORTS ###
################################

import os, time, json

import pandas as pd

from pathlib import Path

# Start timer for performance tracking
t = time.time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_xsec.parsing import create_parser
from logger import get_logger
arg = create_parser('5-Fit', 'bias_test').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

# Load directory configuration and analysis utilities
from zh_xsec.userConfig import loc
from tools.utils import timer
from samples import get_process_dict
from constants import H_DECAYS_WITH_INV, H_DECAYS_FIT
from run import get_extra_args, update_namespace
from fitting import run_cmd
from zh_xsec.func.bias import make_datacard, bias_to_txt



###############################
### CONFIGURATION AND SETUP ###
###############################

if (arg.cat and arg.lep) or (arg.cat and arg.combine):
    raise ValueError("Cannot use '--cat' and '--lep' or '--combine' at the same time")
if arg.lep and arg.combine:
    raise ValueError("Cannot use '--lep' and '--combine' at the same time")
if arg.lep:     arg.cat = 'leptonic'
if arg.combine: arg.cat = 'combined'

# Bundle parsed arguments for configuration
cat, ecm, sel, variation = arg.cat, arg.ecm, arg.sel, arg.pert  # Perturbation factor for bias test
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)    # Integrated luminosity [ab^-1]

# Resolve directory paths for bias test results and input
inputDir   = loc.get('HIST_PROC',  cat, ecm, sel)
outputDir  = loc.get('BIAS_RESULT',     cat, ecm, sel)
bias_fit   = loc.get('BIAS_FIT', cat, ecm, sel)
fit_dir    = loc.get('FIT_RESULT', cat, ecm, sel)
datacard   = loc.get('BIAS_CARD',  cat, ecm, sel)

ENV = os.environ.copy()



#############################
### SETUP CONFIG SETTINGS ###
#############################

mc_stats = arg.mc_stats    # Include MC statistical uncertainties (default: False, assume Poisson)
rebin    = arg.rebin       # Histogram rebinning factor (1 = no rebinning)
intLumi  = arg.intLumi     # Luminosity scaling factor for normalization
tot      = arg.tot         # Use all Z decays for the signal
freeze_bkg = arg.freeze    # Freeze the backgrounds for the fit
float_bkg  = arg.float     # Make the background float during the fit
systs_procs = {}           # Systematics parameters (to be implemented)

# Select cross-section scale: polarization or ILC configuration
scale = 'polL' if arg.polL else ('polR' if arg.polR else ('ILC' if arg.ILC else ''))
# Per-process scale factors (applied to the pseudo-data only)
proc_scales = {'ILC':  {'ZH': 1.048, 'WW': 0.971, 'ZZ': 0.939, 'Zgamma': 0.919},
               'polL': {'ZH': 1.554, 'WW': 2.166, 'ZZ': 1.330, 'Zgamma': 1.263},
               'polR': {'ZH': 1.047, 'WW': 0.219, 'ZZ': 1.011, 'Zgamma': 1.018}
               }.get(scale, {})

sig_key = 'ZH' if tot else f'Z{cat}H'
sig_procs_dict = {'*':   get_process_dict([sig_key], ecm),
                  'inv': get_process_dict([sig_key], ecm, h_decays=H_DECAYS_FIT)}

bkgs = ['WW', 'ZZ', 'Zgamma', 'Rare'] + (['tt'] if cat=='qq' and ecm==365 else [])
bkg_procs = get_process_dict(bkgs, ecm)

# Histograms per region (high/low recoil mass) for the leptonic and hadronic channels
hist_names = {
    'lep': ['zll_recoil_m_fit_high', 'zll_recoil_m_fit_low'],
    'had': ['zqq_m_recoil_m_mva_fit_high_1D', 'zqq_m_recoil_m_mva_fit_low_1D'],
}.get('had' if cat=='qq' else 'lep', [])
categories = [f'z_{cat}_high', f'z_{cat}_low']



#################
### FUNCTIONS ###
#################

def run_fit(target: str) -> float:
    """Generate pseudodata and run pseudodata fit for a Higgs decay mode.

    Builds the datacard with do_combine, then runs fit via subprocess.
    """
    from tools.utils import load_json
    if not arg.combine:
        sig_procs = sig_procs_dict['inv' if target=='inv' else '*']
        make_datacard(inputDir, datacard, target, variation, hist_names, categories,
                      sig_procs, bkg_procs, systs_procs, proc_scales, freeze_bkg,
                      float_bkg, rebin, intLumi, mc_stats)

    fit_args = update_namespace(arg, cat=None if (arg.lep or arg.combine) else arg.cat, bias=True,
                                ecm=str(arg.ecm), sels=arg.sel, timer=False, print=False, target=target)
    cmd = ['python', 'fit.py'] + get_extra_args(fit_args, {'directory': '5-Fit', 'script': 'fit'}, create_parser)
    run_cmd(cmd, None, Path(__file__).parent, ENV)

    # Extract and return the fitted signal strength
    return load_json(f'{bias_fit}/results_fit_{target}.json')['r']['best_fit']



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    outputDir.mkdir()
    
    biases = []
    targets = [h for h in H_DECAYS_WITH_INV if (h in arg.decays or 'all' in arg.decays)]

    if arg.skip_fit:
        df = pd.read_csv(f'{outputDir}/bias_results.csv')
        biases = list(df['bias'])
    else:
        for target in targets:
            LOGGER.info(f'Running fit for {target} channel')

            # Run fit and compute bias as percentage difference from prior
            biases.append(100 * (run_fit(target) - variation))
            LOGGER.info(f'Bias obtained: {biases[-1]:.3f}\n')

            # Generate pseudodata ratio plots (only for single channel, not combined)
            if not arg.combine and arg.plot_pseudo:
                LOGGER.info('Making plots for pseudo-signal')

                from zh_xsec.plots.plotting import PseudoRatio
                procs = list(sig_procs_dict['inv' if target=='inv' else '*']) + list(bkg_procs)
                args = {'inputdir': datacard, 'outputdir': outputDir, 'target': target,
                        'procs': procs, 'ecm': ecm, 'lumi': lumi, 'pert': variation}
                # Create plots for high and low selection regions
                for sel_suffix in ['high', 'low']:
                    PseudoRatio(**args, cat=f'{cat}_{sel_suffix}', sel=f'{sel}_{sel_suffix}')

        # Save bias results to CSV file
        LOGGER.debug('Saving bias in a .csv file')
        df = pd.DataFrame({'mode':targets, 'bias':biases})
        df.to_csv(f'{outputDir}/bias_results.csv', index=False)
        LOGGER.debug(f'Bias saved at {outputDir}/bias_results.csv')

    # Generate bias summary plots
    from zh_xsec.plots.plotting import Bias, BiasNorm, BiasSorted
    kwargs = {'df': df, 'nomDir': fit_dir, 'outputdir': outputDir,
              'h_decays': targets, 'ecm': ecm, 'lumi': lumi}
    Bias(**kwargs); BiasNorm(**kwargs); BiasSorted(**kwargs)

    # Save bias results to formatted text file
    bias_to_txt(outputDir, biases, targets)

    return None



######################
### CODE EXECUTION ###
######################

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
