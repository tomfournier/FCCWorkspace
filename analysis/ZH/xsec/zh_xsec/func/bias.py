'''Pseudo-data generation and result tools for bias tests.

Provides:
- Target-channel scaling: `scaling()`.
- Datacard creation with perturbed asimov data: `make_datacard()`.
- Histogram extraction from datacards: `hist_from_datacard()`.
- Bias summary table: `bias_to_txt()`.

Conventions:
- The target channel is the set of signal samples named `H{target}`, scaled so that the total signal cross-section changes by `variation`:
  scale_target = 1 + (variation - 1) * xsec_tot / xsec_target.
- The perturbation is only injected in the asimov data of the datacard, the templates stay nominal.

Lazy Imports:
- hist and uproot are lazy-loaded only when needed
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import hist

from datacard import do_combine
from tools.process import getMetaInfo
from logger import get_logger

LOGGER = get_logger(__name__)



########################
### HELPER FUNCTIONS ###
########################

def scaling(
    sig_procs: dict[str, list[str]],
    target: str,
    variation: float,
    scale: float = 1.0
     ) -> dict[str, float]:
    '''Per-sample signal scales for which the total signal changes by `variation`.

    The target samples are the ones of the `H{target}` decay (e.g., `_Hbb_ecm240`).

    Args:
        sig_procs: Signal process -> samples.
        target: Target Higgs decay channel to perturb.
        variation: Total signal factor (e.g., 1.05 for +5%).
        scale: Extra scale applied to all signal samples (e.g., polarisation).

    Returns:
        Sample name -> scale, the target channel samples carrying the perturbation.
    '''
    xsecs = {s: getMetaInfo(s, rmww=True) for samples in sig_procs.values() for s in samples}
    xsec_tot    = sum(xsecs.values())
    
    is_target = lambda s: f'_H{target}_ecm' in s
    xsec_target = sum(x for s, x in xsecs.items() if is_target(s))
    scale_target = 1.0 + xsec_tot * (variation - 1.0) / xsec_target

    LOGGER.debug(f'Perturbation: {(variation-1)*100:.2f} %, scale of {target}: {scale_target:.3f}')
    LOGGER.debug(f'Target xsec: {xsec_target:.3e} pb-1')

    return {s: scale * (scale_target if is_target(s) else 1.0) for s in xsecs}



######################
### MAIN FUNCTIONS ###
######################

def make_datacard(
    inputDir: str,
    outputDir: str,
    target: str,
    variation: float,
    hist_names: list[str],
    categories: list[str],
    sig_procs: dict[str, list[str]],
    bkg_procs: dict[str, list[str]],
    systs_procs: dict[str, list[str]],
    proc_scales: dict[str, float] | None = None,
    freeze_bkg: bool = False,
    float_bkg: bool = False,
    rebin: int = 1,
    intLumi: float = 1,
    mc_stats: bool = False,
     ) -> None:
    '''Write the datacard of a bias test with `do_combine`, injecting `variation` in the asimov data.

    Args:
        inputDir: Directory with the processed histograms.
        outputDir: Output directory of the datacard (`datacard_{target}.txt/.root`).
        target: Higgs decay channel to perturb.
        variation: Total signal factor (e.g., 1.05 for +5%).
        hist_names: Histogram name of each category.
        categories: Datacard categories.
        sig_procs: Signal process -> samples.
        bkg_procs: Background process -> samples.
        proc_scales: Extra scale per process (e.g., polarisation), applied to the asimov data.
        freeze: Only keep the (negligible) signal normalisation uncertainty.
        float_bkg: Treat backgrounds as signal processes, scaled by the signal strength.
        rebin: Histogram rebinning factor.
        intLumi: Luminosity scaling factor.
        mc_stats: Include MC statistical uncertainties.
    '''
    proc_scales = proc_scales or {}

    if freeze_bkg and float_bkg:
        raise ValueError("Can't freeze and float the backgrounds at the same time, choose one option")

    scales  = scaling(sig_procs, target, variation, proc_scales.get('ZH', 1.0))
    scales |= {s: proc_scales.get(p, 1.0) for p, samples in bkg_procs.items() for s in samples}

    if float_bkg:
        sig_procs, bkg_procs = sig_procs | bkg_procs, {}

    if freeze_bkg:
        systs = {f'{p}_norm': {'type': 'lnN', 'value': 1.000000005, 'procs': [p]} for p in sig_procs}
    else:
        systs = {f'{p}_norm': {'type': 'lnN', 'value': 1.01, 'procs': [p]} for p in bkg_procs}

    do_combine(inputDir, outputDir, hist_names, categories, sig_procs, bkg_procs,
               systs, systs_procs, rebin, intLumi, scales, True, mc_stats,
               f'_{target}')


def bias_to_txt(
    outputdir: str,
    bias: list[float],
    targets: list[str],
    header: str = 'Decay modes',
    col_w: int = 15
     ) -> None:
    '''Format and save bias results to a text file with fixed-width columns.'''

    txt = [f"{header:<{col_w}}" + ''.join(f'{d:<{col_w}}'    for d in targets),  # Decay modes header
           '-' * col_w * (len(targets) + 1),                                          # Line separator
           f"{'Bias':<{col_w}}" + ''.join(f'{b:<{col_w}.3f}' for b in bias),     # Biases with 3 digits precision
           f"{'Bias':<{col_w}}" + ''.join(f'{b:<{col_w}.2f}' for b in bias)]     # Biases with 2 digits precision

    with open(f'{outputdir}/bias_results.txt', 'w') as f:
        f.write('\n'.join(txt) + '\n')
    LOGGER.info(f'Bias saved at {outputdir}/bias_results.txt')

    return None
