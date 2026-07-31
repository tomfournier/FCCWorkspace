#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import time, ROOT

t = time.time()


MASS_SYSTEMATICS = ['BES', 'SQRTS', 'LEPSCALE_MU', 'LEPSCALE_EL']
FIT_COLORS = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen + 1]
BREAKDOWN_COLORS = [
    ROOT.kRed + 1,
    ROOT.kBlue - 4,
    ROOT.kOrange - 3,
    ROOT.kGreen + 3,
    ROOT.kBlack,
]
CHANNEL_LABELS = {
    'ee': 'e^{+}e^{-}',
    'mumu': '#mu^{+}#mu^{-}',
}
CATEGORY_LABELS = {
    'cat0': 'inclusive',
    'cat1': 'central-central',
    'cat2': 'central-forward',
    'cat3': 'forward-forward',
    'combined': 'combined',
}


########################
### ARGUMENT PARSING ###
########################

from package.config import timer
from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sel=True,
    mass_fit=True,
    do_fit=True,
    description='Fit mass script'
)
arg = parse_args(parser)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from package.userConfig import loc
loc.set_default_type('Path')

from package.func.fit import (
    breakDown,
    combineCards,
    run_mass_pipeline
)
from package.plots.fit import (
    plot_mass_multiple as plotMultiple,
)



################################
### PARAMETERS CONFIGURATION ###
################################

ecm, sel, tag, mode = arg.ecm, arg.sel, arg.tag, arg.mode
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)

topLeft = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
topRight = f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}'

combineDir = loc.get('COMBINEDIR', '', ecm, sel)  # Baseline combine directory
outDir = loc.get('COMBINE', '', ecm, sel)

combineOptions, suffix = [], ''
freezeParameters, setParameters = [], []

if arg.statOnly:
    suffix = '_stat'
    freezeParameters.extend(['BES_ecm240', 'SQRTS_ecm240', 'LEPSCALE_MU_ecm240', 'LEPSCALE_EL_ecm240'])
    freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365'])
    freezeParameters.extend(['BES_ecm365', 'SQRTS_ecm365', 'LEPSCALE_MU_ecm365', 'LEPSCALE_EL_ecm365'])

if arg.freezeBkg:
    suffix = f'_freezeBkg{suffix}'
    freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365'])

if arg.noBkg:
    suffix = f'_noBkg{suffix}'
    freezeParameters.extend(['bkg_norm_mumu_ecm240', 'bkg_norm_ee_ecm240', 'bkg_norm_mumu_ecm365', 'bkg_norm_ee_ecm365', 'r'])  # r to be frozen to avoid issues with no-bkg fit
    setParameters.extend(['bkg_norm_mumu_ecm240=0', 'bkg_norm_ee_ecm240=0', 'bkg_norm_mumu_ecm365=0', 'bkg_norm_ee_ecm365=0'])


# Systematic variations, unfreeze them
systs = [f'{n}_ecm{e}' for n in ['BES', 'SQRTS', 'LEPSCALE_MU', 'LEPSCALE_EL'] for e in [240, 365]]

freezeParameters.extend(systs)


# 240 + 365
if arg.combine:

    fit_tag = 'mumu_ee_combined_categorized'
    label = '#mu^{#plus}#mu^{#minus}+e^{#plus}e^{#minus}, categorized'

    tag = 'combined_ecm_240_365'
    topRight = '#sqrt{s} = 240/365 GeV, 10.8/3.12 ab^{#minus1}'

    base_dir = loc.get('COMBINE_BASE', '', '', sel)

    combineDir     = loc.get('COMBINEDIR', '', '',  sel)
    combineDir_240 = loc.get('COMBINEDIR', '', 240, sel)
    combineDir_365 = loc.get('COMBINEDIR', '', 365, sel)

    outDir_240 = loc.get('COMBINE', '', 240, sel)
    outDir_365 = loc.get('COMBINE', '', 365, sel)



########################
### HELPER FUNCTIONS ###
########################

def _combine_options_from_flags() -> list[str]:
    options = []
    if len(freezeParameters) > 0:
        options.extend(['--freezeParameters', ','.join(freezeParameters)])
    if len(setParameters) > 0:
        options.extend(['--setParameters', ','.join(setParameters)])
    return options



##########################
### EXECUTION FUNCTION ###
##########################

def main():

    if arg.doBreakdown:
        breakDown(outDir, topRight)

    if arg.doSummary:
        modes = ['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD']
        plotMultiple([outDir / f'{mode}/mumu_combined_ecm240' for mode in modes],
                     ['IDEA', 'IDEA perfect resolution', 'IDEA 3T', 'IDEA CLD silicon tracker'],
                     outDir / 'modes_mumu',
                     124.99, 125.01,
                     legLabel='Muon final state Z(#mu^{#plus}#mu^{#minus})H (stat. + syst.)')
        plotMultiple([outDir / f'{mode}/mumu_combined_ecm240' for mode in modes],
                     ['IDEA', 'IDEA perfect resolution', 'IDEA 3T', 'IDEA CLD silicon tracker'],
                     outDir / 'IDEA_IDEAL_2T_3T_CLD_mumu_stat',
                     124.99, 125.01,
                     legLabel='Muon final state Z(#mu^{#plus}#mu^{#minus})H (stat. only)',
                     forceStat=[True, True, True, True])
        plotMultiple([outDir / 'IDEA/lumi10p8/mumu_ee_combined_categorized_ecm240/',
                      outDir / 'IDEA/lumi10p8/mumu_ee_combined_categorized_ecm240/'],
                     ['Statistical', 'Statistical+systematic'],
                     outDir / 'IDEA_stat_syst',
                     124.995, 125.005,
                     legLabel='Combined muon and electron final states',
                     forceStat=[True, False])

    combineOptions = _combine_options_from_flags()

    tag_suffix       = f'_{arg.tag}' if arg.tag else ''
    selected_tag     = f'{arg.cat}{tag_suffix}_ecm{ecm}'
    selected_label   = f'{CHANNEL_LABELS.get(arg.cat, arg.cat)}, {CATEGORY_LABELS.get(arg.tag, arg.tag)}'
    selected_run_dir = f'{combineDir}/{selected_tag}'
    if arg.tag == 'combined':
        combineCards(
            selected_run_dir,
            [
                f'{combineDir}/{arg.cat}_cat1_ecm{ecm}/datacard.txt',
                f'{combineDir}/{arg.cat}_cat2_ecm{ecm}/datacard.txt',
                f'{combineDir}/{arg.cat}_cat3_ecm{ecm}/datacard.txt',
            ],
        )
    run_mass_pipeline(
        selected_run_dir,
        f'{outDir}/{selected_tag}/',
        selected_label,
        combineOptions,
        top_right=topRight,
        suffix=suffix,
    )

    ############### MUON+ELECTRON
    combined_channels = [
        (
            f'mumu_ee_combined_inclusive_ecm{ecm}',
            '#mu^{#plus}#mu^{-}+e^{#plus}e^{#minus}, inclusive',
            [
                f'{combineDir}/mumu_cat0_ecm{ecm}/datacard.txt',
                f'{combineDir}/ee_cat0_ecm{ecm}/datacard.txt',
            ],
        ),
        (
            f'mumu_ee_combined_categorized_ecm{ecm}',
            '#mu^{#plus}#mu^{-}+e^{#plus}e^{#minus}, categorized',
            [
                f'{combineDir}/mumu_combined_ecm{ecm}/datacard.txt',
                f'{combineDir}/ee_combined_ecm{ecm}/datacard.txt',
            ],
        ),
    ]
    for tag_name, label, cards in combined_channels:
        combineCards(f'{combineDir}/{tag_name}', cards)
        run_mass_pipeline(
            f'{combineDir}/{tag_name}',
            f'{outDir}/{tag_name}/',
            label,
            combineOptions,
            top_right=topRight,
            suffix=suffix,
        )

    plotMultiple([f'{outDir}/mumu_combined_ecm{ecm}/',
                  f'{outDir}/ee_combined_ecm{ecm}/',
                  f'{outDir}/mumu_ee_combined_categorized_ecm{ecm}/'],
                 ['#mu^{#plus}#mu^{-}', 'e^{#plus}e^{#minus}', '#mu^{#plus}#mu^{-} + e^{#plus}e^{#minus}'],
                 f'{outDir}/mumu_ee_combined_categorized_ecm{ecm}', 124.99, 125.01)


    # 240 + 365
    if arg.combine:
        combineCards(combineDir, [combineDir_240+'/datacard.txt', combineDir_365+'/datacard.txt'])
        run_mass_pipeline(combineDir, outDir, 'Combination', combineOptions, top_right=topRight, suffix=suffix)
        plotMultiple([outDir_240, outDir_365, outDir],
                     ['#sqrt{s} = 240 GeV', '#sqrt{s} = 365 GeV', 'Combination'],
                     outDir, 124.98, 125.02)


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
