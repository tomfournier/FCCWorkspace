#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import time, json, ROOT

t = time.time()


########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sel=True,
    mass_fit=True,
    description='Fit mass script'
)
arg = parse_args(parser)
set_log(arg)

LOGGER = get_logger(__name__)



#########################################################
## IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
#########################################################

from package.config import param_config, timer
from package.userConfig import loc
loc.set_default_type('Path')

from definitions import SIGNAL_MODELS, make_var_dict

from package.func.fit import build_params_from_spec, build_pdf_from_spec, get_hist
from package.plots.fit import plot_decomposition, plot_fit, plot_fit_all



################################
### PARAMETERS CONFIGURATION ###
################################

flavor, ecm, sel, cat, _ = arg.cat, arg.ecm, arg.sel, arg.category, arg.tag
hName = 'zll_recoil_m_cat'

label = f'#el^+#el^-, category {cat}'.replace('#el', '#mu' if flavor=='mumu' else 'e')
topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
topRight = '#sqrt{s} = ECM GeV, 1 ab^{-1}'.replace('ECM', ecm)

inDir  = loc.get('HIST_PROCESSED', flavor, ecm, sel)
outDir = loc.get('PARAMETRIC',     flavor, ecm, sel)  # Should define the output file
outDir.mkdir(exist_ok=True, parents=True)
# subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])

signal_spec = SIGNAL_MODELS['parametrize_signal']
procs = signal_spec['processes'](flavor, ecm)
mHs = signal_spec['masses']

if cat == 0: cat_idx = (0, 5)
else:        cat_idx = (cat, cat)
nBins = arg.nBins  # Total number of bins, for plotting



##########################
### EXECUTION FUNCTION ###
##########################

def main():

    mrec = ROOT.RooRealVar('zll_recoil_m', 'Recoil mass [GeV]', 125, arg.recoilMin, arg.recoilMax)
    MH   = ROOT.RooRealVar('MH', 'Higgs mass (GeV)', 125, 124.95, 125.05)  # name Higgs mass as MH to be compatible with combine

    # Define output workspace, then import MH and mrec
    workspace = ROOT.RooWorkspace('w', 'workspace')
    workspace.Import(MH); workspace.Import(mrec)

    # Get the histogram from the nominal mass (125 GeV)
    hist_nom = get_hist(f'{flavor}_{hName}', inDir, [procs[0]], 'norm', cat_idx,
                        1, 1, '', False)
    yMax  = hist_nom.Rebin(int(hist_nom.GetNbinsX() / nBins)).GetMaximum()
    yield_nom = hist_nom.Integral()

    # Define a dictionary to store the values of the parameters
    var_dict = make_var_dict(signal_spec, extra=('yield', 'mH'))

    # Loop over mH = 124.95, 125, 125.05 GeV
    for mH, proc in zip(mHs, procs):

        # Replace the '.' by 'p' to name the mass categories
        mH_label = f'{mH:.3f}'.replace('.', 'p')
        LOGGER.info(f'Doing {mH_label} mass category')

        hist_zh = get_hist(f'{flavor}_{hName}', inDir, proc, mH_label, cat_idx,
                           1, yield_nom, '', arg.normYields)
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', 'rdh_zh',
                                  ROOT.RooArgList(mrec), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        params = build_params_from_spec(param_config[ecm][flavor][cat],
                                        signal_spec, mH, mH_label)

        sig_fit, _, sig_norm = build_pdf_from_spec(mrec, params, yield_zh, mH_label, signal_spec)
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))

        plot_fit(outDir, mrec, rdh_zh, sig_fit, mH_label, yMax, label, nBins, topLeft, topRight)
        plot_decomposition(outDir, workspace, label, mH_label, yield_zh, signal_spec)

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

        # Store the values of the parameters to write them in a json file later
        for k in var_dict.keys():
            if k == 'mH': var_dict[k].append(mH)
            else: var_dict[k].append(params[k].getVal())
        var_dict['yield'].append(sig_norm.getVal())

    plot_fit_all(outDir, workspace, mHs, yield_zh, yMax, label, topLeft, topRight)

    # Export values
    with open(f'{outDir}/coeff.json', 'w') as file:
        file.write(json.dumps(var_dict, indent=4))

    # Delete workspaces to avoid segfault
    del workspace


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Fumili2')
    ROOT.Math.MinimizerOptions.PrintDefault('Minuit2')
    ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200)

    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
