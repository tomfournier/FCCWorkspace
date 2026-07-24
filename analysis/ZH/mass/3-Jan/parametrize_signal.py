
import json, argparse, subprocess,  ROOT

from package.config import param_config
from package.userConfig import loc
loc.set_default_type('Path')

from package.func.fit import get_hist, make_params
from package.plots.fit import plot_fit, plot_fit_all, decomposition_plot

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)



parser = argparse.ArgumentParser()
parser.add_argument('--flavor', type=str, help='Flavor (mumu or ee)', default='mumu')
parser.add_argument('--mode', type=str, help='Detector mode', choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim',
                                                                       'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'], default='IDEA')
parser.add_argument('--cat', type=int, help='Category (0, 1, 2 or 3) (default 0)', choices=[0, 1, 2, 3], default=0)
parser.add_argument('--ecm', type=int, help='Center-of-mass', choices=[240, 365], default=240)
parser.add_argument('--sel', type=str, help='Selection', default='Baseline')
parser.add_argument('--tag', type=str, help='Analysis tag for versioning, optional', default='')
parser.add_argument('--recoilMin', type=int, help='Lower mass limit for fit (default 120)',    default=120)
parser.add_argument('--recoilMax', type=int, help='Upper mass limit for fit (default 140)',    default=140)
parser.add_argument('--nBins',     type=int, help='Number of bins for plotting (default 250)', default=250)
parser.add_argument('--normYield', type=bool, help='Normalize the histograms (default True)', action='store_true', default=True)
args = parser.parse_args()



################################
### PARAMETERS CONFIGURATION ###
################################

flavor, ecm, sel, cat, _ = args.flavor, args.ecm, args.sel, args.cat, args.tag
flavorLabel = '#mu^{#plus}#mu^{#minus}' if flavor == 'mumu' else 'e^{#plus}e^{#minus}'

topRight = f'#sqrt{{s}} = {ecm} GeV, 1 ab^{{#minus1}}'
topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
label  = f'{flavorLabel}, category {cat}'
normYields = args.normYield
# inDir  = loc.get('HIST_PROCESSED', flavor, ecm, sel)
outDir = loc.get('PARAMETRIC', flavor, ecm, sel)  # Should define the output file

outDir.mkdir(exist_ok=True, parents=True)
subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])

hName = 'zll_recoil_m_cat'

if cat == 0: cat_idx_min, cat_idx_max = 0, 5
else: cat_idx_min, cat_idx_max = cat, cat

nBins = args.nBins  # total number of bins, for plotting



##########################
### EXECUTION FUNCTION ###
##########################

def main():

    recoilmass = ROOT.RooRealVar('zll_recoil_m', 'Recoil mass (GeV)', 125, args.recoilMin, args.recoilMax)
    MH = ROOT.RooRealVar('MH', 'Higgs mass (GeV)', 125, 124.95, 125.05)  # name Higgs mass as MH to be compatible with combine

    # Define temporary output workspace
    workspace = ROOT.RooWorkspace('w_tmp', 'workspace')

    workspace.Import(recoilmass)
    workspace.Import(MH)

    mHs = [124.95, 125.0, 125.05]
    procs = [f'wzp6_ee_{flavor}H_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-lower-50MeV_ecm{ecm}', f'wzp6_ee_{flavor}H_mH-higher-50MeV_ecm{ecm}']
    recoilmass = workspace.var('zll_recoil_m')
    var_list = ['mean_cb0', 'mean_cb1', 'sigma_cb', 'mean_gt', 'mean_gt1', 'sigma_gt',
                'alpha_1', 'alpha_2', 'n_1', 'n_2', 'cb_1', 'cb_2', 'yield', 'mH']
    var_dict = {k:[] for k in  var_list}

    hist_norm = get_hist(f'{flavor}_{hName}', [procs[0]], 'norm', cat_idx_min,
                         cat_idx_max, 1, 1, '', False)
    yield_norm = hist_norm.Integral()

    tmp = hist_norm.Clone()
    print(hist_norm.GetNbinsX() / nBins)
    tmp  = tmp.Rebin(int(hist_norm.GetNbinsX() / nBins))
    yMax = tmp.GetMaximum()


    for mH, proc in zip(mHs, procs):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        print(f'Do {mH = :.3f}')

        hist_zh = get_hist(f'{flavor}_{hName}', [proc], mH_label, cat_idx_min,
                           cat_idx_max, 1, yield_norm, '', normYields)
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        params = make_params(flavor, ecm, cat, mH_label, param_config)

        # construct the 2CBG and perform the fit: pdf = cb_1*cbs_1 + cb_2*cbs_2 + gauss (cb_1 and cb_2 are the fractions, floating)
        cbs_1 = ROOT.RooCBShape(f'CrystallBall_1_{mH_label}', 'CrystallBall_1', recoilmass, params['mean_cb'], params['sigma_cb'], params['alpha_1'], params['n_1'])  # 1st CrystallBall
        cbs_2 = ROOT.RooCBShape(f'CrystallBall_2_{mH_label}', 'CrystallBall_2', recoilmass, params['mean_cb'], params['sigma_cb'], params['alpha_2'], params['n_2'])  # 2nd CrystallBall
        gauss = ROOT.RooGaussian(f'gauss_{mH_label}', 'gauss', recoilmass, params['mean_gt'], params['sigma_gt'])  # Gaussian

        sig      = ROOT.RooAddPdf(f'sig_{mH_label}',       '', ROOT.RooArgList(cbs_1, cbs_2, gauss), ROOT.RooArgList(params['cb_1'], params['cb_2']))  # half of both CB functions
        sig_norm = ROOT.RooRealVar(f'sig_{mH_label}_norm', '', yield_zh, 0, 1e8)  # Fix normalization
        sig_fit  = ROOT.RooAddPdf(f'zh_model_{mH_label}',  '', ROOT.RooArgList(sig), ROOT.RooArgList(sig_norm))
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))

        cb1_val = params['cb_1'].getVal()
        cb2_val = params['cb_2'].getVal()

        plot_fit(outDir, recoilmass, rdh_zh, sig_fit, mH_label, yMax, label, nBins, topLeft, topRight)
        decomposition_plot(outDir, workspace, cbs_1, cbs_2, gauss, sig_fit,
                           cb1_val, cb2_val, yield_zh, mH_label, yMax, label, topLeft, topRight)

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

        for k in var_dict.keys():
            if k == 'mH': var_dict[k].append(mH)
            else: var_dict[k].append(params[k].getVal())
        var_dict['yield'].append(sig_norm.getVal())

    plot_fit_all(outDir, workspace, mHs, yield_zh, yMax, label, topLeft, topRight)

    # Export values
    with open(f'{outDir}/coeff.json', 'w'):
        json.dumps(var_dict, indent=4)

    # Delete workspaces to avoid segfault
    del workspace


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    sumw2err = ROOT.kTRUE

    ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Fumili2')
    ROOT.Math.MinimizerOptions.PrintDefault('Minuit2')
    ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200)

    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        # LOGGER.error('Error occured during execution', exc_info=True)
        pass  # Will uncomment it later
    finally:
        # Print execution time
        # timer(t)
        pass
