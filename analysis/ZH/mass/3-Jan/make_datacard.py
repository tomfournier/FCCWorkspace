
import os, array, argparse, subprocess, ROOT

import numpy as np

import definitions as defs
import package.plots.root.plotter as plotter
from package.tools.process import getHist
from package.userConfig import loc
loc.set_default_type('Path')
from package.plots.fit import (
    fit_plot,
    plot_signal,
    plot_syst_dist,
    plot_params_vs_mh,
    plot_decomposition,
    plot_fit_with_pull
)

from package.func.fit import (
    get_hist,
    build_2cbg_pdf,
    build_background_pdf,
    build_pdf_from_spec,
    make_p,
    make_unc_import
)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)


parser = argparse.ArgumentParser()
parser.add_argument('--flavor',    type=str,   help='Flavor (mumu or ee)', default='mumu')
parser.add_argument('--mode',      type=str,   help='Detector mode', choices=['IDEA', 'IDEA_MC', 'IDEA_3T', 'CLD', 'CLD_FullSim', 'IDEA_noBES', 'IDEA_2E', 'IDEA_BES6pct'], default='IDEA')
parser.add_argument('--cat',       type=int,   help='Category (0, 1, 2 or 3)', choices=[0, 1, 2, 3], default=0)
parser.add_argument('--lumi',      type=float, help='Luminosity scale', default=1.0)
parser.add_argument('--ecm',       type=str,   help='Center-of-mass', choices=['240', '365'],     default='240')
parser.add_argument('--sel',       type=str,   help='Selection', default='Baseline')
parser.add_argument('--tag',       type=str,   help='Analysis tag for versioning, optional',      default='')
parser.add_argument('--recoilMin', type=int,   help='Lower mass value for the fit (default 120)', default=120)
parser.add_argument('--recoilMax', type=int,   help='Upper mass value for the fit (default 140)', default=140)
parser.add_argument('--nBins',     type=int,   help='Number of bins for the plots (default 250)', default=250)
parser.add_argument('--doSyst',    type=bool,  help='Include systematic uncertainties in th fit (default True)', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()



flavor, ecm, sel = args.flavor, args.ecm, args.sel
cat, tag, mode = args.cat, args.tag, args.mode
flavorLabel = '#mu^{#plus}#mu^{#minus}' if flavor == 'mumu' else 'e^{#plus}e^{#minus}'
lumiScale = args.lumi

# re-normalize lumis, by default histograms are scaled to 10.8 and 3.12 ab-1
if ecm == 240:
    lumiScale /= 10.8
elif ecm == 365:
    lumiScale /= 3.12
else:
    raise ValueError(f'{ecm = } is not supported, choose between [240, 365]')
lumiStr   = str(int(args.lumi)) if args.lumi.is_integer() else str(args.lumi)
lumiLabel = lumiStr.replace('.', 'p')

topRight = f'#sqrt{{s}} = {ecm} GeV, {lumiStr} ab^{{-1}}'
topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
label    = f'{flavorLabel}, category {cat}'
inDir    = loc.get('HIST_PROCESSED', flavor, ecm, sel)
outDir   = loc.get('DATACARD',       flavor, ecm, sel)
runDir   = loc.get('RUNDIR',         flavor, ecm, sel)

outDir.mkdir(exist_ok=True, parents=True)
runDir.mkdir(exist_ok=True, parents=True)
subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])

parId = f'{flavor}_cat{cat}_ecm{ecm}'
hName = 'zll_recoil_m_cat'
signal_spec = defs.get_signal_model('make_datacard')
background_spec = defs.BACKGROUND_MODEL
if cat == 0: cat_idx_min, cat_idx_max = 0,   5
else:        cat_idx_min, cat_idx_max = cat, cat

nBins = args.nBins  # total number of bins, for plotting
recoilMin, recoilMax = args.recoilMin, args.recoilMax


# Recoil mass plot settings
cfg = {

    'logX':             False,
    'logY':             False,

    'xmin':             120,
    'xmax':             140,
    'ymin':             0,
    'ymax':             1,

    'xtitle':           'm_{recoil} [GeV]',
    'ytitle':           'Events',

    'topLeft':          topLeft,
    'topRight':         topRight,

    'ratiofraction':    0.3,
    'ytitleR':         'Pull',
    'yminR':           -3.5,
    'ymaxR':            3.5
}



def doSignal(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        pdf_sigs: list,
        w_tmp: ROOT.RooWorkspace,
        label: str,
        lumiScale: float | int,
        cat_idx_min: int,
        cat_idx_max: int,
        nBins: int,
        normYields: bool = True):

    global h_obs, yield_nom, yMax

    mHs = signal_spec['masses']
    procs = defs.get_signal_processes('make_datacard', flavor, ecm)
    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')

    hist_norm = get_hist(f'{flavor}_{hName}', inDir, [procs[0]], 'norm',
                         cat_idx_min, cat_idx_max, lumiScale, 1, '', False)
    yield_nom = hist_norm.Integral()

    tmp = hist_norm.Clone()
    yMax = 1.25 * tmp.Rebin(int(hist_norm.GetNbinsX() / nBins)).GetMaximum()

    garbage = []  # Need to store the variables for memory issues

    ## Build model
    ## linear functions for mean and mean_gt
    ## constants for all the rest

    cats    = ROOT.RooCategory('category', '')           # For each mass bin, define category
    hists   = ROOT.std.map('string, RooDataHist*')()     # Container holding all RooDataHists
    pdf_tot = ROOT.RooSimultaneous('pdf_tot', '', cats)  # Total pdf, containing all the categories

    var_list = ['mean', 'mean_gt', 'sigma', 'sigma_gt', 'mean0', 'mean1', 'mean_gt1',
                'alpha_1', 'alpha_2', 'n_1', 'n_2', 'cb_1', 'cb_2', 'norm', 'mH']
    var_dict = {k:[] for k in var_list}

    for i, mH, proc in enumerate(zip(mHs, procs)):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        print(f'Do {mH = :.3f}')

        print(f'{proc}/{hName}')
        hist_zh = get_hist(f'{flavor}_{hName}', inDir, [proc], mH_label, cat_idx_min, cat_idx_max,
                           lumiScale, yield_nom, '', normYields)
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', '', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        rdh_zh.SetName(f'rdh_zh_{mH_label}')
        yield_zh = rdh_zh.sum(False)

        catIDx = mH_label
        hists.insert(ROOT.std.pair('string, RooDataHist*')(catIDx, rdh_zh))  # Does not work with recent ROOT versions?
        cats.defineType(catIDx, i)

        params = make_p(mH, mH_label, inDir)
        pdf_sig, _, sig_norm = build_pdf_from_spec(recoilmass, params, yield_zh, mH_label, signal_spec)
        pdf_sigs.append(pdf_sig)

        # must store the individual vars for later, to extract the values
        # seems not to work with workspace
        for k in var_dict.keys():
            if k == 'mH': var_dict[k].append(mH)
            else: var_dict[k].append(params[k])
        var_dict['norm'].append(sig_norm)

        garbage.append(pdf_sig)

        pdf_sig.Print()
        pdf_tot.addPdf(pdf_sig, catIDx)

        if mH == 125.0 and h_obs is None: h_obs = hist_zh.Clone('h_obs')  # Take 125.0 GeV to add to observed (need to add background later as well)

    rdh_tot = ROOT.RooDataHist('rdh_tot', '', ROOT.RooArgList(recoilmass), cats, hists)

    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Minimizer('Minimizer', 'simplex'))
    print('****************************')
    print('FIT STATUS')
    print(f'Covariance Quality = {fitRes.covQual()}')
    print(f'Fit status = {fitRes.status()}')
    print('****************************')

    w_tmp.Import(pdf_tot)  # Import after fit, to have fit values in the workspace

    cov = fitRes.covarianceMatrix()
    cov.Print()

    # Plot
    plotter.cfg = cfg
    cfg['ytitle'] = f'Events / {20_000/nBins:.0f} MeV'
    for i, mH, proc in enumerate(zip(mHs, procs)):
        mH_label = f'{mH:.3f}'.replace('.', 'p')

        pdf = pdf_sigs[i]
        rdh_zh = hists[mH_label]

        cfg['ymax'] = yMax
        fit_plot(rdh_zh, pdf, mH_label, recoilmass, nBins, label, outDir)

        cfg['ymax'] = 2.5 * yMax
        plotter.cfg = cfg
        sig_fit = plot_decomposition(w_tmp, outDir, label, mH_label, yield_nom)

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)

        vals = {k:[v.getVal() if k!='mH' else v for v in values] for k, values in var_dict.items()}


    ##################################
    # plot all fitted signals
    ##################################
    cfg['xmin'] = 124
    cfg['xmax'] = 128
    cfg['ymax'] = 2.5 * np.average(yMax)
    plotter.cfg = cfg

    plot_signal(w_tmp, mHs, outDir, label, yield_nom, pdf_sigs)

    # Make splines, to connect the fit parameters a function of the Higgs mass
    # Plot them afterwards
    splines = {k:ROOT.RooSpline1D(f'spline_{k}', f'spline_{k}', MH, len(vals['mH']),
                                  array.array('d', vals['mH']),
                                  array.array('d', vals[k]))
               for k in vals.keys() if k!='mH'}

    form_mean = ROOT.RooFormulaVar('form_mean', '@0*@1 + @2', ROOT.RooArgList(params['mean0'], MH, params['mean1']))
    form_mean.Print()

    # TO DO for other parameters
    for param, spline in splines.items():
        plot_params_vs_mh(MH, outDir, param, vals, spline)

    # Was getattr(w_tmp, 'import')(spline_<variable>)
    # Should test if it works
    for spl in splines.values(): w_tmp.Import(spl)

    return hist_norm, w_tmp




def doBackgrounds(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        w_tmp: ROOT.RooWorkspace,
        label: str,
        lumiScale: float | int,
        cat_idx_min: str,
        cat_idx_max: str,
        nBins: int,
         ):

    global h_obs

    recoilmass = w_tmp.var('zll_recoil_m')
    hist_bkg = None

    procs = defs.get_background_processes(flavor, ecm)

    for proc in procs:
        hist = getHist(f'{flavor}_{hName}', [proc])
        hist.Scale(lumiScale)
        hist = hist.ProjectionX(f'hist_{proc}', cat_idx_min, cat_idx_max)

        # Add to total background
        if hist_bkg is None: hist_bkg = hist
        else: hist_bkg.Add(hist)

        # Add to observed
        if h_obs is None: h_obs = hist.Clone('h_obs')
        else: h_obs.Add(hist)

    hist_bkg.SetName('total_bkg')
    rdh_bkg   = ROOT.RooDataHist('rdh_bkg', 'rdh_bkg', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_bkg))
    yield_bkg = rdh_bkg.sum(False)

    tmp = hist_bkg.Clone()
    tmp = tmp.Rebin(int(hist_bkg.GetNbinsX() / nBins))
    cfg['ymax'] = 1.5 * tmp.GetMaximum()

    coeffs = {
        coeff['name']: ROOT.RooRealVar(coeff['name'], coeff['title'], coeff['value'], *coeff['range'])
        for coeff in background_spec['coefficients']
    }
    bkg, bkg_norm, bkg_fit = build_background_pdf(
        recoilmass,
        coeffs,
        yield_bkg,
        '',
        background_spec['order'],
        background_spec['name'],
        background_spec['model_name'],
        background_spec['yield_name'],
    )
    bkg_fit.fitTo(rdh_bkg, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(sumw2err))


    ########### PLOTTING ###########
    plotter.cfg = cfg
    plot_fit_with_pull(
        rdh_bkg,
        bkg_fit,
        recoilmass,
        nBins,
        f'{outDir}/fit_bkg',
        label,
        title=None,
        fit_color=ROOT.kRed,
        save_pdf=True,
        param_layout=(0.5, 0.9, 0.9),
    )

    # import background parameterization to the workspace
    bkg_norm.setVal(yield_bkg)
    for coeff in coeffs.values():
        coeff.setConstant(True)

    w_tmp.Import(bkg)
    w_tmp.Import(bkg_norm)

    return hist_bkg, w_tmp



def setup_syst(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        label: str,
        w_tmp: ROOT.RooWorkspace,
        syst: str,
        nBins: int,
        lumiScale: float | int,
        cat_idx_min: int,
        cat_idx_max: int,
        mH: float | int = 125.0,
         ) -> None:

    mH_label = f'{mH:.3f}'.replace('.', 'p')
    recoilmass = w_tmp.var('zll_recoil_m')
    MH = w_tmp.var('MH')
    param_names = defs.get_systematic_parameters(syst)
    spline_map = {name: w_tmp.obj(f'spline_{name}') for name in ['mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2']}
    spline_vals = [spline_map[name] for name in param_names]

    MH.setVal(125.0)
    val_up, val_dw = [], []

    for s in ['Up', 'Down']:
        suffix = f'{mH_label}_{syst}{s}'

        s_ = defs.systematic_hist_suffix(syst, s)
        proc = defs.get_systematic_process(syst, flavor, ecm, s)

        hist_zh  = get_hist(f'{flavor}_{hName}{s_}', [proc], suffix,
                            cat_idx_min, cat_idx_max, lumiScale, f'hist_zh_{suffix}')
        rdh_zh   = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        Vars = {
            name: ROOT.RooRealVar(f'{name}_{suffix}', '', spline_map[name].getVal())
            for name in ['mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2']
        }
        if 'BES' in syst:
            Vars['sigma']    = ROOT.RooRealVar(f'sigma_{suffix}',    '', spline_map['sigma'].getVal(),    0, 5)
            Vars['sigma_gt'] = ROOT.RooRealVar(f'sigma_gt_{suffix}', '', spline_map['sigma_gt'].getVal(), 0, 5)
        elif 'SQRTS' in syst:
            Vars['mean']     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_map['mean'].getVal(),    mH - 5, mH + 5)
            Vars['mean_gt']  = ROOT.RooRealVar(f'mean_gt_{suffix}',  '', spline_map['mean_gt'].getVal(), mH - 5, mH + 5)
        elif 'LEPSCALE' in syst:
            Vars['mean']     = ROOT.RooRealVar(f'mean_{suffix}',     '', spline_map['mean'].getVal(),    mH - 5, mH + 5)

        sig_fit = build_2cbg_pdf(recoilmass, Vars, yield_zh, suffix, signal_spec)
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTrue), ROOT.RooFit.SumW2Error(sumw2err))

        values = [Vars[name].getVal() for name in param_names]
        if s == 'Up':
            val_up = values
        else:
            val_dw = values

        cfg['ymax'] = yMax
        plotter.cfg = cfg
        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            recoilmass,
            nBins,
            f'{outDir}/fit_mH{suffix}',
            label,
            'ZH signal',
            ROOT.kRed
        )

        # Import
        w_tmp.Import(rdh_zh)
        w_tmp.Import(sig_fit)

    # Plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    plot_syst_dist(w_tmp, yield_nom, outDir, syst, mH_label)


    make_unc_import(w_tmp, spline_vals, list(param_names), syst, val_up, val_dw)

    return None



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    h_obs = None  # should hold the data_obs = sum of signal and backgrounds

    recoilmass = ROOT.RooRealVar('zll_recoil_m', 'm_{recoil} [GeV]', 125, recoilMin, recoilMax)
    MH = ROOT.RooRealVar('MH', 'Higgs mass [GeV]', 125, 124.95, 125.05)

    pdf_sigs = []

    # Define temporary output workspace
    w     = ROOT.RooWorkspace('w',     'workspace')  # final workspace for combine
    w_tmp = ROOT.RooWorkspace('w_tmp', 'workspace')

    w_tmp.Import(recoilmass)
    w_tmp.Import(MH)

    _, w_tmp = doSignal(flavor, ecm, hName, outDir, pdf_sigs, w_tmp, label,
                        lumiScale, cat_idx_min, cat_idx_max, nBins)
    _, w_tmp = doBackgrounds(flavor, ecm, hName, outDir, w_tmp, label,
                             lumiScale, cat_idx_min, cat_idx_max, nBins)

    # Build signal model
    spline_mean     = w_tmp.obj('spline_mean')
    spline_sigma    = w_tmp.obj('spline_sigma')
    # spline_yield    = w_tmp.obj('spline_yield')
    spline_alpha_1  = w_tmp.obj('spline_alpha_1')
    spline_alpha_2  = w_tmp.obj('spline_alpha_2')
    spline_n_1      = w_tmp.obj('spline_n_1')
    spline_n_2      = w_tmp.obj('spline_n_2')
    spline_cb_1     = w_tmp.obj('spline_cb_1')
    spline_cb_2     = w_tmp.obj('spline_cb_2')
    spline_mean_gt  = w_tmp.obj('spline_mean_gt')
    spline_sigma_gt = w_tmp.obj('spline_sigma_gt')

    sig_alpha_1 = ROOT.RooFormulaVar('sig_alpha_1', '@0', ROOT.RooArgList(spline_alpha_1))
    sig_alpha_2 = ROOT.RooFormulaVar('sig_alpha_2', '@0', ROOT.RooArgList(spline_alpha_2))
    sig_n_1     = ROOT.RooFormulaVar('sig_n_1',     '@0', ROOT.RooArgList(spline_n_1))
    sig_n_2     = ROOT.RooFormulaVar('sig_n_2',     '@0', ROOT.RooArgList(spline_n_2))
    sig_cb_1    = ROOT.RooFormulaVar('sig_cb_1',    '@0', ROOT.RooArgList(spline_cb_1))
    sig_cb_2    = ROOT.RooFormulaVar('sig_cb_2',    '@0', ROOT.RooArgList(spline_cb_2))
    # sig_norm    = ROOT.RooFormulaVar('sig_norm',    '@0', ROOT.RooArgList(spline_yield))

    if args.doSyst:
        setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'BES',
                   nBins, lumiScale, cat_idx_min, cat_idx_max, 125.0)
        setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'SQRTS',
                   nBins, lumiScale, cat_idx_min, cat_idx_max, 125.0)
        setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'LEPSCALE',
                   nBins, lumiScale, cat_idx_min, cat_idx_max, 125.0)

        # Systematic strenghts
        flav = 'MU' if flavor=='mumu' else ('EL' if flavor=='ee' else flavor)
        # ISR      = ROOT.RooRealVar(f'ISR_ecm{ecm}',             'ISR',      0)         # ISR      uncertainty parameter
        BES      = ROOT.RooRealVar(f'BES_ecm{ecm}',             'BES',      0, -5, 5)  # BES      uncertainty parameter
        SQRTS    = ROOT.RooRealVar(f'SQRTS_ecm{ecm}',           'SQRTS',    0, -5, 5)  # SQRTS    uncertainty parameter
        LEPSCALE = ROOT.RooRealVar(f'LEPSCALE_{flav}_ecm{ecm}', 'LEPSCALE', 0, -5, 5)  # LEPSCALE uncertainty parameter

        # BES
        sigma_BES    = w_tmp.obj('sig_sigma_BES')
        sigma_gt_BES = w_tmp.obj('sig_sigma_gt_BES')

        # SQRTS
        mean_SQRTS    = w_tmp.obj('sig_mean_SQRTS')
        mean_gt_SQRTS = w_tmp.obj('sig_mean_gt_SQRTS')

        # LEPSCALE
        mean_LEPSCALE = w_tmp.obj('sig_mean_LEPSCALE')


        sig_mean     = ROOT.RooFormulaVar('sig_mean',     '@0*(1+@1*@2)*(1+@3*@4)', ROOT.RooArgList(spline_mean,     LEPSCALE, mean_LEPSCALE, SQRTS, mean_SQRTS))
        sig_sigma    = ROOT.RooFormulaVar('sig_sigma',    '@0*(1+@1*@2)',           ROOT.RooArgList(spline_sigma,    BES,      sigma_BES))
        sig_mean_gt  = ROOT.RooFormulaVar('sig_mean_gt',  '@0*(1+@1*@2)',           ROOT.RooArgList(spline_mean_gt,  SQRTS,    mean_SQRTS))
        sig_sigma_gt = ROOT.RooFormulaVar('sig_sigma_gt', '@0*(1+@1*@2)*(1+@3*@4)', ROOT.RooArgList(spline_sigma_gt, BES,      sigma_gt_BES,  SQRTS, mean_gt_SQRTS))

    else:
        sig_mean     = ROOT.RooFormulaVar('sig_mean',     '@0', ROOT.RooArgList(spline_mean))
        sig_sigma    = ROOT.RooFormulaVar('sig_sigma',    '@0', ROOT.RooArgList(spline_sigma))
        sig_mean_gt  = ROOT.RooFormulaVar('sig_mean_gt',  '@0', ROOT.RooArgList(spline_mean_gt))
        sig_sigma_gt = ROOT.RooFormulaVar('sig_sigma_gt', '@0', ROOT.RooArgList(spline_sigma_gt))

    # Construct final signal pdf
    cbs_1 = ROOT.RooCBShape('CrystallBall_1', 'CrystallBall_1', recoilmass, sig_mean, sig_sigma, sig_alpha_1, sig_n_1)
    cbs_2 = ROOT.RooCBShape('CrystallBall_2', 'CrystallBall_2', recoilmass, sig_mean, sig_sigma, sig_alpha_2, sig_n_2)
    gauss = ROOT.RooGaussian('gauss', 'gauss', recoilmass, sig_mean_gt, sig_sigma_gt)
    sig   = ROOT.RooAddPdf('sig', 'sig', ROOT.RooArgList(cbs_1, cbs_2, gauss), ROOT.RooArgList(sig_cb_1, sig_cb_2))

    w.Import(sig)

    # Construct background model
    bkg_yield = w_tmp.obj('bkg_norm_tmp').getVal()
    bkg_norm  = ROOT.RooRealVar('bkg_norm', 'bkg_norm', bkg_yield)  # Nominal background yield (automatically done by Combine with pdfName_norm, floating)
    bkg_norm.setVal(bkg_yield)  # Not constant!
    bkg = w_tmp.obj('bkg')
    w.Import(bkg, ROOT.RooFit.RenameAllVariablesExcept(parId, 'zll_recoil_m'))

    data_obs = ROOT.RooDataHist('data_obs', 'data_obs', ROOT.RooArgList(recoilmass), ROOT.RooFit.Import(h_obs))
    w.Import(data_obs)

    w.writeToFile(f'{runDir}/datacard.root')
    w.Print()

    poi = ROOT.RooArgSet(MH)
    w.defineSet('POI', poi)

    del w, w_tmp

    if   ecm == 240 and flavor == 'mumu': bkg_id = 1
    elif ecm == 240 and flavor == 'ee':   bkg_id = 2
    elif ecm == 365 and flavor == 'mumu': bkg_id = 3
    elif ecm == 365 and flavor == 'ee':   bkg_id = 4

    # Make datacard
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/datacard_template.txt', 'r') as file:
        dc = file.read()
        dc = dc.replace('$rate_sig', f'{yield_nom}')
        dc = dc.replace('$rate_bkg', f'{bkg_yield}')
        dc = dc.replace('$flavor',   f'{flavor}')
        dc = dc.replace('$ecm',      f'{ecm}')
        dc = dc.replace('$cat',      f'{cat}')
        dc = dc.replace('$bkg_id',   f'{bkg_id}')

    with open(f'{runDir}/datacard.txt', 'w') as file:
        file.write(dc)

    # remove lepton scale uncertainty for other flavor
    if flavor == 'mumu':
        cmd = "sed -i '/LEPSCALE_EL/d' datacard.txt"
    else:
        cmd = "sed -i '/LEPSCALE_MU/d' datacard.txt"
    subprocess.call(cmd, shell=True, cwd=runDir)

    # Build the Combine workspace based on the datacard, save it to ws.root
    subprocess.call(['text2workspace.py', 'datacard.txt' '-o', 'ws.root', '-v', '10', '--X-allow-no-background'], shell=True, cwd=runDir)


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    sumw2err = ROOT.kTRUE

    ROOT.Math.MinimizerOptions.SetDefaultMinimizer('Fumili2')
    # ROOT.Math.MinimizerOptions.SetMinimizerAlgorithm('Simplex') # Migrad Minimize Simplex Fumili2
    ROOT.Math.MinimizerOptions.PrintDefault('Minuit2')
    ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    ROOT.Math.MinimizerOptions.SetDefaultMaxIterations(200)
    # ROOT.Math.MinimizerOptions.PrintDefault()

    try:
        main()
    except KeyboardInterrupt:
        pass  # Do not show Traceback when doing keyboard interrupt
    except Exception:
        # LOGGER.error('Error occured during execution', exc_info=True)
        pass  # Will uncomment later
    finally:
        # Print execution time
        # timer(t)
        pass
