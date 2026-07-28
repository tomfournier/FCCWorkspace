#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import array, subprocess, json, ROOT

import numpy as np



########################
### ARGUMENT PARSING ###
########################

from package.parsing import create_parser, parse_args, set_log
from package.logger import get_logger
parser = create_parser(
    cat_single=True,
    include_sel=True,
    mass_fit=True,
    datacard=True,
    description='Datacard making script'
)
arg = parse_args(parser)
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import package.plots.root.plotter as plotter
from package.tools.process import getHist
from package.userConfig import loc
loc.set_default_type('Path')

from definitions import (
    SIGNAL_MODELS,
    BACKGROUND_MODEL,
    SYSTEMATIC_MODELS,
    make_var_dict,
    make_systematic_fit_vars,
    build_datacard_signal_pdf,
    systematic_hist_suffix
)
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
    build_background_pdf,
    build_pdf_from_spec,
    build_params_from_spec,
    make_unc_import
)



###############################
### PARAMETER CONFIGURATION ###
###############################

flavor, ecm, sel = arg.flavor, arg.ecm, arg.sel
cat, tag, mode = arg.cat, arg.tag, arg.mode
lumi = 10.8 if ecm==240 else (3.12 if ecm==365 else -1)
hName = 'zll_recoil_m_cat'

label = f'#el^+#el^-, category {cat}'.replace('#el', '#mu' if flavor=='mumu' else 'e')
topLeft  = '#bf{FCC-ee} #scale[0.7]{#it{Internal}}'
topRight = f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{-1}}'

inDir    = loc.get('HIST_PROCESSED', flavor, ecm, sel)
paramDir = loc.get('PARAMETRIC',     flavor, ecm, sel)
outDir   = loc.get('DATACARD',       flavor, ecm, sel)
runDir   = loc.get('RUNDIR',         flavor, ecm, sel)

outDir.mkdir(exist_ok=True, parents=True)
runDir.mkdir(exist_ok=True, parents=True)
# subprocess.run(['cp', '/home/submit/jaeyserm/public_html/fccee/h_mass/index.php', f'{outDir}'])


sig_spec = SIGNAL_MODELS['make_datacard']
bkg_spec = BACKGROUND_MODEL
mHs = sig_spec['masses']

if cat == 0: cat_idx = (0, 5)
else:        cat_idx = (cat, cat)

nBins = arg.nBins  # total number of bins, for plotting
recoilMin, recoilMax = arg.recoilMin, arg.recoilMax


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



########################################
### SIGNAL AND BACKGROUND DEFINITION ###
########################################

def doSignal(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        workspace: ROOT.RooWorkspace,
        label: str,
        cat_idx: tuple[int, int],
        nBins: int,
        normYields: bool = True):

    global h_obs, yield_nom, yMax

    procs = sig_spec['processes'](flavor, ecm)
    MH, mrec = workspace.var('MH'), workspace.var('zll_recoil_m')

    hist_nom = get_hist(f'{flavor}_{hName}', inDir, [procs[0]], 'norm',
                        cat_idx, 1, 1, '', False)
    yMax = 1.25 * hist_nom.Rebin(int(hist_nom.GetNbinsX() / nBins)).GetMaximum()
    yield_nom = hist_nom.Integral()

    pdf_sigs = []
    garbage = []  # Need to store the variables for memory issues

    ## Build model
    ## linear functions for mean and mean_gt
    ## constants for all the rest

    cats    = ROOT.RooCategory('category', '')           # For each mass bin, define category
    hists   = ROOT.std.map('string, RooDataHist*')()     # Container holding all RooDataHists
    pdf_tot = ROOT.RooSimultaneous('pdf_tot', '', cats)  # Total pdf, containing all the categories

    var_dict = make_var_dict('make_datacard', extra=('norm', 'mH'))
    with open(f'{paramDir}/coeff.json') as coeff_file:
        coeffs = json.load(coeff_file)

    for i, mH, proc in enumerate(zip(mHs, procs)):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        LOGGER.info(f'Doing {mH_label} mass category')

        hist_zh = get_hist(f'{flavor}_{hName}', inDir, [proc], mH_label, cat_idx,
                           1, yield_nom, '', normYields)
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', f'rdh_zh_{mH_label}',
                                  ROOT.RooArgList(mrec), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        hists.insert(ROOT.std.pair('string, RooDataHist*')(mH_label, rdh_zh))  # Does not work with recent ROOT versions?
        cats.defineType(mH_label, i)

        params = build_params_from_spec(coeffs, sig_spec, mH, mH_label)
        pdf_sig, _, sig_norm = build_pdf_from_spec(mrec, params, yield_zh, mH_label, sig_spec)
        pdf_sigs.append(pdf_sig)

        # Must store the individual vars for later, to extract the values
        # Seems not to work with workspace
        for k in var_dict.keys():
            if k == 'mH': var_dict[k].append(mH)
            else: var_dict[k].append(params[k])
        var_dict['norm'].append(sig_norm)

        garbage.append(pdf_sig)

        pdf_sig.Print()
        pdf_tot.addPdf(pdf_sig, mH_label)

        if mH == 125.0 and h_obs is None: h_obs = hist_zh.Clone('h_obs')  # Take 125.0 GeV to add to observed (need to add background later as well)

    rdh_tot = ROOT.RooDataHist('rdh_tot', '', ROOT.RooArgList(mrec), cats, hists)
    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Minimizer('Minimizer', 'simplex'))

    cov_sts, fit_sts = f'Covariance Quality = {fitRes.covQual()}', f'Fit status = {fitRes.status()}'
    l = max([len(cov_sts), len(fit_sts)]) + 6
    LOGGER.info(f'\n{" FIT STATUS ":=^{l}}\n{cov_sts:^{l}}\n{fit_sts:^{l}}\n{"="*l}\n')

    workspace.Import(pdf_tot)  # Import after fit, to have fit values in the workspace

    cov = fitRes.covarianceMatrix()
    cov.Print()

    # Plot
    plotter.cfg = cfg
    cfg['ytitle'] = f'Events / {20_000/nBins:.0f} MeV'
    for i, mH, proc, pdf in enumerate(zip(mHs, procs, pdf_sigs)):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        rdh_zh = hists[mH_label]

        cfg['ymax'] = yMax
        fit_plot(rdh_zh, pdf, mH_label, mrec, nBins, label, outDir)

        cfg['ymax'] = 2.5 * yMax
        plotter.cfg = cfg
        sig_fit = plot_decomposition(workspace, outDir, label, mH_label, yield_nom, sig_spec)

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

        vals = {k:[v.getVal() if k!='mH' else v for v in values] for k, values in var_dict.items()}


    ###########################
    # plot all fitted signals #
    ###########################
    cfg['xmin'] = 124
    cfg['xmax'] = 128
    cfg['ymax'] = 2.5 * np.average(yMax)
    plotter.cfg = cfg

    plot_signal(workspace, mHs, outDir, label, yield_nom, pdf_sigs)

    # Make splines, to connect the fit parameters a function of the Higgs mass
    # Plot them afterwards
    splines = {k:ROOT.RooSpline1D(f'spline_{k}', f'spline_{k}', MH, len(vals['mH']),
                                  array.array('d', vals['mH']),
                                  array.array('d', vals[k]))
               for k in vals.keys() if k!='mH'}

    for param, spline in splines.items():
        plot_params_vs_mh(MH, outDir, param, vals, spline)

    # Was getattr(workspace, 'import')(spline_<variable>)
    # Should test if it works
    for spl in splines.values(): workspace.Import(spl)

    return workspace




def doBackgrounds(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        workspace: ROOT.RooWorkspace,
        label: str,
        cat_idx: tuple[int, int],
        nBins: int,
         ):

    global h_obs

    mrec = workspace.var('zll_recoil_m')
    hist_bkg = None

    procs = bkg_spec['processes'](flavor, ecm)
    for proc in procs:
        hist = getHist(f'{flavor}_{hName}', [proc])
        hist = hist.ProjectionX(f'hist_{proc}', *cat_idx)

        # Add to total background
        if hist_bkg is None: hist_bkg = hist
        else: hist_bkg.Add(hist)

        # Add to observed
        if h_obs is None: h_obs = hist.Clone('h_obs')
        else: h_obs.Add(hist)

    hist_bkg.SetName('total_bkg')
    rdh_bkg   = ROOT.RooDataHist('rdh_bkg', 'rdh_bkg', ROOT.RooArgList(mrec), ROOT.RooFit.Import(hist_bkg))
    yield_bkg = rdh_bkg.sum(False)

    tmp = hist_bkg.Clone()
    tmp = tmp.Rebin(int(hist_bkg.GetNbinsX() / nBins))
    cfg['ymax'] = 1.5 * tmp.GetMaximum()

    coeffs = {
        coeff['name']: ROOT.RooRealVar(coeff['name'], coeff['title'], coeff['value'], *coeff['range'])
        for coeff in bkg_spec['coefficients']
    }
    bkg, bkg_norm, bkg_fit = build_background_pdf(
        mrec, coeffs, yield_bkg, '',
        bkg_spec['order'], bkg_spec['name'],
        bkg_spec['model_name'],
        bkg_spec['yield_name'],
    )
    bkg_fit.fitTo(rdh_bkg, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))


    ########### PLOTTING ###########
    plotter.cfg = cfg
    plot_fit_with_pull(
        rdh_bkg,
        bkg_fit,
        mrec,
        nBins,
        f'{outDir}/fit_bkg',
        label,
        param_layout=(0.5, 0.9, 0.9),
    )

    # Import background parameterization to the workspace
    bkg_norm.setVal(yield_bkg)
    for coeff in coeffs.values():
        coeff.setConstant(True)

    workspace.Import(bkg)
    workspace.Import(bkg_norm)

    return workspace



##############################
### SYSTEMATICS DEFINITION ###
##############################

def setup_syst(
        flavor: str,
        ecm: int,
        hName: str,
        outDir: str,
        label: str,
        workspace: ROOT.RooWorkspace,
        syst: str,
        nBins: int,
        lumiScale: float | int,
        cat_idx: tuple[int, int],
        mH: float | int = 125.0,
         ) -> None:

    mH_label = f'{mH:.3f}'.replace('.', 'p')
    MH, mrec = workspace.var('MH'), workspace.var('zll_recoil_m')
    MH.setVal(125.0)

    Vars, spline_vals, param_names = make_systematic_fit_vars(workspace, syst, mH, mH_label)
    val_up, val_dw = [], []

    for s in ['Up', 'Down']:
        suffix = f'{mH_label}_{syst}{s}'

        s_   = systematic_hist_suffix(syst, s)
        proc = SYSTEMATIC_MODELS[syst]['process'](syst, flavor, ecm, s)

        hist_zh  = get_hist(f'{flavor}_{hName}{s_}', inDir, [proc], suffix,
                            cat_idx, lumiScale, f'hist_zh_{suffix}')
        rdh_zh   = ROOT.RooDataHist(f'rdh_zh_{suffix}', 'rdh_zh', ROOT.RooArgList(mrec), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        sig_fit, _, _ = build_pdf_from_spec(mrec, Vars, yield_zh, suffix, sig_spec)
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTrue), ROOT.RooFit.SumW2Error(ROOT.kTRUE))

        values = [Vars[name].getVal() for name in param_names]
        if s == 'Up': val_up = values
        else:         val_dw = values

        cfg['ymax'] = yMax
        plotter.cfg = cfg
        plot_fit_with_pull(
            rdh_zh,
            sig_fit,
            mrec,
            nBins,
            f'{outDir}/fit_mH{suffix}',
            label,
            'ZH signal',
            ROOT.kRed
        )

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

    # Plot all fitted signals
    cfg['ymax'] = 2.5 * yMax
    cfg['xmin'] = 124
    cfg['xmax'] = 127
    plotter.cfg = cfg

    plot_syst_dist(workspace, yield_nom, outDir, syst, mH_label)
    make_unc_import(workspace, spline_vals, list(param_names), syst, val_up, val_dw)

    return workspace



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    h_obs = None  # should hold the data_obs = sum of signal and backgrounds

    mrec = ROOT.RooRealVar('zll_recoil_m', 'm_{recoil} [GeV]', 125, recoilMin, recoilMax)
    MH = ROOT.RooRealVar('MH', 'Higgs mass [GeV]', 125, 124.95, 125.05)

    # Define temporary output workspace
    w     = ROOT.RooWorkspace('w',     'workspace')  # final workspace for combine
    w_tmp = ROOT.RooWorkspace('w_tmp', 'workspace')
    w_tmp.Import(MH); w_tmp.Import(mrec)

    w_tmp = doSignal(flavor, ecm, hName, outDir, w_tmp, label,
                     1, cat_idx, nBins)
    w_tmp = doBackgrounds(flavor, ecm, hName, outDir, w_tmp, label,
                          1, cat_idx, nBins)

    if arg.syst:
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'BES',
                           nBins, 1, cat_idx, 125.0)
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'SQRTS',
                           nBins, 1, cat_idx, 125.0)
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'LEPSCALE',
                           nBins, 1, cat_idx, 125.0)

    sig = build_datacard_signal_pdf(w_tmp, 'make_datacard', mrec, flavor, ecm, arg.syst)
    w.Import(sig)

    # Construct background model
    bkg_yield = w_tmp.obj('bkg_norm_tmp').getVal()
    bkg_norm  = ROOT.RooRealVar('bkg_norm', 'bkg_norm', bkg_yield)  # Nominal background yield (automatically done by Combine with pdfName_norm, floating)
    bkg_norm.setVal(bkg_yield)  # Not constant!
    bkg = w_tmp.obj('bkg')
    w.Import(bkg, ROOT.RooFit.RenameAllVariablesExcept(f'{flavor}_cat{cat}_ecm{ecm}', 'zll_recoil_m'))

    data_obs = ROOT.RooDataHist('data_obs', 'data_obs', ROOT.RooArgList(mrec), ROOT.RooFit.Import(h_obs))
    w.Import(data_obs)

    poi = ROOT.RooArgSet(MH)
    w.defineSet('POI', poi)

    w.writeToFile(f'{runDir}/datacard.root')
    w.Print()

    del w, w_tmp

    if   ecm == 240 and flavor == 'mumu': bkg_id = 1
    elif ecm == 240 and flavor == 'ee':   bkg_id = 2
    elif ecm == 365 and flavor == 'mumu': bkg_id = 3
    elif ecm == 365 and flavor == 'ee':   bkg_id = 4

    # Make datacard
    with open(runDir / 'datacard_template.txt', 'r') as file:
        dc = file.read()
        dc = dc.replace('$rate_sig', f'{yield_nom}')
        dc = dc.replace('$rate_bkg', f'{bkg_yield}')
        dc = dc.replace('$flavor',   f'{flavor}')
        dc = dc.replace('$ecm',      f'{ecm}')
        dc = dc.replace('$cat',      f'{cat}')
        dc = dc.replace('$bkg_id',   f'{bkg_id}')

    with open(f'{runDir}/datacard.txt', 'w') as file:
        file.write(dc)

    # Remove lepton scale uncertainty for other flavor
    if flavor == 'mumu': cmd = "sed -i '/LEPSCALE_EL/d' datacard.txt"
    else:                cmd = "sed -i '/LEPSCALE_MU/d' datacard.txt"
    subprocess.call(cmd, shell=True, cwd=runDir)

    # Build the Combine workspace based on the datacard, save it to ws.root
    subprocess.call(['text2workspace.py', 'datacard.txt' '-o', 'ws.root', '-v', '10', '--X-allow-no-background'], shell=True, cwd=runDir)


######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

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
