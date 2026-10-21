#################################
### IMPORT STANDARD LIBRARIES ###
#################################

import time, array, subprocess, json, ROOT

t = time.time()



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_mass.parsing import create_parser
from logger import get_logger
arg = create_parser('3-Jan', 'make_datacard').parse_args()
set_log(arg)

LOGGER = get_logger(__name__)



##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

from zh_mass.tools.process import getHist
from tools.utils import timer
from zh_mass.userConfig import loc, param_config
loc.set_default_type('Path')

from definitions import (
    SIGNAL_MODELS,
    BACKGROUND_MODEL,
    SYSTEMATIC_MODELS,
    build_datacard_signal_params,
    make_var_dict,
    make_systematic_fit_vars,
    systematic_hist_suffix
)
from zh_mass.plots.fit import (
    fit_plot,
    plot_signal,
    plot_syst_dist,
    plot_params_vs_mh,
    plot_decomposition,
    plot_fit_with_pull
)
from zh_mass.func.fit import (
    RooModel,
    RooModelBuilder,
    SignalParametrizer,
    get_hist,
    make_unc_import
)
from combine import do_combine_mass

MODEL_OWNERS = []



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

sig_spec = SIGNAL_MODELS['make_datacard']
sig_builder = RooModelBuilder(sig_spec)
bkg_spec = BACKGROUND_MODEL
mHs = sig_spec['masses']

if cat == 0: cat_idx = (0, 5)
else:        cat_idx = (cat, cat)

nBins = arg.nBins  # total number of bins, for plotting
recoilMin, recoilMax = arg.recoilMin, arg.recoilMax


def ensure_signal_coefficients() -> None:
    """Run the signal parametrization inline when no coefficient file exists."""
    coefficient_file = paramDir / 'coeff.json'
    if coefficient_file.exists():
        return

    LOGGER.info('No signal parametrization found; fitting signal mass points inline')
    signal_spec = SIGNAL_MODELS['parametrize_signal']
    processes = signal_spec['processes'](flavor, ecm)
    recoilmass = ROOT.RooRealVar('zll_recoil_m_param', 'Recoil mass [GeV]', 125, arg.recoilMin, arg.recoilMax)
    parametrizer = SignalParametrizer(signal_spec, param_config[ecm][flavor][cat])

    def load_hist(process, suffix, normalize=False, yield_nom=1):
        return get_hist(f'{flavor}_{hName}', inDir, process, suffix, cat_idx,
                        1, yield_nom, '', normalize)

    coefficients, _, _ = parametrizer.fit_grid(recoilmass, signal_spec['masses'], processes, load_hist,
                                               normalize=getattr(arg, 'normYield', True))
    paramDir.mkdir(exist_ok=True, parents=True)
    coefficient_file.write_text(json.dumps(coefficients, indent=4))



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

    global h_obs, yield_nom

    procs = sig_spec['processes'](flavor, ecm)
    MH, mrec = workspace.var('MH'), workspace.var('zll_recoil_m')

    hist_nom = get_hist(f'{flavor}_{hName}', inDir, [procs[0]], 'norm',
                        cat_idx, 1, 1, '', False)
    yield_nom = hist_nom.Integral()

    pdf_sigs = []
    signal_model = RooModel('signal')

    ## Build model
    ## linear functions for mean and mean_gt
    ## constants for all the rest

    cats    = ROOT.RooCategory('category', '')           # For each mass bin, define category
    hists   = ROOT.std.map('string, RooDataHist*')()     # Container holding all RooDataHists
    pdf_tot = ROOT.RooSimultaneous('pdf_tot', '', cats)  # Total pdf, containing all the categories

    var_dict = make_var_dict('make_datacard', extra=('norm', 'mH'))
    with open(f'{paramDir}/coeff.json') as coeff_file:
        coeffs = json.load(coeff_file)

    signal_model.add('category', cats)
    signal_model.add('hists', hists)
    signal_model.add('pdf_tot', pdf_tot)

    for i, (mH, proc) in enumerate(zip(mHs, procs)):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        LOGGER.info(f'Doing {mH_label} mass category')

        hist_zh = get_hist(f'{flavor}_{hName}', inDir, [proc], mH_label, cat_idx,
                           1, yield_nom, '', normYields)
        rdh_zh = ROOT.RooDataHist(f'rdh_zh_{mH_label}', f'rdh_zh_{mH_label}',
                                  ROOT.RooArgList(mrec), ROOT.RooFit.Import(hist_zh))
        yield_zh = rdh_zh.sum(False)

        hists.insert(ROOT.std.pair('string, RooDataHist*')(mH_label, rdh_zh))  # Does not work with recent ROOT versions?
        cats.defineType(mH_label, i)

        params = sig_builder.build_params(coeffs, mH)
        mass_model = sig_builder.build(mrec, params, yield_zh, mH_label)
        pdf_sig = mass_model.get('pdf')
        sig_norm = mass_model.get(f'{sig_spec["yield_name"]}_{mH_label}')
        pdf_sigs.append(pdf_sig)
        mass_model.add(rdh_zh.GetName(), rdh_zh)
        signal_model.extend(mass_model)

        # Must store the individual vars for later, to extract the values
        # Seems not to work with workspace
        for k in var_dict.keys():
            if k == 'mH': var_dict[k].append(mH)
            else: var_dict[k].append(params[k])
        var_dict['norm'].append(sig_norm)

        pdf_sig.Print()
        pdf_tot.addPdf(pdf_sig, mH_label)

        if mH == 125.0 and h_obs is None: h_obs = hist_zh.Clone('h_obs')  # Take 125.0 GeV to add to observed (need to add background later as well)

    rdh_tot = ROOT.RooDataHist('rdh_tot', '', ROOT.RooArgList(mrec), cats, hists)
    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Minimizer('Minimizer', 'simplex'))
    signal_model.add(rdh_tot.GetName(), rdh_tot)
    signal_model.add('fit_result', fitRes)
    MODEL_OWNERS.append(signal_model)

    cov_sts, fit_sts = f'Covariance Quality = {fitRes.covQual()}', f'Fit status = {fitRes.status()}'
    l = max([len(cov_sts), len(fit_sts)]) + 6
    LOGGER.info(f'\n{" FIT STATUS ":=^{l}}\n{cov_sts:^{l}}\n{fit_sts:^{l}}\n{"="*l}\n')

    workspace.Import(pdf_tot)  # Import after fit, to have fit values in the workspace

    cov = fitRes.covarianceMatrix()
    cov.Print()

    # Plot
    for i, mH, proc, pdf in enumerate(zip(mHs, procs, pdf_sigs)):

        mH_label = f'{mH:.3f}'.replace('.', 'p')
        rdh_zh = hists[mH_label]

        fit_plot(
            rdh_zh, pdf, mrec, nBins, label, outDir,
            ecm=ecm, lumi=lumi,
        )

        sig_fit = plot_decomposition(
            outDir, workspace, label, mH_label, yield_nom, sig_spec,
            ecm=ecm, lumi=lumi,
        )

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

    vals = {k: [v.getVal() if k != 'mH' else v for v in values] for k, values in var_dict.items()}


    ###########################
    # plot all fitted signals #
    ###########################
    plot_signal(
        workspace, mHs, outDir, label, yield_nom, pdf_sigs,
        ecm=ecm, lumi=lumi,
    )

    # Make splines, to connect the fit parameters a function of the Higgs mass
    # Plot them afterwards
    splines = {k:ROOT.RooSpline1D(f'spline_{k}', f'spline_{k}', MH, len(vals['mH']),
                                  array.array('d', vals['mH']),
                                  array.array('d', vals[k]))
               for k in vals.keys() if k!='mH'}

    for param, spline in splines.items():
        plot_params_vs_mh(
            MH, outDir, param, vals, spline,
            ecm=ecm, lumi=lumi,
        )

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

    bkg_builder = RooModelBuilder(bkg_spec)
    bkg_model = bkg_builder.build(
        mrec, bkg_builder.build_params({}, 0), yield_bkg
    )
    coeffs = {
        entry['name']: bkg_model.get(entry['name']) for entry in bkg_spec['params'] if entry['kind'] == 'variable'
    }
    bkg_model.add('rdh_bkg', rdh_bkg)
    bkg_model.fit_to(
        rdh_bkg, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE)
    )
    MODEL_OWNERS.append(bkg_model)


    ########### PLOTTING ###########
    plot_fit_with_pull(
        rdh_bkg,
        bkg_model.get('bkg_fit'),
        mrec,
        nBins,
        outDir,
        label,
        ecm=ecm, lumi=lumi, outName='fit_bkg',
        param_layout=(0.5, 0.9, 0.9),
    )

    # Import background parameterization to the workspace
    bkg_norm = bkg_model.get('bkg_norm_tmp')
    bkg_norm.setVal(yield_bkg)
    for coeff in coeffs.values():
        coeff.setConstant(True)

    workspace.Import(bkg_model.get('bkg'))
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

        sig_fit = sig_builder.build(mrec, Vars, yield_zh, suffix).get('pdf')
        sig_fit.fitTo(rdh_zh, ROOT.RooFit.Extended(ROOT.kTrue), ROOT.RooFit.SumW2Error(ROOT.kTRUE))

        values = [Vars[name].getVal() for name in param_names]
        if s == 'Up': val_up = values
        else:         val_dw = values

        plot_fit_with_pull(
            rdh_zh, sig_fit, mrec, nBins, outDir, label,
            ecm=ecm, lumi=lumi, outName='fit_mH', suffix=suffix,
            title='ZH signal', fit_color=ROOT.kRed,
        )

        # Import
        workspace.Import(rdh_zh)
        workspace.Import(sig_fit)

    # Plot all fitted signals
    plot_syst_dist(
        workspace, yield_nom, outDir, syst, mH_label,
        ecm=ecm, lumi=lumi,
    )
    make_unc_import(workspace, spline_vals, list(param_names), syst, val_up, val_dw)

    return workspace



##########################
### EXECUTION FUNCTION ###
##########################

def main():
    h_obs = None  # should hold the data_obs = sum of signal and backgrounds

    ensure_signal_coefficients()

    MH   = ROOT.RooRealVar('MH', 'Higgs mass [GeV]', 125, 124.95, 125.05)
    mrec = ROOT.RooRealVar('zll_recoil_m', 'm_{recoil} [GeV]', 125, recoilMin, recoilMax)

    # Define temporary output workspace
    w     = ROOT.RooWorkspace('w',     'workspace')  # final workspace for combine
    w_tmp = ROOT.RooWorkspace('w_tmp', 'workspace')
    w_tmp.Import(MH); w_tmp.Import(mrec)

    w_tmp = doSignal(flavor, ecm, hName, outDir, w_tmp, label,
                     cat_idx, nBins)
    w_tmp = doBackgrounds(flavor, ecm, hName, outDir, w_tmp, label,
                          1, cat_idx, nBins)

    if arg.syst:
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'BES', nBins, 1, cat_idx, 125.0)
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'SQRTS', nBins, 1, cat_idx, 125.0)
        w_tmp = setup_syst(flavor, ecm, hName, outDir, label, w_tmp, 'LEPSCALE', nBins, 1, cat_idx, 125.0)

    params = build_datacard_signal_params(w_tmp, 'make_datacard', flavor, ecm, arg.syst)
    sig    = RooModelBuilder(sig_spec).build(mrec, params, 1.0, '', False).get('pdf')
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

    w.Print()

    if ecm == 240:   bkg_id = 1 if flavor=='mumu' else (2 if flavor=='ee' else None)
    elif ecm == 365: bkg_id = 3 if flavor=='mumu' else (4 if flavor=='ee' else None)
    else: raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    # Make datacard
    with open(runDir / 'datacard_template.txt', 'r') as file:
        dc = file.read()
        dc = dc.replace('$rate_sig', f'{yield_nom}')
        dc = dc.replace('$rate_bkg', f'{bkg_yield}')
        dc = dc.replace('$flavor',   f'{flavor}')
        dc = dc.replace('$ecm',      f'{ecm}')
        dc = dc.replace('$cat',      f'{cat}')
        dc = dc.replace('$bkg_id',   f'{bkg_id}')

    remove_systematic = 'LEPSCALE_EL' if flavor == 'mumu' else 'LEPSCALE_MU'

    def write_workspace(_run_dir):
        w.writeToFile(str(_run_dir / 'datacard.root'))

    do_combine_mass(runDir, dc, write_workspace, remove_systematic)
    del w, w_tmp


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
        LOGGER.error('Error occured during execution', exc_info=True)
    finally:
        # Print execution time
        timer(t)
