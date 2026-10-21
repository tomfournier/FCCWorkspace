'''Parametrize the ZH signal and create a systematic datacard in one run.'''

################################
### STANDARD LIBRARY IMPORTS ###
################################

from __future__ import annotations

import array
import json
import sys
from pathlib import Path

import ROOT



########################
### ARGUMENT PARSING ###
########################

from parsing import set_log
from zh_mass.parsing import create_parser
from logger import get_logger

arg = create_parser('3-Jan', 'make_datacard',
                    'Parametrize the signal and create the datacard and workspace').parse_args()
set_log(arg)
LOGGER = get_logger(__name__)



#########################################################
## IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
#########################################################

from zh_mass.userConfig import loc, param_config
loc.set_default_type('Path')
from zh_mass.tools.process import getHist
from definitions import (
    BACKGROUND_MODEL, SIGNAL_MODELS, SYSTEMATIC_MODELS,
    build_datacard_signal_params,
    make_systematic_fit_vars,
    systematic_hist_suffix,
)
from zh_mass.func.fit import (
    RooModel, RooModelBuilder,
    SignalParametrizer,
    get_hist, make_unc_import,
)
from zh_mass.plots.fit import (
    fit_plot, plot_decomposition,
    plot_fit_all, plot_fit_with_pull,
    plot_params_vs_mh, plot_signal,
    plot_syst_dist,
)
from combine import do_combine_mass



################################
### PARAMETERS CONFIGURATION ###
################################

# Channel, c.o.m energy, selection, category and mode definition
flavor, ecm, sel, cat, mode = arg.cat, arg.ecm, arg.sel, arg.category, arg.mode
lumi = 10.8 if ecm == 240 else (3.12 if ecm==365 else -1)

hName = arg.hName
label    = f'#el^+#el^-, category {cat}'.replace('#el', '#mu' if flavor == 'mumu' else 'e')
topRight = f'#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{-1}}'

inputdir  = loc.get('HIST_RAW', flavor, ecm, sel)
outputdir = loc.get('DATACARD', flavor, ecm, sel)

param_dir = loc.get('PARAMETRIC', flavor, ecm, sel)
run_dir   = loc.get('RUNDIR',     flavor, ecm, sel)
for directory in (param_dir, outputdir, run_dir):
    directory.mkdir(exist_ok=True, parents=True)

cat_idx = (0, 5) if cat == 0 else (cat, cat)
recoil_min, recoil_max = arg.recoilMin, arg.recoilMax
model_owners: list[RooModel] = []
h_obs = None
sig_yield = 0.0



##############################
### SIGNAL PARAMETRIZATION ###
##############################

def parametrize_signal() -> dict[str, list[float]]:
    spec = SIGNAL_MODELS['parametrize_signal']

    # TO DO: make a dictionary for processes
    processes = spec['processes'](flavor, ecm)
    nominal = processes[1]

    observable   = ROOT.RooRealVar('mrec_param', 'Recoil mass [GeV]', 125, recoil_min, recoil_max)
    parametrizer = SignalParametrizer(spec, param_config[ecm][flavor][cat])

    def load_signal(proc: str, suffix: str, normalize: bool = False, sig_yield: float | int = 1):
        if suffix == 'norm': proc = nominal
        return get_hist(hName, inputdir, proc, suffix, cat_idx, 1, sig_yield, '', normalize)

    coeffs = parametrizer.fit_grid(observable, spec['masses'], processes, load_signal, arg.normYield, True)
    (param_dir / 'coeff_parametrize.json').write_text(json.dumps(coeffs, indent=4))

    dc_coeffs = {'mean0': [1.0, 1.0, 1.0]}
    source_names = {'mean1': 'mean_cb1', 'sigma': 'sigma_cb', 'sigma_gt': 'sigma_gt', 'mean_gt1': 'mean_gt1', 'alpha_1': 'alpha_1',
                    'alpha_2': 'alpha_2', 'n_1': 'n_1', 'n_2': 'n_2', 'cb_1': 'cb_1', 'cb_2': 'cb_2'}
    dc_coeffs.update({target: coeffs[source] for target, source in source_names.items()})
    (param_dir / 'coeff.json').write_text(json.dumps(dc_coeffs, indent=4))
    return dc_coeffs



######################
### SIGNAL FITTING ###
######################

def fit_signal(workspace: ROOT.RooWorkspace, coefficients: dict) -> tuple[float, list]:
    global h_obs

    sig_data, sig_models, mass_models = [], [], []
    
    spec = SIGNAL_MODELS['make_datacard']
    builder = RooModelBuilder(spec)
    masses, processes = spec['masses'], spec['processes'](flavor, ecm)
    observable = workspace.var('m_recoil')

    nominal_hist = get_hist(hName, inputdir, [processes[1]], 'norm', cat_idx, 1, 1, '', False)
    sig_yield = nominal_hist.Integral()
    
    values = {name: [] for name in (*spec['param_names'], 'mH', 'norm')}
    
    categories   = ROOT.RooCategory('category', '')
    data_hists   = ROOT.std.map('string, RooDataHist*')()
    simultaneous = ROOT.RooSimultaneous('pdf_tot', '', categories)

    for index, (mass, process) in enumerate(zip(masses, processes)):
        mass_label = f'{mass:.3f}'.replace('.', 'p')
        hist = get_hist(hName, inputdir, [process], mass_label, cat_idx,
                        1, sig_yield, '', arg.normYield)
        data = ROOT.RooDataHist(f'rdh_zh_{mass_label}', f'rdh_zh_{mass_label}', ROOT.RooArgList(observable), ROOT.RooFit.Import(hist))
        categories.defineType(mass_label, index)
        data_hists.insert(ROOT.std.pair('string, RooDataHist*')(mass_label, data))

        # Define parameters and build model
        params = builder.build_params(coefficients, mass)
        model  = builder.build(observable, params, data.sum(False), mass_label)
        
        model.add(data.GetName(), data)
        mass_models.append(model)
        sig_models.append(model.get('pdf'))
        sig_data.append(data)
        simultaneous.addPdf(model.get('pdf'), mass_label)
        
        for name in spec['param_names']:
            values[name].append(params[name].getVal())
        values['mH'].append(mass)

        # Add normalization
        norm = model.get(f'{spec["yield_name"]}_{mass_label}')
        values['norm'].append(norm.getVal())

        if mass == 125.0:
            h_obs = hist.Clone('h_obs')

    combined_data = ROOT.RooDataHist('rdh_tot', '', ROOT.RooArgList(observable), categories, data_hists)
    fit_result = simultaneous.fitTo(combined_data, ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.Minimizer('Minimizer', 'simplex'))
    
    owner = RooModel('signal')
    owner.add('category', categories)
    owner.add('hists', data_hists)
    owner.add('pdf_tot', simultaneous)
    owner.add(combined_data.GetName(), combined_data)
    owner.add('fit_result', fit_result)
    
    for model in mass_models:
        owner.extend(model)
    model_owners.extend([owner, *mass_models])
    workspace.Import(simultaneous)
    for data in sig_data:
        workspace.Import(data)

    for mass, data, pdf in zip(masses, sig_data, sig_models):
        mass_label = f'{mass:.3f}'.replace('.', 'p')
        fit_plot(data, pdf, observable, arg.nBins, label, outputdir, ecm, lumi, suffix=f'_{mass_label}')
        decomposition = plot_decomposition(outputdir, workspace, label, mass_label, sig_yield, spec, ecm, lumi)
        workspace.Import(decomposition)

    value_arrays = {name: [item.getVal() if name not in ('mH',) else item for item in items] for name, items in values.items()}
    splines = {}
    for name, y_values in value_arrays.items():
        if name == 'mH': continue
        spline = ROOT.RooSpline1D(f'spline_{name}', f'spline_{name}', workspace.var('MH'), len(value_arrays['mH']),
                                  array.array('d', value_arrays['mH']),
                                  array.array('d', y_values))
        splines[name] = spline
        plot_params_vs_mh(workspace.var('MH'), outputdir, name, value_arrays, spline, label, ecm, lumi)
        workspace.Import(spline)

    plot_signal(workspace, outputdir, label, sig_yield, sig_models, ecm, lumi)
    plot_fit_all(outputdir, workspace, masses, sig_yield, label, ecm, lumi)
    return sig_yield



##########################
### BACKGROUND FITTING ###
##########################

def fit_background(workspace: ROOT.RooWorkspace, nBins: int = 250) -> float:
    global h_obs

    observable = workspace.var('m_recoil')
    hist_total = None
    for proc in BACKGROUND_MODEL['processes'](flavor, ecm):
        hist = getHist(hName, [proc], inputdir)
        hist = hist.ProjectionX(f'hist_{proc}', *cat_idx)
        if hist_total is None:
            hist_total = hist
        else:
            hist_total.Add(hist)
        if h_obs is None:
            h_obs = hist.Clone('h_obs')
        else:
            h_obs.Add(hist)

    hist_total.SetName('total_bkg')
    data = ROOT.RooDataHist('rdh_bkg', 'rdh_bkg', ROOT.RooArgList(observable), ROOT.RooFit.Import(hist_total))
    bkg_yield = data.sum(False)
    builder = RooModelBuilder(BACKGROUND_MODEL)
    
    model = builder.build(observable, builder.build_params({}, 0), bkg_yield)
    model.add(data.GetName(), data)
    model.fit_to(data, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))

    for entry in BACKGROUND_MODEL['params']:
        if entry['kind'] == 'variable':
            model.get(entry['name']).setConstant(ROOT.kTRUE)
    model_owners.append(model)

    plot_fit_with_pull(data, model.get('bkg_fit'), observable, outputdir, label, nBins,
                       ecm, lumi, outName='fit_bkg', param_layout=(0.5, 0.9, 0.9))
    workspace.Import(model.get('bkg'))
    workspace.Import(model.get('bkg_norm_tmp'))
    return bkg_yield



###########################
### SYSTEMATICS FITTING ###
###########################

def fit_systematics(workspace: ROOT.RooWorkspace, nBins: int = 250) -> None:
    spec = SIGNAL_MODELS['make_datacard']
    builder = RooModelBuilder(spec)
    mass = 125.0
    mass_label = f'{mass:.3f}'.replace('.', 'p')
    observable = workspace.var('m_recoil')

    for syst in ('BES', 'SQRTS', 'LEPSCALE'):
        variables, spline_vals, val_names = make_systematic_fit_vars(workspace, syst, mass, mass_label)
        values_up, values_down = [], []
        for direction in ('Up', 'Down'):
            suffix = f'{mass_label}_{syst}{direction}'
            process = SYSTEMATIC_MODELS[syst]['process'](syst, flavor, ecm, direction)
            hist_suffix = systematic_hist_suffix(syst, direction)
            hist = get_hist(f'{flavor}_{arg.hName}_cat{hist_suffix}', inputdir, [process], suffix,
                            cat_idx, 1, sig_yield, f'hist_zh_{suffix}', arg.normYield)
            data = ROOT.RooDataHist(f'rdh_zh_{suffix}', f'rdh_zh_{suffix}', ROOT.RooArgList(observable), ROOT.RooFit.Import(hist))
            model = builder.build(observable, variables, data.sum(False), suffix)
            pdf = model.get('pdf')
            pdf.fitTo(data, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))
            fitted_values = [variables[name].getVal() for name in val_names]
            if direction == 'Up': val_up = fitted_values
            else:                 val_dw = fitted_values
            plot_fit_with_pull(data, pdf, observable, outputdir, label, nBins, ecm, lumi,
                               outName='fit_mH', suffix=suffix, title='ZH signal')
            workspace.Import(data)
            workspace.Import(pdf)
            model_owners.append(model)

        plot_syst_dist(workspace, sig_yield, outputdir, syst, mass_label, ecm, lumi)
        make_unc_import(workspace, spline_vals, val_names, syst, val_up, val_dw)



##########################
### EXECUTION FUNCTION ###
##########################

def main() -> None:
    global sig_yield

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    signal_coefficients = parametrize_signal()
    mass_observable = ROOT.RooRealVar('m_recoil', 'm_{recoil} [GeV]', 125, recoil_min, recoil_max)
    higgs_mass = ROOT.RooRealVar('MH', 'Higgs mass [GeV]', 125, 124.95, 125.05)
    
    tmp_workspace = ROOT.RooWorkspace('w_tmp', 'workspace')
    tmp_workspace.Import(higgs_mass)
    tmp_workspace.Import(mass_observable)

    sig_yield = fit_signal(tmp_workspace, signal_coefficients)
    bkg_yield = fit_background(tmp_workspace, arg.nBins)
    if arg.syst:
        fit_systematics(tmp_workspace, arg.nBins)
        for syst_name in (f'BES_ecm{ecm}', f'SQRTS_ecm{ecm}', f'LEPSCALE_MU_ecm{ecm}', f'LEPSCALE_EL_ecm{ecm}',):
            tmp_workspace.Import(ROOT.RooRealVar(syst_name, syst_name, 0, -5, 5))

    params = build_datacard_signal_params(tmp_workspace, 'make_datacard', flavor, ecm, arg.syst)
    sig_pdf = RooModelBuilder(SIGNAL_MODELS['make_datacard']
    ).build(mass_observable, params, 1.0, '', extended=False).get('pdf')
    workspace = ROOT.RooWorkspace('w', 'workspace')
    workspace.Import(sig_pdf)

    bkg_pdf = tmp_workspace.pdf('bkg')
    workspace.Import(bkg_pdf,ROOT.RooFit.RenameAllVariablesExcept(f'{flavor}_cat{cat}_ecm{ecm}', 'm_recoil'))
    observed = ROOT.RooDataHist('data_obs', 'data_obs', ROOT.RooArgList(mass_observable), ROOT.RooFit.Import(h_obs))
    workspace.Import(observed)
    if workspace.var('MH') is None:
        workspace.Import(higgs_mass)
    workspace.defineSet('POI', ROOT.RooArgSet(workspace.var('MH')))

    if ecm == 240:
        background_id = 1 if flavor == 'mumu' else 2
    elif ecm == 365:
        background_id = 3 if flavor == 'mumu' else 4
    else:
        raise ValueError(f'{ecm = } not supported; choose between 240 and 365')

    template = Path('datacard_template.txt').read_text()
    datacard = template.replace('$rate_sig', f'{sig_yield}')
    datacard = datacard.replace('$rate_bkg', f'{bkg_yield}')
    datacard = datacard.replace('$flavor', flavor)
    datacard = datacard.replace('$ecm', str(ecm))
    datacard = datacard.replace('$cat', str(cat))
    datacard = datacard.replace('$bkg_id', str(background_id))
    removed_systematic = 'LEPSCALE_EL' if flavor == 'mumu' else 'LEPSCALE_MU'

    def write_workspace(directory: Path) -> None:
        workspace.writeToFile(str(directory / 'datacard.root'))

    do_combine_mass(run_dir, datacard, write_workspace, removed_systematic)
    del workspace, tmp_workspace



######################
### CODE EXECUTION ###
######################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        LOGGER.error('Mass-fit workflow failed', exc_info=True)
        raise
