import json, math, array, subprocess, ROOT

from ..userConfig import PathObj
from ..tools.process import getHist
from ..plots.fit import (
    plot_mass_scan,
    plot_mass_breakdown_curves,
    plot_mass_breakdown_impacts
)


def _source_value(source: dict, key: str, idx: int | None = None):
    value = source[key]
    if idx is None:
        return value
    return value[idx]


def _component_pdf(kind: str, name: str, title: str, recoilmass, params: dict, spec: dict):
    if kind == 'cbshape':
        return ROOT.RooCBShape(
            name,
            title,
            recoilmass,
            params[spec['mean']],
            params[spec['sigma']],
            params[spec['alpha']],
            params[spec['n']],
        )
    if kind == 'gauss':
        return ROOT.RooGaussian(
            name,
            title,
            recoilmass,
            params[spec['mean']],
            params[spec['sigma']],
        )
    raise ValueError(f'{kind = } not supported')


def build_params_from_spec(
    source: dict,
    model_spec: dict,
    mH: float | int,
     ) -> dict[str, ROOT.RooAbsArg]:

    params: dict[str, ROOT.RooAbsArg] = {}

    for entry in model_spec['params']:
        # name = f"{entry.get('name', '')}_{mH_label}" if entry.get('suffix', True) else entry['name']
        if entry['kind'] == 'mH':
            params[entry['name']] = ROOT.RooRealVar(entry['name'], '', mH)
            continue

        if entry['kind'] == 'source':
            raw = _source_value(source, entry['key'], entry.get('idx'))
            if isinstance(raw, (list, tuple)) and len(raw) == 3:
                params[entry['name']] = ROOT.RooRealVar(entry['name'], '', *raw)
            elif 'lo' in entry and 'hi' in entry:
                params[entry['name']] = ROOT.RooRealVar(entry['name'], '', raw, entry['lo'], entry['hi'])
            else:
                params[entry['name']] = ROOT.RooRealVar(entry['name'], '', raw)
            params[entry['name']].setConstant(ROOT.kTRUE)
            continue

        if entry['kind'] == 'formula':
            expr = entry['expr'].format(mH=mH)
            params[entry['name']] = ROOT.RooFormulaVar(
                entry['name'],
                '',
                expr,
                _arg_list(*(params[arg] for arg in entry['args'])),
            )
            continue

        raise ValueError(f"Unsupported parameter kind: {entry['kind']}")

    return params


def build_pdf_from_spec(
    recoilmass: ROOT.RooRealVar,
    params: dict[str, ROOT.RooRealVar],
    yield_zh: float | int,
    suffix: str,
    model_spec: dict,
    extended: bool = True,
     ):

    components = {}
    ordered_components = []

    for comp in model_spec['components']:
        comp_name = _with_suffix(comp['name'], suffix)
        pdf = _component_pdf(comp['kind'], comp_name, comp['title'], recoilmass, params, comp)
        components[comp['name']] = pdf
        ordered_components.append(pdf)

    sig = ROOT.RooAddPdf(
        _with_suffix(model_spec.get('sum_name', 'sig'), suffix),
        '',
        _arg_list(*ordered_components),
        _arg_list(*(params[name] for name in model_spec['fractions'])),
    )
    sig_norm = ROOT.RooRealVar(_with_suffix(model_spec.get('yield_name', 'sig_norm'), suffix), '', yield_zh, 0, 1e8)
    if extended:
        sig_fit = ROOT.RooAddPdf(
            _with_suffix(model_spec.get('model_name', 'zh_model'), suffix),
            '',
            _arg_list(sig),
            _arg_list(sig_norm),
        )
        return sig_fit, components, sig_norm

    return sig, components, sig_norm

def get_hist(
        hName: str,
        inDir: str,
        procs: str | list[str],
        suffix: str,
        cat_idx: tuple[int, int],
        lumiScale: float | int,
        yield_nom: float | int,
        outName: str = '',
        normYields: bool = True):
    if isinstance(procs, str): procs = [procs]

    hist = getHist(hName, procs, inDir)
    hist.Scale(lumiScale)

    hist = hist.ProjectionX(f'hist_zh_{suffix}', *cat_idx)
    if outName:    hist.SetName(outName)
    if normYields: hist.Scale(yield_nom / hist.Integral())

    return hist


def make_params(
        flavor: str,
        ecm: int,
        cat: int,
        mH: float | int,
        config: dict[int,
                     dict[str,
                          dict[str,
                               dict[str,
                                    list[float | int]]]]]):

    import ROOT

    conf = config[ecm][flavor][cat]

    mean_cb0 = ROOT.RooRealVar(f'mean_cb0_{mH}',     '', mH)                 # Slope
    mean_cb1 = ROOT.RooRealVar(f'mean_cb1_{mH}',     '', *conf['mean_cb1'])  # Offset
    mean_cb  = ROOT.RooFormulaVar(f'mean_cb_{mH}', '@0+@1', ROOT.RooArgList(mean_cb0, mean_cb1))

    sigma_cb = ROOT.RooRealVar(f'sigma_cb_{mH}', '', *conf['sigma_cb'])
    sigma_gt = ROOT.RooRealVar(f'sigma_gt_{mH}', '', *conf['sigma_gt'])

    alpha_1 = ROOT.RooRealVar(f'alpha_1_{mH}', '', *conf['alpha_1'])
    alpha_2 = ROOT.RooRealVar(f'alpha_2_{mH}', '', *conf['alpha_2'])
    n_1 = ROOT.RooRealVar(f'n_1_{mH}', '', *conf['n_1'])
    n_2 = ROOT.RooRealVar(f'n_2_{mH}', '', *conf['n_2'])

    mean_gt1 = ROOT.RooRealVar(f'mean_gt1_{mH}',        '', *conf['mean_gt1'])  # Offset
    mean_gt  = ROOT.RooFormulaVar(f'mean_gt_{mH}', '@0+@1', ROOT.RooArgList(mean_cb, mean_gt1))

    cb_1 = ROOT.RooRealVar(f'cb_1_{mH}', '', *conf['cb_1'])
    cb_2 = ROOT.RooRealVar(f'cb_2_{mH}', '', *conf['cb_2'])

    params = {
        'mean_cb0':  mean_cb0, 'mean_cb1':  mean_cb1, 'mean_cb':   mean_cb,
        'sigma_cb':  sigma_cb, 'sigma_gt':  sigma_gt,
        'alpha_1':   alpha_1,  'alpha_2':   alpha_2,
        'n_1':       n_1,      'n_2':       n_2,
        'mean_gt1':  mean_gt1, 'mean_gt':   mean_gt,
        'cb_1':      cb_1,     'cb_2':      cb_2
    }

    return params


def _with_suffix(base: str, suffix: str) -> str:
    return base if suffix == '' else f'{base}_{suffix}'


def _const_real(
        name: str,
        coeffs: dict[str, list[float | int]],
        lo: float | int | None = None,
        hi: float | int | None = None,
        idx: int = 1) -> ROOT.RooRealVar:
    value = coeffs.get(name, 1)[idx]
    if lo is None or hi is None:
        var = ROOT.RooRealVar(name, '', value)
    else:
        var = ROOT.RooRealVar(name, '', value, lo, hi)
    var.setConstant(ROOT.kTRUE)
    return var


def _arg_list(*items, name: str | None = None):
    arg_list = ROOT.RooArgList()
    if name is not None:
        arg_list.setName(name)
    for item in items:
        arg_list.add(item)
    return arg_list


def _formula_var(name: str, expr: str, *items):
    return ROOT.RooFormulaVar(name, '', expr, _arg_list(*items))


def make_p(mH, mH_label, inDir):

    with open(f'{inDir}/coeff.json') as fIn: coeffs = json.load(fIn)

    mean0     = _const_real('mean0',    coeffs, .5,  1.5)   # slope
    mean1     = _const_real('mean_cb1', coeffs, -1,  1)     # offset
    mean_gt1  = _const_real('mean_gt1', coeffs, -2,  2)
    sigma0    = _const_real('sigma_cb', coeffs, +0,  10)    # 0.4335
    sigma_gt0 = _const_real('sigma_gt', coeffs, +0,  10)
    alpha10   = _const_real('alpha_1',  coeffs, -5,  5)
    alpha20   = _const_real('alpha_2',  coeffs, -5,  5)
    n10       = _const_real('n_1',      coeffs, -50, 50)
    n20       = _const_real('n_2',      coeffs, -50, 50)
    cb10      = _const_real('cb_1',     coeffs, +0,  1)
    cb20      = _const_real('cb_2',     coeffs, +0,  1)

    mean = _formula_var(f'mean_{mH_label}', f'@1 + @0*{mH}', mean0, mean1)
    mean_gt = _formula_var(f'mean_gt_{mH_label}', '@0 + @1', mean, mean_gt1)

    params = {
        'mean':     mean,
        'sigma':    _formula_var(f'sigma_{mH_label}', '@0', sigma0),
        'sigma_gt': _formula_var(f'sigma_gt_{mH_label}', '@0', sigma_gt0),
        'alpha_1':  _formula_var(f'alpha_1_{mH_label}', '@0', alpha10),
        'alpha_2':  _formula_var(f'alpha_2_{mH_label}', '@0', alpha20),
        'n_1':      _formula_var(f'n_1_{mH_label}', '@0', n10),
        'n_2':      _formula_var(f'n_2_{mH_label}', '@0', n20),
        'cb_1':     _formula_var(f'cb_1_{mH_label}', '@0', cb10),
        'cb_2':     _formula_var(f'cb_2_{mH_label}', '@0', cb20),
        'mean0':    mean0,
        'mean1':    mean1,
        'mean_gt1': mean_gt1,
        'mean_gt':  mean_gt,
    }
    return params



def build_2cbg_pdf(
    recoilmass: ROOT.RooRealVar,
    Vars: dict[str, ROOT.RooRealVar],
    yield_zh: float | int,
    suffix: str,
    model_spec: dict | None = None,
     ):
    model_spec = model_spec or {
        'components': (
            {'kind': 'cbshape', 'name': 'CrystallBall_1', 'title': 'CrystallBall_1', 'mean': 'mean', 'sigma': 'sigma', 'alpha': 'alpha_1', 'n': 'n_1'},
            {'kind': 'cbshape', 'name': 'CrystallBall_2', 'title': 'CrystallBall_2', 'mean': 'mean', 'sigma': 'sigma', 'alpha': 'alpha_2', 'n': 'n_2'},
            {'kind': 'gauss', 'name': 'gauss', 'title': 'gauss', 'mean': 'mean_gt', 'sigma': 'sigma_gt'},
        ),
        'fractions': ('cb_1', 'cb_2'),
        'yield_name': 'sig_norm',
        'sum_name': 'sig',
        'model_name': 'zh_model',
    }
    sig_fit, _, _ = build_pdf_from_spec(recoilmass, Vars, yield_zh, suffix, model_spec)
    return sig_fit


def build_background_pdf(
    recoilmass: ROOT.RooRealVar,
    coeffs: dict[str, ROOT.RooRealVar],
    yield_bkg: float | int,
    suffix: str,
    order: int = 3,
    name: str = 'bkg',
    model_name: str = 'bkg_fit',
    yield_name: str = 'bkg_norm_tmp',
):
    coeff_values = [coeffs[key] for key in sorted(coeffs) if key.startswith('bern')][:order]
    coeff_list = _arg_list(*coeff_values)
    bkg = ROOT.RooBernsteinFast(order)(_with_suffix(name, suffix), _with_suffix(name, suffix), recoilmass, coeff_list)
    bkg_norm = ROOT.RooRealVar(_with_suffix(yield_name, suffix), _with_suffix(yield_name, suffix), yield_bkg, 0, 1e6)
    bkg_fit = ROOT.RooAddPdf(_with_suffix(model_name, suffix), '', ROOT.RooArgList(bkg), ROOT.RooArgList(bkg_norm))
    return bkg, bkg_norm, bkg_fit


def make_unc_import(w_tmp, spline_vals, val_names, syst, val_up=None, val_dw=None):

    for idx, (spline_val, val_name) in enumerate(zip(spline_vals, val_names)):
        nominal_val = spline_val.getVal()
        if val_up is None or val_dw is None:
            value = 0.0
        else:
            value = 0.0 if nominal_val == 0 else 0.5 * abs(val_up[idx] - val_dw[idx]) / nominal_val

        nominal = ROOT.RooRealVar(f'sig_{val_name}_{syst}', f'sig_{val_name}_{syst}', value)
        nominal.setConstant(ROOT.kTRUE)
        w_tmp.Import(nominal)


def findCrossing(
        xv: list[float | int],
        yv: list[float | int],
        left: bool = True,
        flip: float | int = 125,
        cross: float | int = 1.):

    closest, idx = 1e9, -1
    for i in range(0, len(xv)):
        if     left and xv[i] > flip: continue
        if not left and xv[i] < flip: continue
        dy = abs(yv[i] - cross)
        if dy < closest:
            closest = dy
            idx = i

    # Find correct indices around crossing
    if left:
        if yv[idx] > cross: idx_ = idx + 1
        else: idx_ = idx - 1
    else:
        if yv[idx] > cross: idx_ = idx -1
        else: idx_ = idx + 1

    # Do interpolation
    omega = (yv[idx] - yv[idx_]) / (xv[idx] - xv[idx_])
    return (cross - yv[idx]) / omega + xv[idx]


def _graph_from_points(
        xv: list[float | int],
        yv: list[float | int]) -> ROOT.TGraph:
    return ROOT.TGraph(len(xv), array.array('d', xv), array.array('d', yv))


def _load_nll_curve(
        file_path: PathObj) -> tuple[float, float, list[float], list[float], ROOT.TGraph]:
    xv, yv = [], []
    with open(file_path, 'r') as fIn:
        for i, line in enumerate(fIn.readlines()):
            line = line.rstrip()
            if i == 0:
                best_mass = float(line.split(' ')[3])
                unc_mass = float(line.split(' ')[2])
            else:
                xv.append(float(line.split(' ')[0]))
                yv.append(float(line.split(' ')[1]))

    return best_mass, unc_mass, xv, yv, _graph_from_points(xv, yv)


def _load_fit_summary(
        file_path: PathObj,
        scale: float = 1.0) -> tuple[float, float]:
    with open(file_path, 'r') as fIn:
        first_line = fIn.readline().rstrip()
    best = float(first_line.split(' ')[3])
    unc = float(first_line.split(' ')[2]) * scale
    return best, unc


def _write_fit_curve(
        out_path: PathObj,
        unc_m: float,
        unc_p: float,
        unc: float,
        mass: float,
        xv: list[float],
        yv: list[float]) -> None:
    lines = [f'{unc_m} {unc_p} {unc} {mass}\n']
    lines.extend(f'{x} {y}\n' for x, y in zip(xv, yv))
    with open(out_path, 'w') as fOut:
        fOut.writelines(lines)


def _run_combine(
        cmd: list[str],
        runDir: str) -> None:
    subprocess.call(cmd, cwd=runDir)


def analyzeMass(
        runDir: PathObj,
        outDir: PathObj,
        xMin: float | int = -1,
        xMax: float | int = -1,
        yMin: float | int = 0,
        yMax: float | int = 2,
        label: str = 'label',
        top_right: str = '',
        suffix: str = ''):

    outDir.mkdir(exist_ok=True, parents=True)

    fIn = ROOT.TFile(runDir / 'higgsCombinemass.MultiDimFit.mH125.root', 'READ')
    t = fIn.Get('limit')

    xv, yv = [], []
    for i in range(0, t.GetEntries()):

        t.GetEntry(i)

        if t.quantileExpected < -1.5: continue
        if t.deltaNLL > 20: continue
        xv.append(t.MH)
        yv.append(t.deltaNLL*2.)

    xv, yv = zip(*sorted(zip(xv, yv)))
    g = _graph_from_points(list(xv), list(yv))

    # bestfit = minimum
    mass = 1e9
    for i in range(g.GetN()):
        if g.GetY()[i] == 0: mass = g.GetX()[i]

    # extract uncertainties at crossing = 1
    unc_m = findCrossing(xv, yv, True,  mass)
    unc_p = findCrossing(xv, yv, False, mass)
    unc = (abs(mass - unc_m) + abs(unc_p - mass)) / 2

    plot_mass_scan(
        outDir,
        g,
        label,
        unc * 1000.0,
        min(xv) if xMin < 0 else xMin,
        max(xv) if xMax < 0 else xMax,
        yMin,
        yMax,
        '#bf{FCC-ee} #scale[0.7]{#it{Internal}}',
        top_right,
        output_name=f'mass{suffix}',
        graph_color=ROOT.kRed,
        graph_width=2,
    )

    # Write values to text file
    _write_fit_curve(outDir / f'mass{suffix}.txt', unc_m, unc_p, unc, mass, list(xv), list(yv))
    _write_fit_curve(runDir / f'mass{suffix}.txt', unc_m, unc_p, unc, mass, list(xv), list(yv))


def doFit_mass(
        runDir: str,
        mhMin: float | int = 124.99,
        mhMax: float | int = 125.01,
        npoints: int = 50,
        combineOptions: list[str] = []):

    cmd = ['combine', '-M', 'MultiDimFit', 'ws.root', '-t', '-1', '-v', '2', '-n', 'mass',
           '--setParameterRanges', f'MH={mhMin},{mhMax}', '--algo', 'grid', '--points', f'{npoints}',
           '--expectSignal=1', '-m', '125', '--redefinesSignalPOIs', 'MH',
           '--X-rtd', 'TMCSO_AdaptivePseudoAsimov', '--X-rtd', 'ADDNLL_CBNLL=0'] + combineOptions
    _run_combine(cmd, runDir)


def doFitDiagnostics_mass(
        runDir: str,
        mhMin: float | int = 124.99,
        mhMax: float | int = 125.01,
        combineOptions: list[str] = []):

    cmd = ['combine', '-M', 'MultiDimFit', 'ws.root', '-t', '-1', '-v', '2', '-m', '125', '-n', 'mass',
           '--setParameterRanges', f'MH={mhMin},{mhMax}', '--expectSignal=1',
           '--algo', 'singles', '--redefineSignalPOIs', 'MH', '--floatParameters', 'MH',
           '--X-rtd', 'TMCSO_AdaptivePseudoAsimov', '--X-rtd', 'ADD_CBNLL=0'] + combineOptions

    _run_combine(cmd, runDir)

    # Get the uncertainty
    ### Will optimize it later (use code from xsec extraction)
    with ROOT.TFile(f'{runDir}/higgsCombinemass.MultiDimFit.mH125.root') as fIn:
        tt = fIn.Get('limit')
        vals = []
        for i in range(tt.GetEntries()):
            tt.GetEntry(i)
            vals.append(float(tt.MH))

        vals = sorted(vals)
        lo, best, hi = vals[0], vals[1], vals[2]

        err_down, err_up = best - lo, hi - best
        err_avg = (err_up + err_down) / 2
    return err_avg

def breakDown(
        outDir: PathObj,
        top_right: str = '',
        mass_systematics: list[str] | None = None):
    if mass_systematics is None:
        mass_systematics = ['BES', 'SQRTS', 'LEPSCALE_MU', 'LEPSCALE_EL']

    def getUnc(tag, type):
        scale = 1000 if type == 'mass' else 100 if type == 'xsec' else 1
        best, unc = _load_fit_summary(outDir / f'{type}{tag}.txt', scale)
        return best, unc

    _, unc_ref = getUnc('_stat', 'mass')
    params = [f'_{name}' for name in mass_systematics] + ['']
    labels = ['BES', '#sqrt{s} #pm 2 MeV', 'Muon scale (~10^{-5})', 'El. scale (~10^{-5})', 'Syst. combined']

    impacts_mev = []
    for p in params:
        _, unc = getUnc(p, 'mass')
        impacts_mev.append(math.sqrt(unc**2 - unc_ref**2))

    plot_mass_breakdown_impacts(
        outDir,
        impacts_mev,
        labels,
        top_right,
        '#bf{FCCee} #scale[0.7]{#it{Simulation}}',
    )

    params = ['_stat', '_BES', '_LEPSCALE', '_SQRTS', '']
    labels = ['Stat. only', 'Beam energy spread', 'Lepton scale', 'Center-of-mass energy', 'Stat. + syst. combined']

    tags = [f'{outDir}/mass{p}.txt' for p in params]
    curves = []
    style_spec = [
        (ROOT.kRed + 1,    4, 1),
        (ROOT.kBlue - 4,   2, 2),
        (ROOT.kOrange - 3, 2, 2),
        (ROOT.kGreen + 3,  2, 2),
        (ROOT.kBlack,      4, 1),
    ]
    for index, tag in enumerate(tags):
        _, unc, x_values, y_values, _ = _load_nll_curve(tag)
        curves.append((x_values, y_values, unc * 1000.0, *style_spec[index]))

    plot_mass_breakdown_curves(
        outDir,
        curves,
        labels,
        top_right,
        '#bf{FCC-ee} #scale[0.7]{#it{Simulation}}',
    )


def text2workspace(runDir):
    cmd = ['text2workspace.py', 'datacard.txt', '-o', 'ws.root', '-v', '10', '--X-allow-no-background']
    _run_combine(cmd, runDir)


def combineCards(
        runDir: PathObj,
        cards: list[str] = []):

    runDir.mkdir(exist_ok=True, parents=True)
    cmd = ['combineCards.py', '--force-shape'] + cards

    with open(runDir / 'datacard.txt', 'w') as out:
        subprocess.call(cmd, cwd=runDir, stdout=out)


def run_mass_pipeline(
        runDir: PathObj,
        outDir: PathObj,
        label: str,
        combineOptions: list[str],
        fitRange: tuple[float, float] = (124.95, 125.05),
        top_right: str = '',
        suffix: str = '') -> None:
    mh_err = doFitDiagnostics_mass(runDir, fitRange[0], fitRange[1], combineOptions)
    mhMin, mhMax = mHrange(mh_err)
    doFit_mass(runDir, mhMin, mhMax, 50, combineOptions)
    analyzeMass(runDir, outDir, mhMin, mhMax, label=label, top_right=top_right, suffix=suffix)



def mHrange(mh_err: float | int) -> tuple[float | int, float | int]:
    if 1.5 * mh_err > 0.05:  # bound to 50 MeV
        return 124.95, 125.05
    return 125 - 1.5 * mh_err, 125 + 1.5 * mh_err
