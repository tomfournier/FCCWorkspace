import json, ROOT

from ..tools.process import getHist


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


def build_pdf_from_spec(
    recoilmass: ROOT.RooRealVar,
    params: dict[str, ROOT.RooRealVar],
    yield_zh: float | int,
    suffix: str,
    model_spec: dict,
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
    sig_fit = ROOT.RooAddPdf(
        _with_suffix(model_spec.get('model_name', 'zh_model'), suffix),
        '',
        _arg_list(sig),
        _arg_list(sig_norm),
    )

    return sig_fit, components, sig_norm

def get_hist(
        hName: str,
        inDir: str,
        procs: str | list[str],
        suffix: str,
        cat_idx_min: int,
        cat_idx_max: int,
        lumiScale: float | int,
        yield_nom: float | int,
        outName: str = '',
        normYields: bool = True):
    if isinstance(procs, str): procs = [procs]

    hist = getHist(hName, procs, inDir)
    hist.Scale(lumiScale)

    hist = hist.ProjectionX(f'hist_zh_{suffix}', cat_idx_min, cat_idx_max)
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
