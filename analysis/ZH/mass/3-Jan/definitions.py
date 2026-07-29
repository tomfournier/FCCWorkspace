from __future__ import annotations

import ROOT

def _signal_processes(flavor: str, ecm: int) -> list[str]:
    return [
        f'wzp6_ee_{flavor}H_ecm{ecm}',
        f'wzp6_ee_{flavor}H_mH-lower-50MeV_ecm{ecm}',
        f'wzp6_ee_{flavor}H_mH-higher-50MeV_ecm{ecm}',
    ]


def _background_processes(flavor: str, ecm: int) -> list[str]:
    return [
        f'p8_ee_WW_ecm{ecm}',
        f'p8_ee_ZZ_ecm{ecm}',
        f'wz3p6_ee_mumu_ecm{ecm}' if flavor == 'mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
        'wz3p6_ee_tautau_ecm240',
        f'wzp6_egamma_eZ_Z{flavor}_ecm{ecm}',
        f'wzp6_gammae_eZ_Z{flavor}_ecm{ecm}',
        f'wzp6_gaga_{flavor}_60_ecm{ecm}',
        f'wzp6_gaga_tautau_60_ecm{ecm}',
        f'wzp6_ee_nuenueZ_ecm{ecm}',
    ]


def _systematic_process(syst: str, flavor: str, ecm: int, direction: str) -> str:
    if syst == 'BES':
        pct = 1 if ecm == 240 else (10 if ecm == 365 else -1)
        if direction == 'Up':
            proc = f'wzp6_ee_{flavor}_BES-higher-{pct}pc_ecm{ecm}'
        else:
            proc = f'wzp6_ee_{flavor}_BES-lower-{pct}pc_ecm{ecm}'
        if ecm == 365 and flavor == 'ee':
            return 'wzp6_ee_eeH_ecm365'
        return proc

    if syst in {'SQRTS', 'LEPSCALE'}:
        return f'wzp6_ee_{flavor}H_ecm{ecm}'

    raise ValueError(f'{syst = } not supported, choose between [BES, SQRTS, LEPSCALE]')


def systematic_hist_suffix(syst: str, direction: str) -> str:
    return '_scaleup' if syst == 'LEPSCALE' and direction == 'Up' else ('_scaledw' if syst == 'LEPSCALE' else '')


SIGNAL_MODELS = {
    'parametrize_signal': {
        'masses': [124.95, 125.0, 125.05],
        'processes': _signal_processes,
        'param_names': ('mean_cb0', 'mean_cb1', 'sigma_cb', 'mean_gt', 'mean_gt1', 'sigma_gt',
                        'alpha_1', 'alpha_2', 'n_1', 'n_2', 'cb_1', 'cb_2'),
        'params': (
            {'name': 'mean_cb0', 'kind': 'mH'},
            {'name': 'mean_cb1', 'kind': 'source', 'key': 'mean_cb1'},
            {'name': 'mean_cb', 'kind': 'formula', 'expr': '@0+@1', 'args': ('mean_cb0', 'mean_cb1')},
            {'name': 'sigma_cb', 'kind': 'source', 'key': 'sigma_cb'},
            {'name': 'sigma_gt', 'kind': 'source', 'key': 'sigma_gt'},
            {'name': 'alpha_1', 'kind': 'source', 'key': 'alpha_1'},
            {'name': 'alpha_2', 'kind': 'source', 'key': 'alpha_2'},
            {'name': 'n_1', 'kind': 'source', 'key': 'n_1'},
            {'name': 'n_2', 'kind': 'source', 'key': 'n_2'},
            {'name': 'mean_gt1', 'kind': 'source', 'key': 'mean_gt1'},
            {'name': 'mean_gt', 'kind': 'formula', 'expr': '@0+@1', 'args': ('mean_cb', 'mean_gt1')},
            {'name': 'cb_1', 'kind': 'source', 'key': 'cb_1'},
            {'name': 'cb_2', 'kind': 'source', 'key': 'cb_2'},
        ),
        'components': (
            {'kind': 'cbshape', 'name': 'CB_1',  'title': 'CB_1',  'mean': 'mean_cb', 'sigma': 'sigma_cb', 'alpha': 'alpha_1', 'n': 'n_1'},
            {'kind': 'cbshape', 'name': 'CB_2',  'title': 'CB_2',  'mean': 'mean_cb', 'sigma': 'sigma_cb', 'alpha': 'alpha_2', 'n': 'n_2'},
            {'kind': 'gauss',   'name': 'gauss', 'title': 'gauss', 'mean': 'mean_gt', 'sigma': 'sigma_gt'},
        ),
        'fractions': ('cb_1', 'cb_2'),
        'yield_name': 'yield',
        'sum_name': 'sig',
        'model_name': 'zh_model',
    },
    'make_datacard': {
        'masses': [125.0, 124.95, 125.05],
        'processes': _signal_processes,
        'param_names': ('mean', 'mean_gt', 'sigma', 'sigma_gt', 'mean0', 'mean1', 'mean_gt1', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'cb_1', 'cb_2'),
        'params': (
            {'name': 'mean0', 'kind': 'source', 'key': 'mean0', 'idx': 1, 'lo': 0.5, 'hi': 1.5},
            {'name': 'mean1', 'kind': 'source', 'key': 'mean_cb1', 'idx': 1, 'lo': -1, 'hi': 1},
            {'name': 'mean', 'kind': 'formula', 'expr': '@1+@0*{mH}', 'args': ('mean0', 'mean1')},
            {'name': 'sigma', 'kind': 'source', 'key': 'sigma_cb', 'idx': 1, 'lo': 0, 'hi': 10},
            {'name': 'sigma_gt', 'kind': 'source', 'key': 'sigma_gt', 'idx': 1, 'lo': 0, 'hi': 10},
            {'name': 'alpha_1', 'kind': 'source', 'key': 'alpha_1', 'idx': 1, 'lo': -5, 'hi': 5},
            {'name': 'alpha_2', 'kind': 'source', 'key': 'alpha_2', 'idx': 1, 'lo': -5, 'hi': 5},
            {'name': 'n_1', 'kind': 'source', 'key': 'n_1', 'idx': 1, 'lo': -50, 'hi': 50},
            {'name': 'n_2', 'kind': 'source', 'key': 'n_2', 'idx': 1, 'lo': -50, 'hi': 50},
            {'name': 'mean_gt1', 'kind': 'source', 'key': 'mean_gt1', 'idx': 1, 'lo': -2, 'hi': 2},
            {'name': 'mean_gt', 'kind': 'formula', 'expr': '@0 + @1', 'args': ('mean', 'mean_gt1')},
            {'name': 'cb_1', 'kind': 'source', 'key': 'cb_1', 'idx': 1, 'lo': 0, 'hi': 1},
            {'name': 'cb_2', 'kind': 'source', 'key': 'cb_2', 'idx': 1, 'lo': 0, 'hi': 1},
        ),
        'components': (
            {'kind': 'cbshape', 'name': 'CrystallBall_1', 'title': 'CrystallBall_1', 'mean': 'mean',    'sigma': 'sigma', 'alpha': 'alpha_1', 'n': 'n_1'},
            {'kind': 'cbshape', 'name': 'CrystallBall_2', 'title': 'CrystallBall_2', 'mean': 'mean',    'sigma': 'sigma', 'alpha': 'alpha_2', 'n': 'n_2'},
            {'kind': 'gauss',   'name': 'gauss',          'title': 'gauss',          'mean': 'mean_gt', 'sigma': 'sigma_gt'},
        ),
        'fractions': ('cb_1', 'cb_2'),
        'yield_name': 'sig_norm',
        'sum_name': 'sig',
        'model_name': 'zh_model',
    },
}


BACKGROUND_MODEL = {
    'order': 3,
    'name': 'bkg',
    'model_name': 'bkg_fit',
    'yield_name': 'bkg_norm_tmp',
    'processes': _background_processes,
    'coefficients': (
        {'name': 'bern0', 'title': 'bern_coeff', 'value': 1, 'range': (-2,  2)},
        {'name': 'bern1', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern2', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern3', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
    ),
}


SYSTEMATIC_MODELS = {
    'BES': {
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('sigma', 'sigma_gt'),
        'final_modifiers': {
            'sigma': (('BES', 'sig_sigma_BES'),),
            'sigma_gt': (('BES', 'sig_sigma_gt_BES'), ('SQRTS', 'sig_mean_gt_SQRTS')),
        },
        'process': _systematic_process,
    },
    'SQRTS': {
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('mean', 'mean_gt'),
        'final_modifiers': {
            'mean': (('LEPSCALE', 'sig_mean_LEPSCALE'), ('SQRTS', 'sig_mean_SQRTS')),
            'mean_gt': (('SQRTS', 'sig_mean_gt_SQRTS'),),
            'sigma_gt': (('SQRTS', 'sig_mean_gt_SQRTS'),),
        },
        'process': _systematic_process,
    },
    'LEPSCALE': {
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('mean',),
        'final_modifiers': {
            'mean': (('LEPSCALE', 'sig_mean_LEPSCALE'), ('SQRTS', 'sig_mean_SQRTS')),
        },
        'process': _systematic_process,
    },
}


def make_var_dict(model_specs: dict, extra: tuple[str, ...] = ()) -> dict[str, list]:
    return {key: [] for key in (*model_specs['param_names'], *extra)}


def get_systematic_parameters(syst: str) -> tuple[str, ...]:
    return SYSTEMATIC_MODELS[syst]['parameters']


def get_systematic_fit_param_names(syst: str) -> tuple[str, ...]:
    return SYSTEMATIC_MODELS[syst]['fit_param_names']


def get_systematic_range_overrides(syst: str, mH: float | int) -> dict[str, tuple[float | int, float | int]]:
    if syst == 'BES':
        return {'sigma': (0, 5), 'sigma_gt': (0, 5)}
    if syst == 'SQRTS':
        return {'mean': (mH - 5, mH + 5), 'mean_gt': (mH - 5, mH + 5)}
    if syst == 'LEPSCALE':
        return {'mean': (mH - 5, mH + 5)}
    raise ValueError(f'{syst = } not supported, choose between [BES, SQRTS, LEPSCALE]')


def make_systematic_fit_vars(
        workspace: ROOT.RooWorkspace,
        syst: str,
        mH: float | int,
        mH_label: str,
         ) -> tuple[dict[str, ROOT.RooRealVar], list[object], tuple[str, ...]]:

    fit_param_names = get_systematic_fit_param_names(syst)
    spline_map = {name: workspace.obj(f'spline_{name}') for name in fit_param_names}
    overrides = get_systematic_range_overrides(syst, mH)

    vars_dict: dict[str, ROOT.RooRealVar] = {}
    for name in fit_param_names:
        value = spline_map[name].getVal()
        if name in overrides:
            lo, hi = overrides[name]
            vars_dict[name] = ROOT.RooRealVar(f'{name}_{mH_label}_{syst}', '', value, lo, hi)
        else:
            vars_dict[name] = ROOT.RooRealVar(f'{name}_{mH_label}_{syst}', '', value)

    return vars_dict, [spline_map[name] for name in get_systematic_parameters(syst)], get_systematic_parameters(syst)


def _arg_list(*items):
    arg_list = ROOT.RooArgList()
    for item in items:
        arg_list.add(item)
    return arg_list


def _model_param_names(name: str) -> tuple[str, ...]:
    spec = SIGNAL_MODELS[name]
    ordered: list[str] = []
    for component in spec['components']:
        for key in ('mean', 'sigma', 'alpha', 'n'):
            param_name = component.get(key)
            if param_name and param_name not in ordered:
                ordered.append(param_name)
    for fraction in spec['fractions']:
        if fraction not in ordered:
            ordered.append(fraction)
    return tuple(ordered)


def _systematic_strength_name(syst: str, flavor: str, ecm: int) -> str:
    if syst == 'BES':
        return f'BES_ecm{ecm}'
    if syst == 'SQRTS':
        return f'SQRTS_ecm{ecm}'
    if syst == 'LEPSCALE':
        flav = 'MU' if flavor == 'mumu' else ('EL' if flavor == 'ee' else flavor)
        return f'LEPSCALE_{flav}_ecm{ecm}'
    raise ValueError(f'{syst = } not supported, choose between [BES, SQRTS, LEPSCALE]')


def build_datacard_signal_params(
        workspace: ROOT.RooWorkspace,
        signal_name: str,
        flavor: str,
        ecm: int,
        use_syst: bool = True,
         ) -> dict[str, ROOT.RooAbsArg]:

    spec = SIGNAL_MODELS[signal_name]
    params: dict[str, ROOT.RooAbsArg] = {}

    for param_name in _model_param_names(signal_name):
        base = workspace.obj(f'spline_{param_name}')
        modifiers = []
        if use_syst:
            for syst_name, nuisance_name in spec.get('final_modifiers', {}).get(param_name, ()):
                strength_name = _systematic_strength_name(syst_name, flavor, ecm)
                modifiers.append((workspace.obj(strength_name), workspace.obj(nuisance_name)))

        if not modifiers:
            params[param_name] = ROOT.RooFormulaVar(param_name, '', '@0', _arg_list(base))
            continue

        expr = '@0'
        args = [base]
        for strength, nuisance in modifiers:
            expr += f'*(1+@{len(args)}*@{len(args) + 1})'
            args.extend([strength, nuisance])
        params[param_name] = ROOT.RooFormulaVar(param_name, '', expr, _arg_list(*args))

    return params


def build_datacard_signal_pdf(
        workspace: ROOT.RooWorkspace,
        signal_name: str,
        recoilmass: ROOT.RooRealVar,
        flavor: str,
        ecm: int,
        use_syst: bool = True,
         ):

    from package.func.fit import build_pdf_from_spec

    params = build_datacard_signal_params(workspace, signal_name, flavor, ecm, use_syst)
    return build_pdf_from_spec(recoilmass, params, 1.0, '', SIGNAL_MODELS[signal_name], extended=False)[0]
