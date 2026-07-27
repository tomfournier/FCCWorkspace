from __future__ import annotations

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
    'coefficients': (
        {'name': 'bern0', 'title': 'bern_coeff', 'value': 1, 'range': (-2,  2)},
        {'name': 'bern1', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern2', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern3', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
    ),
}


SYSTEMATIC_MODELS = {
    'BES': {
        'parameters': ('sigma', 'sigma_gt'),
        'process': _systematic_process,
    },
    'SQRTS': {
        'parameters': ('mean', 'mean_gt'),
        'process': _systematic_process,
    },
    'LEPSCALE': {
        'parameters': ('mean',),
        'process': _systematic_process,
    },
}


def get_signal_model(name: str = 'make_datacard') -> dict:
    return SIGNAL_MODELS[name]


def get_signal_processes(name: str, flavor: str, ecm: int) -> list[str]:
    return SIGNAL_MODELS[name]['processes'](flavor, ecm)


def get_background_processes(flavor: str, ecm: int) -> list[str]:
    return _background_processes(flavor, ecm)


def get_systematic_process(syst: str, flavor: str, ecm: int, direction: str) -> str:
    return _systematic_process(syst, flavor, ecm, direction)


def get_systematic_parameters(syst: str) -> tuple[str, ...]:
    return SYSTEMATIC_MODELS[syst]['parameters']
