from __future__ import annotations
from typing import Callable

from constants import Z_DECAYS


def sig_procs(ecm: int) -> dict[str, list[str]]:
    from constants import Z_DECAYS
    # Should be ordered: lower, nominal, high
    return {'high':    [f'wzp6_ee_{x}H_mH-higher-50MeV_ecm{ecm}' for x in Z_DECAYS],
            'nominal': [f'wzp6_ee_{x}H_ecm{ecm}'                 for x in Z_DECAYS],
            'low':     [f'wzp6_ee_{x}H_mH-lower-50MeV_ecm{ecm}'  for x in Z_DECAYS]}

def bkg_procs(ecm: int) -> list[str]:
    # Di-boson production processes: ee -> VV (V = W, Z)
    return [f'p8_ee_WW_ecm{ecm}', f'p8_ee_ZZ_ecm{ecm}',
            # Di-lepton production processes: ee -> ll (l = e, mu, tau)
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wz3p6_ee_mumu_ecm{ecm}', f'wz3p6_ee_tautau_ecm{ecm}',
            # Photon induced processes: ega -> eZ
            f'wzp6_egamma_eZ_Zee_ecm{ecm}',f'wzp6_gammae_eZ_Zee_ecm{ecm}',
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
            # Di-photon processes: gaga -> ll (l = e, mu, tau)
            f'wzp6_gaga_ee_60_ecm{ecm}', f'wzp6_gaga_mumu_60_ecm{ecm}', f'wzp6_gaga_tautau_60_ecm{ecm}',
            # ee -> nunuZ process
            f'wzp6_ee_nuenueZ_ecm{ecm}']


def syst_procs(syst: str, flavor: str, ecm: int, direction: str) -> str:
    
    if syst in {'SQRTS', 'LEPSCALE'} or (ecm == 365 and flavor == 'ee'):
        return [f'wzp6_ee_{x}H_ecm365' for x in Z_DECAYS]
    
    elif syst == 'BES':
        suffix = {'Up': 'higher', 'Dw': 'lower'}
        return [f'wzp6_ee_{x}_BES-{suffix[direction]}-1pc_ecm{ecm}' for x in Z_DECAYS]

    raise ValueError(f'{syst = } not supported, choose between [BES, SQRTS, LEPSCALE]')


SIGNAL_MODEL = {
    # same order as the processes: lower, nominal, higher
    'masses': [124.95, 125.0, 125.05],
    'param_names': ('mean', 'mean_gt', 'sigma', 'sigma_gt', 'mean1', 'mean_gt1', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'cb_1', 'cb_2'),
    'params': (
        # 'key' is the name used in param_config; sources must precede the formulas using them
        {'name': 'mean1',    'kind': 'source', 'key': 'mean_cb1'},
        {'name': 'sigma',    'kind': 'source', 'key': 'sigma_cb'},
        {'name': 'sigma_gt', 'kind': 'source', 'key': 'sigma_gt'},
        {'name': 'alpha_1',  'kind': 'source', 'key': 'alpha_1'},
        {'name': 'alpha_2',  'kind': 'source', 'key': 'alpha_2'},
        {'name': 'n_1',      'kind': 'source', 'key': 'n_1'},
        {'name': 'n_2',      'kind': 'source', 'key': 'n_2'},
        {'name': 'mean_gt1', 'kind': 'source', 'key': 'mean_gt1'},
        {'name': 'cb_1',     'kind': 'source', 'key': 'cb_1'},
        {'name': 'cb_2',     'kind': 'source', 'key': 'cb_2'},

        {'name': 'mean',     'kind': 'formula', 'expr': '@0+{mH}', 'args': ('mean1',)},
        {'name': 'mean_gt',  'kind': 'formula', 'expr': '@0+@1',   'args': ('mean', 'mean_gt1')},
    ),
    'components': (
        {'kind': 'RooCBShape',  'name': 'CrystallBall_1', 'title': 'CrystallBall_1', 'args': ('mrecoil', 'mean', 'sigma', 'alpha_1', 'n_1')},
        {'kind': 'RooCBShape',  'name': 'CrystallBall_2', 'title': 'CrystallBall_2', 'args': ('mrecoil', 'mean', 'sigma', 'alpha_2', 'n_2')},
        {'kind': 'RooGaussian', 'name': 'gauss',          'title': 'gauss',          'args': ('mrecoil', 'mean_gt', 'sigma_gt')},
    ),
    'fractions': ('cb_1', 'cb_2'),
    'processes': sig_procs,
    'yield_name': 'sig_norm', 'sum_name': 'sig', 'model_name': 'zh_model',
}


BACKGROUND_MODEL = {
    'order': 3,
    'name': 'bkg',
    'model_name': 'bkg_fit',
    'yield_name': 'bkg_norm_tmp',
    'sum_name': 'bkg',
    'params': (
        {'name': 'bern0', 'kind': 'variable', 'title': 'bern_coeff', 'value': 1, 'range': (-2,  2)},
        {'name': 'bern1', 'kind': 'variable', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern2', 'kind': 'variable', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern3', 'kind': 'variable', 'title': 'bern_coeff', 'value': 1, 'range': (-10, 10)},
        {'name': 'bern_coeffs', 'kind': 'arg_list', 'args': ('bern0', 'bern1', 'bern2')},
    ),
    'components': (
        {'kind': 'RooBernsteinFast', 'root_class': 'RooBernsteinFast', 'factory_args': (3,),
         'name': 'bkg', 'title': 'bkg', 'args': ('mrecoil', 'bern_coeffs')},
    ),
    'fractions': (),
    'processes': bkg_procs,
}


SYSTEMATIC_MODELS: dict[str, dict[str, str | tuple[str, ...] | dict[str, tuple] | Callable]] = {
    'BES': {
        'name': 'BES_ecmECM',
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('sigma', 'sigma_gt'),
        'range_overrides': lambda mH: {'sigma': (0, 5), 'sigma_gt': (0, 5)},
        'final_modifiers': {'sigma':    (('BES', 'sig_sigma_BES'),),
                            'sigma_gt': (('BES', 'sig_sigma_gt_BES'), ('SQRTS', 'sig_mean_gt_SQRTS')), },
        'process': syst_procs,
    },
    'SQRTS': {
        'name': 'SQRTS_ecmECM',
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('mean', 'mean_gt'),
        'range_overrides': lambda mH: {'mean': (mH - 5, mH + 5), 'mean_gt': (mH - 5, mH + 5)},
        'final_modifiers': {'mean':  (('LEPSCALE', 'sig_mean_LEPSCALE'), ('SQRTS', 'sig_mean_SQRTS')),
                            'mean_gt':  (('SQRTS', 'sig_mean_gt_SQRTS'),),
                            'sigma_gt': (('SQRTS', 'sig_mean_gt_SQRTS'),), },
        'process': syst_procs,
    },
    'LEPSCALE': {
        'name': 'LEPSCALE_FLAVOR_ecmECM',
        'suffix': {'Up': '_scaleup', 'Dw': '_scaledw'},
        'fit_param_names': ('mean', 'sigma', 'alpha_1', 'alpha_2', 'n_1', 'n_2', 'mean_gt', 'sigma_gt', 'cb_1', 'cb_2'),
        'parameters': ('mean',),
        'range_overrides': lambda mH: {'mean': (mH - 5, mH + 5)},
        'final_modifiers': {'mean': (('LEPSCALE', 'sig_mean_LEPSCALE'), ('SQRTS', 'sig_mean_SQRTS')), },
        'process': syst_procs,
    },
}
