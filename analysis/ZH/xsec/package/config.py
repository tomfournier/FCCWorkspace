'''Core configuration for the FCC-ee ZH cross-section analysis.

Provides:
- Feature set for BDT training: `input_vars`.
- Decay mode enumerations: `Z_DECAYS`, `H_DECAYS`, `H_DECAYS_WITH_INV`, `H_DECAYS_ALL`, `QUARKS`
    plus lowercase aliases for backward compatibility.
- Color palettes for ROOT and matplotlib: `colors`, `h_colors`, `modes_color`.
- Physics and axis labels (ROOT TLatex and LaTeX): `labels`, `h_labels`,
    `vars_label`, `vars_xlabel`, `modes_label`, `process_label`.
- Process builders:
    - `get_process_dict()`: Simple process dictionary builder with optional filtering.
    - `get_process_list()`: Full-featured process builder with signal/background handling.
- Background/signal process construction for analysis workflows.
- Utilities: `warning()` for formatted exceptions and `timer()` for timing output.

Conventions:
- Process naming follows FCC patterns, e.g. ``wzp6_ee_{z}H_H{h}_ecm{ecm}``,
    ``p8_ee_WW_ecm{ecm}``.
- Labels use ROOT TLatex syntax for ROOT displays and LaTeX for matplotlib.
- Units are appended in `vars_xlabel` (e.g., GeV, GeV^2).

Usage:
- Simple process dictionary: ``get_process_dict(procs=['ZH','WW'], ecm=365)``.
- Full analysis workflow: ``get_process_list(cat='mumu', ecm=240, train=True)``.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from time import time
from typing import Sequence, Union

from .logger import get_logger

LOGGER = get_logger(__name__)



#############################
##### VARIABLES FOR BDT #####
#############################

# Tuple of kinematic variables used as input features for BDT training
input_vars_ll = (
    'leading_p',    'leading_theta',
    'subleading_p', 'subleading_theta',
    'acolinearity', 'acoplanarity',
    'zll_m', 'zll_p', 'zll_theta'
)

input_vars_qq = (
    'leading_p',    'leading_costheta',
    'subleading_p', 'subleading_costheta',
    'acolinearity', 'acoplanarity',
    'zqq_p',        'zqq_costheta',
)



##########################
### Z AND HIGGS DECAYS ###
##########################

# Standard Z boson decay modes
Z_DECAYS: tuple[str, ...] = ('bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu')

# Standard Higgs boson decay modes
H_DECAYS: tuple[str, ...] = ('bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa')

# Higgs decays use to make the fit
H_DECAYS_FIT: tuple[str, ...] = ('bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'ZZ_noInv', 'WW', 'Za', 'aa', 'inv')

# Higgs decay modes including invisible decays
H_DECAYS_WITH_INV: tuple[str, ...] = H_DECAYS + ('inv',)

H_DECAYS_ALL: tuple[str, ...] = H_DECAYS + ('inv', 'ZZ_noInv',)

# Quark decay channels
QUARKS: tuple[str, ...] = ('bb', 'cc', 'ss', 'qq')

# Lowercase aliases for backward compatibility
z_decays = Z_DECAYS
h_decays = H_DECAYS
H_decays = H_DECAYS_WITH_INV
quarks   = QUARKS



#######################
### PROCESSES COLOR ###
#######################

# Lazy import of ROOT - only load when colors are accessed
_ROOT = None

def _get_root():
    """Lazily import ROOT.

    Defers ROOT import until it is actually accessed to avoid unnecessary overhead.

    Returns:
        The ROOT module.
    """
    global _ROOT
    if _ROOT is None:
        LOGGER.info('Loading ROOT...')
        import ROOT
        _ROOT = ROOT
    return _ROOT


# Lazy-loaded ROOT colors - these are computed on first access
# ROOT color indices (lazily initialized on first access)
_ZH_COLOR   = None   # Red for ZH signal
_WW_COLOR   = None   # Orange for WW background
_ZZ_COLOR   = None   # Blue for ZZ background
_ZG_COLOR   = None   # Purple for Z/gamma
_RARE_COLOR = None   # Gray for rare processes

def _init_colors() -> None:
    """Initialize ROOT color objects on first access.

    Creates ROOT color indices for signal and background processes.
    Called automatically by _get_colors_dict() on first use.
    """
    global _ZH_COLOR, _WW_COLOR, _ZZ_COLOR, _ZG_COLOR, _RARE_COLOR, _TT_COLOR
    if _ZH_COLOR is None:
        root = _get_root()
        _ZH_COLOR   = root.TColor.GetColor('#e42536')  # Red       for ZH signal
        _WW_COLOR   = root.TColor.GetColor('#f89c20')  # Orange    for WW background
        _ZZ_COLOR   = root.TColor.GetColor('#5790fc')  # Blue      for ZZ background
        _ZG_COLOR   = root.TColor.GetColor('#964a8b')  # Purple    for Z/gamma
        _RARE_COLOR = root.TColor.GetColor('#9c9ca1')  # Gray      for rare processes
        _TT_COLOR   = root.TColor.GetColor("#1414ad")  # Dark blue for tt processes

def _get_h_colors_dict() -> dict:
    """Lazy-load h_colors with color constants.

    Returns:
        Dictionary mapping decay modes to color codes.
    """

    root = _get_root()
    return {
        'bb'     : root.kViolet,
        'cc'     : root.kBlue,
        'ss'     : root.kRed,
        'gg'     : root.kGreen+1,
        'mumu'   : root.kOrange,
        'tautau' : root.kCyan,
        'ZZ'     : root.kGray,
        'WW'     : root.kGray+2,
        'Za'     : root.kGreen+2,
        'aa'     : root.kRed+2,
        'inv'    : root.kBlue+2
    }

def _get_colors_dict() -> dict:
    """Lazy-load colors dictionary with color constants.

    Returns:
        Dictionary mapping process names to color codes.
    """
    _init_colors()
    return {
        'ZH'       : _ZH_COLOR,
        'ZeeH'     : _ZH_COLOR,
        'ZmumuH'   : _ZH_COLOR,
        'ZqqH'     : _ZH_COLOR,
        'ZnunuH'   : _ZH_COLOR,

        'zh'       : _ZH_COLOR,
        'zeeh'     : _ZH_COLOR,
        'zmumuh'   : _ZH_COLOR,
        'zqqh'     : _ZH_COLOR,
        'znunuh'   : _ZH_COLOR,

        'WW'       : _WW_COLOR,
        'ZZ'       : _ZZ_COLOR,
        'Zgamma'   : _ZG_COLOR,
        'Zqqgamma' : _ZG_COLOR,
        'Rare'     : _RARE_COLOR,
        'tt'       : _TT_COLOR
    }


class LazyColorDict(dict):
    """Dictionary proxy that initializes ROOT colors on first access."""

    def __init__(self, builder):
        super().__init__()
        self._builder = builder

    def _ensure(self):
        if not self:
            self.update(self._builder())

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def items(self):
        self._ensure()
        return super().items()

    def keys(self):
        self._ensure()
        return super().keys()

    def values(self):
        self._ensure()
        return super().values()


# Maps decay modes and process names to ROOT color indices while keeping the
# import side-effect free. Callers can still use colors['ZH'] and h_colors['bb']
# without re-defining anything in each script.
h_colors = LazyColorDict(_get_h_colors_dict)  # Decay mode   -> ROOT color
colors   = LazyColorDict(_get_colors_dict)    # Process name -> ROOT color

# Matplotlib tab colors for different analysis modes by channel (no lazy loading needed)
modes_color = {
    'ZmumuH':      'tab:blue',
    'ZZ':          'tab:orange',
    'Zmumu':       'tab:red',
    'WWmumu':      'tab:green',
    'egamma_mumu': 'tab:purple',
    'gammae_mumu': 'tab:brown',
    'gaga_mumu':   'tab:pink',

    'ZeeH':        'tab:blue',
    'Zee':         'tab:red',
    'WWee':        'tab:green',
    'egamma_ee':   'tab:purple',
    'gammae_ee':   'tab:brown',
    'gaga_ee':     'tab:pink',

    'ZqqH':        'tab:blue',
    'Zqq':         'tab:red',
    'WWqq':        'tab:green',
    'egamma_qq':   'tab:purple',
    'gammae_qq':   'tab:brown',
    'gaga_qq':     'tab:pink',

    'ttbar':       'tab:olive'
}



#######################
### PROCESSES LABEL ###
#######################

# ROOT TLatex labels for Z decay modes
z_labels = {
    'bb'     : 'Z#rightarrowb#bar{b}',
    'cc'     : 'Z#rightarrowc#bar{c}',
    'ss'     : 'Z#rightarrows#bar{s}',
    'qq'     : 'Z#rightarrowq#bar{q}',
    'ee'     : 'Z#rightarrowe^{#plus}e^{#minus}',
    'mumu'   : 'Z#rightarrow#mu^{#plus}#mu^{#minus}',
    'tautau' : 'Z#rightarrow#tau^{#plus}#tau^{#minus}',
    'nunu'   : 'Z#rightarrow#nu#bar{#nu}',
}

# ROOT TLatex labels for Higgs decay modes
h_labels = {
    'bb'     : 'H#rightarrowb#bar{b}',
    'cc'     : 'H#rightarrowc#bar{c}',
    'ss'     : 'H#rightarrows#bar{s}',
    'gg'     : 'H#rightarrowgg',
    'mumu'   : 'H#rightarrow#mu^{#plus}#mu^{#minus}',
    'tautau' : 'H#rightarrow#tau^{#plus}#tau^{#minus}',
    'ZZ'     : 'H#rightarrowZZ*',
    'WW'     : 'H#rightarrowWW*',
    'Za'     : 'H#rightarrowZ#gamma',
    'aa'     : 'H#rightarrow#gamma#gamma',
    'inv'    : 'H#rightarrowInv'
}

H_labels = {
    'bb'     : r'$H\to b\bar{b}$',
    'cc'     : r'$H\to c\bar{c}$',
    'ss'     : r'$H\to s\bar{s}$',
    'gg'     : r'$H\to gg$',
    'mumu'   : r'$H\to \mu^+\mu^-$',
    'tautau' : r'$H\to \tau^+\tau^-$',
    'ZZ'     : r'$H\to ZZ^*$',
    'WW'     : r'$H\to WW^*$',
    'Za'     : r'$H\to Z\gamma$',
    'aa'     : r'$H\to \gamma\gamma$',
    'inv'    : r'$H\to$ Inv'
}

# ROOT TLatex labels for main physics processes
labels = {
    'ZH'     : 'ZH',
    'ZmumuH' : 'Z(#mu^{+}#mu^{#minus})H',
    'ZeeH'   : 'Z(e^{+}e^{#minus})H',
    'ZqqH'   : 'Z(q#bar{q})H',

    'zh'     : 'ZH',
    'zmumuh' : 'Z(#mu^{+}#mu^{#minus})H',
    'zeeh'   : 'Z(e^{+}e^{#minus})H',
    'zqqh'   : 'Z(q#bar{q})H',

    'WW'     : 'W^{+}W^{-}',
    'ZZ'     : 'ZZ',
    'Zgamma' : 'Z/#gamma^{*} #rightarrow f#bar{f}+#gamma(#gamma)',
    'Rare'   : 'Rare',
    'tt'     : 't#bar{t}'
}

# LaTeX labels for kinematic variables with units
vars_xlabel_ll = {
    'leading_e':        r'$E_{\ell,leading}$ [GeV]',
    'leading_p':        r'$p_{\ell,leading}$ [GeV]',
    'leading_pT':       r'$p_{T,leading}$ [GeV]',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'leading_phi':      r'$\phi_{\ell,leading}$',

    'subleading_e':     r'$E_{\ell,subleading}$ [GeV]',
    'subleading_p':     r'$p_{\ell,subleading}$ [GeV]',
    'subleading_pT':    r'$p_{T,subleading}$ [GeV]',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'subleading_phi':   r'$\phi_{\ell,subleading}$',

    'acolinearity':     r'$\Delta\alpha_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'acopolarity':      r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'deltaR':           r'$\Delta R$',

    'zll_m':            r'$m_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_e':            r'$E_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_p':            r'$p_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_pT':           r'$p_{T, \ell^{+}\ell^{-}}$ [GeV]',
    'zll_theta':        r'$\theta_{\ell^{+}\ell^{+}}$',
    'zll_costheta':     r'$\cos\theta_{\ell^{+}\ell^{+}}$',
    'zll_phi':          r'$\phi_{\ell^{+}\ell^{-}}$',

    'zll_recoil_m':     r'$m_{recoil}$ [GeV]',
    'cosTheta_miss':    r'$\cos\theta_{miss}$',

    'visibleEnergy':    r'$E_{vis}$ [GeV]',
    'missingEnergy':    r'$E_{miss}$ [GeV]',
    'missingMass':      r'$m_{miss}$ [GeV]',

    'H':                r'$H$ [GeV]',
    'BDTscore':         r'BDT Score',

    'leps_iso':         r'$I_{rel}$',
    'leps_iso_no':      r'Isolated leptons'
}

vars_xlabel_qq = {
    'leading_e':             r'$E_{jet,leading}$ [GeV]',
    'leading_p':             r'$p_{jet,leading}$ [GeV]',
    'leading_pT':            r'$p_{T,leading}$ [GeV]',
    'leading_theta':         r'$\theta_{jet,leading}$',
    'leading_costheta':      r'$\cos\theta_{jet,leading}$',
    'leading_phi':           r'$\phi_{jet,leading}$',

    'subleading_e':          r'$E_{jet,subleading}$ [GeV]',
    'subleading_p':          r'$p_{jet,subleading}$ [GeV]',
    'subleading_pT':         r'$p_{T,subleading}$ [GeV]',
    'subleading_theta':      r'$\theta_{jet,subleading}$',
    'subleading_costheta':   r'$\cos\theta_{jet,subleading}$',
    'subleading_phi':        r'$\phi_{jet,subleading}$',

    'acolinearity':          r'$\Delta\alpha_{jj}$',
    'acoplanarity':          r'$\pi - \Delta\phi_{jj}$',
    'acopolarity':           r'$\Delta\theta_{jj}$',
    'deltaR':                r'$\Delta R$',

    'zqq_m':                 r'$m_{jj}$ [GeV]',
    'zqq_e':                 r'$E_{jj}$ [GeV]',
    'zqq_p':                 r'$p_{jj}$ [GeV]',
    'zqq_pT':                r'$p_{T,jj}$ [GeV]',
    'zqq_theta':             r'$\theta_{jj}$',
    'zqq_costheta':          r'$\cos\theta_{jj}$',
    'zqq_phi':               r'$\phi_{jj}$',

    'delta_mWW':             r'$\Delta m_{WW}$ [GeV]',
    'delta_mWW4':            r'$\Delta m_{WW}$ (4 jets algo) [GeV]',

    'thrust':                r'$T$',
    'thrust_costheta':       r'$\cos\theta_{T}$',

    'zqq_recoil_m':          r'$m_{recoil}$ [GeV]',
    'cosTheta_miss':         r'$\cos\theta_{miss}$',

    'visibleEnergy':         r'$E_{vis}$ [GeV]',
    'missingEnergy':         r'$E_{miss}$ [GeV]',
    'missingMass':           r'$m_{miss}$ [GeV]',

    'BDTscore':              r'BDT Score',

    'best_clustering_idx':   'Best clustering algorithm',
    'best_cluster_idx':      'Best clustering algorithm',
    'njets_inclusive':       'Number of jets (inclusive)',
    'njets_incl':            'Number of jets (inclusive)',
    'njets':                 r'n_{jets}',

    'H':                     r'$H$ [GeV]'
}

# LaTeX x-axis labels without units
vars_label_ll = {k: v.replace(' [GeV]', '') for k, v in vars_xlabel_ll.items()}
vars_label_qq = {k: v.replace(' [GeV]', '') for k, v in vars_xlabel_qq.items()}


# LaTeX labels for analysis modes (physics processes)
modes_label = {
    'ZmumuH':      r'$e^+e^-\rightarrow Z(\mu^+\mu^-)H$',
    'ZZ':          r'$e^+e^-\rightarrow ZZ$',
    'Zmumu':       r'$e^+e^-\rightarrow Z/\gamma^{*}\rightarrow\mu^+\mu^-$',
    'WWmumu':      r'$e^+e^-\rightarrow W^{+}W^{-}[\nu_{\mu}\mu]$',
    'egamma_mumu': r'$e^-\gamma\rightarrow e^-Z(\mu^+\mu^-)$',
    'gammae_mumu': r'$e^+\gamma\rightarrow e^+Z(\mu^+\mu^-)$',
    'gaga_mumu':   r'$\gamma\gamma\rightarrow\mu^+\mu^-$',

    'ZeeH':        r'$e^+e^-\rightarrow Z(e^+e^-)H$',
    'Zee':         r'$e^+e^-\rightarrow Z/\gamma^{*}\rightarrow e^+e^-$',
    'WWee':        r'$e^+e^-\rightarrow W^{+}W^{-}[\nu_{e}e]$',
    'egamma_ee':   r'$e^-\gamma\rightarrow e^-Z(e^+e^-)$',
    'gammae_ee':   r'$e^+\gamma\rightarrow e^+Z(e^+e^-)$',
    'gaga_ee':     r'$\gamma\gamma\rightarrow e^+e^-$',

    'ZqqH':        r'$e^+e^-\rightarrow Z(q\bar{q})H$',
    'Zqq':         r'$e^+e^-\rightarrow Z/\gamma^{*}\rightarrow q\bar{q}$',
    'WWqq':        r'$e^+e^-\rightarrow W^{+}W^{-}[had]$',
    'egamma_qq':   r'$e^-\gamma\rightarrow e^-Z(q\bar{q})$',
    'gammae_qq':   r'$e^+\gamma\rightarrow e^+Z(q\bar{q})$',
    'gaga_qq':     r'$\gamma\gamma\rightarrow q\bar{q}$',

    'ttbar':       r'$e^+ e^-\rightarrow t\bar{t}$'
}

process_label = {
    'bb':       r'b\bar{b}',
    'cc':       r'c\bar{c}',
    'ss':       r's\bar{s}',
    'gg':       r'gg',
    'mumu':     r'\mu^{+}\mu^{-}',
    'tautau':   r'\tau^{+}\tau^{-}',
    'WW':       r'WW^{*}',
    'ZZ':       r'ZZ^{*}',
    'ZZ_noInv': r'ZZ^{*}(No Inv)',
    'Za':       r'Z\gamma',
    'aa':       r'\gamma\gamma',
    'inv':      r'Inv'
}



########################
### CONFIG FUNCTIONS ###
########################

# __________________
def timer(t: float
          ) -> None:
    '''Log formatted elapsed time since provided timestamp.

    Calculates and logs elapsed time in human-readable format (hours, minutes,
    seconds, milliseconds) with formatted header and footer separators.

    Args:
        t: Starting timestamp from time.time().
    '''
    dt = time() - t

    # Split time into components
    h, m  = int(dt // 3600), int(dt // 60 % 60),
    s, ms = int(dt % 60), int((dt % 1) * 1000)

    # Build time string with non-zero components
    time_parts = []
    if h>0:
        time_parts.append(f'{h} h')
    if m>0:
        time_parts.append(f'{m} min')
    if s>0:
        time_parts.append(f'{s} s')
    if ms>0:
        time_parts.append(f'{ms} ms')
    if not time_parts:
        time_parts.append('0 ms')

    elapsed = f"Elapsed time: {' '.join(time_parts)}"
    lenght = len(elapsed) + 4

    LOGGER.info(f'\n{" CODE ENDED ":=^{lenght}}\n{elapsed:^{lenght}}\n{"="*lenght}\n')



##########################
### PROCESSES FUNCTION ###
##########################

def get_process_dict(
        procs:    Union[Sequence[str], None] = None,
        ecm: int = 240,
        z_decays: Union[Sequence[str], None] = None,
        h_decays: Union[Sequence[str], None] = None,
        H_decays: Union[Sequence[str], None] = None,
        quarks:   Union[Sequence[str], None] = None,
     ) -> dict[str, tuple[str, ...]]:
    '''Generate process dictionary with optional filtering and custom decay modes.

    Simple process builder for creating FCC sample dictionaries.
    Can use defaults (cached) or custom decay modes. Optionally filters to specific process keys.
    Returns process key -> sample names mapping (e.g., 'ZH' -> ('wzp6_ee_bbH_Hbb_ecm240', ...)).

    Args:
        procs: Process keys to include. If None, returns all available processes.
        z_decays: Z decay modes. Uses Z_DECAYS if None.
        h_decays: Higgs decay modes (no invisible). Uses H_DECAYS if None.
        H_decays: Higgs decay modes (with invisible). Uses H_DECAYS_WITH_INV if None.
        quarks: Quark channels. Uses QUARKS if None.
        ecm: Center-of-mass energy in GeV (default 240).

    Returns:
        Dictionary mapping process keys to tuples of FCC sample names.

    Examples:
        >>> get_process_dict()  # All processes, default decays, 240 GeV
        >>> get_process_dict(procs=['ZH', 'WW'], ecm=365)  # Filtered, 365 GeV
        >>> get_process_dict(h_decays=['bb', 'cc'])  # Custom Higgs decays
    '''
    z_set = Z_DECAYS          if z_decays is None else tuple(z_decays)
    h_set = H_DECAYS          if h_decays is None else tuple(h_decays)
    H_set = H_DECAYS_WITH_INV if H_decays is None else tuple(H_decays)
    q_set = QUARKS            if quarks   is None else tuple(quarks)

    processes = {
        # All signals for the Z and Higgs exclusive decay
        'ZH':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set for y in h_set),

        # All signals for a specific Z decays and Higgs exclusive decay
        'ZeeH':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_set),
        'ZmumuH': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_set),
        'ZqqH':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set for y in h_set),

        # All signals for the Z and Higgs exclusive decay (Include invisible decay)
        'zh':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set for y in H_set),

        # All signals for a specific Z decays and Higgs exclusive decay (Include invisible decay)
        'zeeh':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in H_set),
        'zmumuh': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in H_set),
        'zqqh':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set for y in H_set),

        # Diboson production e+e- -> VV (V = W or Z)
        'WW':     (f'p8_ee_WW_ee_ecm{ecm}', f'p8_ee_WW_mumu_ecm{ecm}', f'p8_ee_WW_ecm{ecm}'),
        'ZZ':     (f'p8_ee_ZZ_ecm{ecm}',),

        # 2 fermion production e+e- -> ff
        'Zgamma': (f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                   f'wzp6_ee_tautau_ecm{ecm}',        f'wzp6_ee_qq_ecm{ecm}'),

        # Rare processes: photon induced, diphoton and nunuZ processes
        'Rare':   (f'wzp6_gammae_eZ_Zee_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}',
                   f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',  f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
                   f'wzp6_gammae_eZ_Zqq_ecm{ecm}',    f'wzp6_egamma_eZ_Zqq_ecm{ecm}',
                   f'wzp6_gaga_ee_60_ecm{ecm}',       f'wzp6_gaga_mumu_60_ecm{ecm}',
                   f'wzp6_gaga_tautau_60_ecm{ecm}',   f'wzp6_ee_nuenueZ_ecm{ecm}'),
    }
    if ecm == 365:
        # Include e+e- -> tt process for ecm = 365 GeV
        processes['tt'] = ('wzp6_ee_WbWb_ecm365',)

    if procs:
        return {proc: processes[proc] for proc in procs if proc in processes}
    return processes


def get_process_list(
        cat: str,
        ecm: int,
        z_decays: tuple[str, ...] = Z_DECAYS,
        h_decays: tuple[str, ...] = H_DECAYS_ALL,
        quarks: tuple[str, ...] = QUARKS,
        train: bool = False,
        batch: bool = False,
        onlysig: bool = False,
        onlybkg: bool = False,
        frac: dict[str, float] | None = None,
        chunks: dict[str, int] | None = None,
        include: dict[str, dict] | None = None,
        exclude: set[str] | None = None,
        all_train_sig: bool = True
         ) -> dict[str, dict[str, float | int]]:
    '''Generate analysis-ready process dictionary with signals and backgrounds.

    Full-featured process builder for analysis workflows. Combines signal and background
    samples with event counts and fractions. Training mode uses simplified samples.
    Supports filtering, custom overrides, and batch mode scaling.

    Args:
        cat: Category ('ee', 'mumu', 'qq').
        ecm: Center-of-mass energy in GeV (240 or 365).
        z_decays: Z decay modes (non-training mode only; training uses defaults).
        h_decays: Higgs decay modes (non-training mode only; training uses defaults).
        train: If True, use training-mode samples (category-specific backgrounds).
        batch: If True, scale chunk sizes for batch processing.
        onlysig: Return only signal processes (mutually exclusive with onlybkg).
        onlybkg: Return only background processes (mutually exclusive with onlysig).
        frac: Custom fractions by sample name (overrides defaults).
        chunks: Custom event chunk counts by sample name (overrides defaults).
        include: Additional processes to add, dict with 'sig' and/or 'bkg' keys.
        exclude: Set of sample names to exclude from output.

    Returns:
        Dictionary mapping sample names to {'fraction': float, 'chunks': int}.

    Raises:
        ValueError: If onlysig and onlybkg are both True.
    '''
    # Initialize optional parameters
    frac    = frac    or    {}
    chunks  = chunks  or    {}
    include = include or    {}
    exclude = exclude or set()

    # Validate conflicting options
    if onlysig and onlybkg:
        raise ValueError('Cannot set both onlysig and onlybkg to True. Choose one.')

    if train:
        if cat in ['ee', 'mumu']:
            sigs = [f'wzp6_ee_{cat}H_ecm{ecm}']
            if all_train_sig:
                sigs += [f'wzp6_ee_{cat}H_H{y}_ecm{ecm}' for y in h_decays if 'noInv' not in y]
        elif cat == 'qq':
            sigs = [f'wzp6_ee_{x}H_ecm{ecm}' for x in quarks]
            if all_train_sig:
                sigs += [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in quarks for y in h_decays if 'noInv' not in y]
        else:
            raise ValueError(f'{cat} is not a valid category. Use [ee, mumu, qq].')
    else:
        sigs = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays]

    small, middle, big = ((5, 5, 10) if batch else (1, 5, 10)) if train \
        else ((5, 20, 30) if batch else (1, 5, 10))
    common = {f'p8_ee_ZZ_ecm{ecm}': {'frac': 0.25 if cat == 'qq' else 1, 'nb': middle}}
    if not train or cat == 'qq':
        common[f'p8_ee_WW_ecm{ecm}'] = {'frac': (0.3 if ecm == 240 else 1) if train else (0.1 if cat == 'qq' else 1), 'nb': big}

    category_specific: dict[str, dict[str, float | int]] = {
        'ee': {
            f'p8_ee_WW_ee_ecm{ecm}':           {'frac': 1, 'nb': middle},
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}': {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gaga_ee_60_ecm{ecm}':       {'frac': 1, 'nb': middle}},
        'mumu': {
            f'p8_ee_WW_mumu_ecm{ecm}':        {'frac': 1, 'nb': middle},
            f'wzp6_ee_mumu_ecm{ecm}':         {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}': {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zmumu_ecm{ecm}': {'frac': 1, 'nb': middle},
            f'wzp6_gaga_mumu_60_ecm{ecm}':    {'frac': 1, 'nb': middle}},
        'qq': {
            f'wzp6_ee_qq_ecm{ecm}':         {'frac': 1,   'nb': middle},
            f'wzp6_egamma_eZ_Zqq_ecm{ecm}': {'frac': 1,   'nb': middle},
            f'wzp6_gammae_eZ_Zqq_ecm{ecm}': {'frac': 1,   'nb': middle}},
    }
    if ecm == 365:
        category_specific['qq'].update({
            'wzp6_ee_WbWb_ecm365': {'frac': 1, 'nb': small},
            'p8_ee_tt_ecm365':     {'frac': 1, 'nb': small}})

    if train:
        bkgs = {**common, **category_specific.get(cat, {})}
    else:
        bkgs = {**common, **category_specific.get(cat, {}),
                f'wzp6_ee_tautau_ecm{ecm}':      {'frac': 1, 'nb': small},
                f'wzp6_gaga_tautau_60_ecm{ecm}': {'frac': 1, 'nb': small},
                f'wzp6_ee_nuenueZ_ecm{ecm}':     {'frac': 1, 'nb': small}}
        if cat == 'qq':
            bkgs.update({
                f'p8_ee_WW_ee_ecm{ecm}':           {'frac': 1,   'nb': middle},
                f'p8_ee_WW_mumu_ecm{ecm}':         {'frac': 1,   'nb': middle},
                f'wzp6_ee_ee_Mee_30_150_ecm{ecm}': {'frac': 0.1, 'nb': middle},
                f'wzp6_ee_mumu_ecm{ecm}':          {'frac': 0.1, 'nb': middle}})

    # Build signal dict with custom overrides
    process_sig = {
        s: {'fraction': frac.get(s, 1), 'chunks': chunks.get(s, 1)}
        for s in sigs if s not in exclude
    }

    # Build background dict with custom overrides
    process_bkg = {
        b: {'fraction': frac.get(b, v['frac']), 'chunks': chunks.get(b, v['nb'])}
        for b, v in bkgs.items() if b not in exclude
    }

    # Apply custom inclusions
    if 'sig' in include:
        process_sig = {**process_sig, **include['sig']}
    if 'bkg' in include:
        process_bkg = {**process_bkg, **include['bkg']}

    # Return requested subset
    if onlysig:
        return process_sig
    if onlybkg:
        return process_bkg
    return {**process_sig, **process_bkg}
