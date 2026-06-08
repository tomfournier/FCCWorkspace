'''Core configuration for the FCC-ee ZH cross-section analysis.

Provides:
- Feature set for BDT training: `input_vars`.
- Decay mode enumerations: `Z_DECAYS`, `H_DECAYS`, `H_DECAYS_WITH_INV`, `H_DECAYS_ALL`, `QUARKS`
    plus lowercase aliases for backward compatibility.
- Color palettes for ROOT and matplotlib: `colors`, `h_colors`, `modes_color`.
- Physics and axis labels (ROOT TLatex and LaTeX): `labels`, `h_labels`,
    `vars_label`, `vars_xlabel`, `modes_label`, `process_label`.
- Process builders:
    - `mk_processes()`: Simple process dictionary builder with optional filtering.
    - `get_process_list()`: Full-featured process builder with signal/background handling.
    - Helper functions: `_default_processes()`, `_build_processes()`, `_as_tuple()`.
- Background/signal builders for analysis workflows:
    - `_get_training_signals()`: Training-mode signal samples.
    - `_build_background_dict()`: Category and mode-specific background processes.
- Utilities: `warning()` for formatted exceptions and `timer()` for timing output.

Conventions:
- Process naming follows FCC patterns, e.g. ``wzp6_ee_{z}H_H{h}_ecm{ecm}``,
    ``p8_ee_WW_ecm{ecm}``.
- Labels use ROOT TLatex syntax for ROOT displays and LaTeX for matplotlib.
- Units are appended in `vars_xlabel` (e.g., GeV, GeV^2).

Usage:
- Simple process dictionary: ``mk_processes(procs=['ZH','WW'], ecm=365)``.
- Full analysis workflow: ``get_process_list(cat='mumu', ecm=240, train=True)``.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

# Lazy import of ROOT - only load when colors are accessed
_ROOT = None

def _get_root():
    """Lazily import ROOT when needed for color definitions.

    Defers ROOT import until colors are actually accessed to avoid unnecessary
    overhead if only simple features are used.

    Returns:
        The ROOT module.
    """
    global _ROOT
    if _ROOT is None:
        import ROOT
        _ROOT = ROOT
    return _ROOT


import sys

from time import time
from functools import lru_cache
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
    'W1_m', 'W1_p', 'W1_costheta',
    'W2_m', 'W2_p', 'W2_costheta',
    'thrust'
)



##########################
### Z AND HIGGS DECAYS ###
##########################

# Standard Z boson decay modes
Z_DECAYS: tuple[str, ...] = ('bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu')

# Standard Higgs boson decay modes
H_DECAYS: tuple[str, ...] = ('bb', 'cc', 'ss', 'gg', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa')

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
        _ZH_COLOR   = root.TColor.GetColor('#e42536')  # Red for ZH signal
        _WW_COLOR   = root.TColor.GetColor('#f89c20')  # Orange for WW background
        _ZZ_COLOR   = root.TColor.GetColor('#5790fc')  # Blue for ZZ background
        _ZG_COLOR   = root.TColor.GetColor('#964a8b')  # Purple for Z/gamma
        _RARE_COLOR = root.TColor.GetColor('#9c9ca1')  # Gray for rare processes
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

class LazyDict(dict):
    """Dictionary that defers initialization until first access.

    Avoids importing ROOT or loading color definitions until the dictionary
    is actually used. Supports standard dict operations.
    """
    def __init__(self, lazy_func):
        super().__init__()
        self._lazy_func = lazy_func
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            data = self._lazy_func()
            self.update(data)
            self._loaded = True

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()


# Lazy-loaded color dictionaries - ROOT is imported only on first access
# Maps decay modes and process names to ROOT color indices
h_colors = LazyDict(_get_h_colors_dict)  # Decay mode -> ROOT color
colors   = LazyDict(_get_colors_dict)    # Process name -> ROOT color

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
    'gaga_qq':     'tab:pink'
}



#######################
### PROCESSES LABEL ###
#######################

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
    'inv'    : r'$H\to\text{Inv}$'
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

# LaTeX labels for kinematic variables (used in matplotlib importance plots)
vars_label_ll = {
    'leading_p':        r'$p_{\ell,leading}$',
    'leading_pT':       r'$p_{T,leading}$',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'leading_phi':      r'$\phi_{\ell,leading}$',

    'subleading_p':     r'$p_{\ell,subleading}$',
    'subleading_pT':    r'$p_{T,subleading}$',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'subleading_phi':   r'$\phi_{\ell,subleading}$',

    'acolinearity':     r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'deltaR':           r'$\Delta R$',

    'zll_m':            r'$m_{\ell^{+}\ell^{-}}$',
    'zll_p':            r'$p_{\ell^{+}\ell^{-}}$',
    'zll_pT':           r'$p_{T, \ell^{+}\ell^{-}}$',
    'zll_theta':        r'$\theta_{\ell^{+}\ell^{+}}$',
    'zll_phi':          r'$\phi_{\ell^{+}\ell^{-}}$',

    'zll_recoil_m':     r'$m_{recoil}$',
    'cosTheta_miss':    r'$\cos\theta_{miss}$',

    'visibleEnergy':    r'$E_{vis}$',
    'missingMass':      r'$m_{miss}$',

    'H':                r'$H$',
    'BDTscore':         r'BDT Score'
}

vars_label_qq = {
    'leading_p':             r'$p_{jet,leading}$',
    'leading_pT':            r'$p_{T,leading}$',
    'leading_theta':         r'$\theta_{jet,leading}$',
    'leading_costheta':      r'$\cos\theta_{jet,leading}$',
    'leading_phi':           r'$\phi_{jet,leading}$',

    'subleading_p':          r'$p_{jet,subleading}$',
    'subleading_pT':         r'$p_{T,subleading}$',
    'subleading_theta':      r'$\theta_{jet,subleading}$',
    'subleading_costheta':   r'$\cos\theta_{jet,subleading}$',
    'subleading_phi':        r'$\phi_{jet,subleading}$',

    'acolinearity':          r'$\Delta\theta_{jj}$',
    'acoplanarity':          r'$\pi - \Delta\phi_{jj}$',
    'deltaR':                r'$\Delta R$',

    'zqq_m':                 r'$m_{jj}$',
    'zqq_p':                 r'$p_{jj}$',
    'zqq_pT':                r'$p_{T,jj}$',
    'zqq_theta':             r'$\theta_{jj}$',
    'zqq_costheta':          r'$\cos\theta_{jj}$',
    'zqq_phi':               r'$\phi_{jj}$',

    'W1_m':                  r'$m_{W1}$',
    'W1_p':                  r'$p_{W1}$',
    'W1_costheta':           r'$\cos\theta_{W1}$',

    'W2_m':                  r'$m_{W2}$',
    'W2_p':                  r'$p_{W2}$',
    'W2_costheta':           r'$\cos\theta_{W2}$',

    'thrust':                r'$T$',
    'thrust_costheta':       r'$\cos\theta_{T}$',

    'zqq_recoil_m':          r'$m_{recoil}$',
    'cosTheta_miss':         r'$\cos\theta_{miss}$',

    'visibleEnergy':         r'$E_{vis}$',
    'missingMass':           r'$m_{miss}$',

    'BDTscore':              r'BDT Score'
}



# LaTeX x-axis labels with units (used in histogram plots)
vars_xlabel_ll = vars_label_ll.copy()
for v in ['leading_p', 'leading_pT', 'subleading_p', 'subleading_pT',
          'zll_m', 'zll_p', 'zll_recoil_m', 'visibleEnergy', 'missingMass']:
    vars_xlabel_ll[v] += ' [GeV]'
vars_xlabel_ll['H'] += ' [GeV$^{2}$]'

vars_xlabel_qq = vars_label_qq.copy()
for v in ['leading_p', 'leading_pT', 'subleading_p', 'subleading_pT',
          'zqq_m', 'zqq_p', 'zqq_recoil_m', 'visibleEnergy', 'missingMass']:
    vars_xlabel_qq[v] += ' [GeV]'

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
    'gaga_qq':     r'$\gamma\gamma\rightarrow q\bar{q}$'
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

# _____________________________
def warning(log_msg: str,
            lenght: int = -1,
            abort_msg: str = ''
            ) -> None:
    '''Log formatted error message and exit.

    Displays an error message in a centered box and terminates execution.
    Message box width is auto-calculated if not specified.

    Args:
        log_msg: Error message to display.
        lenght: Box width. Auto-calculated if -1 (default).
        abort_msg: Header text for error box (default: ' ERROR CODE ').
    '''
    if not abort_msg:
        abort_msg = ' ERROR CODE '
    # Auto-calculate box width if not specified
    if lenght==-1:
        if len(log_msg) < len(abort_msg) + 6:
            lenght = len(abort_msg) + 6
        else:
            lenght = len(log_msg) + 6

    # Format and raise exception
    msg  = f'\n{abort_msg:=^{lenght}}\n'
    msg += f'{log_msg:^{lenght}}\n'
    sep  = '=' * lenght
    msg += f'{sep:^{lenght}}\n'

    LOGGER.error(msg)

    sys.exit(1)

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

def _as_tuple(seq: Union[Sequence[str], None],
              fallback: tuple[str, ...]
              ) -> tuple[str, ...]:
    '''Convert sequence to tuple or use fallback if None.

    Helper for flexible parameter handling in process builders.

    Args:
        seq: Input sequence to convert, or None.
        fallback: Default tuple if seq is None.

    Returns:
        Tuple from seq if provided, otherwise fallback tuple.
    '''
    return tuple(seq) if seq is not None else fallback

def _build_processes(z_set: tuple[str, ...],
                     h_set: tuple[str, ...],
                     H_set: tuple[str, ...],
                     q_set: tuple[str, ...],
                     ecm: int) -> dict[str,
                                       tuple[str, ...]]:
    '''Build process dictionary from decay modes and center-of-mass energy.

    Generates full FCC process names by combining decay modes with energy.
    Uppercase keys (ZH, ZeeH, etc.) use h_set (no invisible).
    Lowercase keys (zh, zeeh, etc.) use H_set (with invisible).

    Args:
        z_set: Z boson decay modes (bb, cc, ss, qq, ee, mumu, tautau, nunu).
        h_set: Higgs decay modes without invisible (bb, cc, ss, gg, mumu, etc.).
        H_set: Higgs decay modes with invisible (includes 'inv').
        q_set: Quark channels (bb, cc, ss, qq).
        ecm: Center-of-mass energy in GeV (240 or 365).

    Returns:
        Dictionary mapping process keys to tuples of FCC sample names.
    '''
    processes =  {
        'ZH':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set for y in h_set),
        'ZeeH':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_set),
        'ZmumuH': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_set),
        'ZqqH':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set for y in h_set),

        # Lowercase variants include H -> Inv decay
        'zh':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set for y in H_set),
        'zeeh':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in H_set),
        'zmumuh': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in H_set),
        'zqqh':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set for y in H_set),

        'WW': (
            f'p8_ee_WW_ee_ecm{ecm}',
            f'p8_ee_WW_mumu_ecm{ecm}',
            f'p8_ee_WW_ecm{ecm}',
        ),
        'ZZ': (f'p8_ee_ZZ_ecm{ecm}',),
        'Zgamma': (
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
            f'wzp6_ee_mumu_ecm{ecm}',
            f'wzp6_ee_tautau_ecm{ecm}',
            f'wzp6_ee_qq_ecm{ecm}',
        ),
        'Rare': (
            f'wzp6_gammae_eZ_Zee_ecm{ecm}',
            f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
            f'wzp6_gammae_eZ_Zqq_ecm{ecm}',
            f'wzp6_egamma_eZ_Zee_ecm{ecm}',
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
            f'wzp6_egamma_eZ_Zqq_ecm{ecm}',
            f'wzp6_gaga_ee_60_ecm{ecm}',
            f'wzp6_gaga_mumu_60_ecm{ecm}',
            f'wzp6_gaga_tautau_60_ecm{ecm}',
            f'wzp6_ee_nuenueZ_ecm{ecm}',
        ),
    }
    if ecm == 365:
        processes['tt'] = ('p8_ee_tt_ecm365',)
    return processes

@lru_cache(maxsize=None)
def _default_processes(ecm: int
                       ) -> dict[str,
                                 tuple[str, ...]]:
    '''Generate process dictionary with standard decay modes (cached).

    Uses default Z_DECAYS, H_DECAYS, H_DECAYS_WITH_INV, and QUARKS.
    Results are cached by functools.lru_cache for performance.

    Args:
        ecm: Center-of-mass energy in GeV.

    Returns:
        Process dictionary with default decay modes.
    '''
    return _build_processes(Z_DECAYS, H_DECAYS, H_DECAYS_WITH_INV, QUARKS, ecm)

# __________________________________________________
def mk_processes(
        procs:    Union[Sequence[str], None] = None,
        z_decays: Union[Sequence[str], None] = None,
        h_decays: Union[Sequence[str], None] = None,
        H_decays: Union[Sequence[str], None] = None,
        quarks:   Union[Sequence[str], None] = None,
        ecm: int = 240
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
        >>> mk_processes()  # All processes, default decays, 240 GeV
        >>> mk_processes(procs=['ZH', 'WW'], ecm=365)  # Filtered, 365 GeV
        >>> mk_processes(h_decays=['bb', 'cc'])  # Custom Higgs decays
    '''
    # Use cached defaults if all decay parameters are None
    use_defaults = all(val is None for val in (z_decays, h_decays, H_decays, quarks))
    if use_defaults:
        processes = _default_processes(ecm)  # Cached for performance
    else:
        processes = _build_processes(
            _as_tuple(z_decays, Z_DECAYS),
            _as_tuple(h_decays, H_DECAYS),
            _as_tuple(H_decays, H_DECAYS_WITH_INV),
            _as_tuple(quarks, QUARKS),
            ecm
        )

    # Filter to requested process keys if specified
    if procs:
        requested = tuple(procs)
        return {proc: processes[proc] for proc in requested if proc in processes}
    return processes





def _get_training_signals(cat: str, ecm: int) -> list[str]:
    '''Build training signal samples for specified category.

    Returns category-specific signal processes for training mode.
    For leptonic categories (ee, mumu), returns single mode samples.
    For hadronic (qq), returns all quark flavors at 365 GeV and qqH at 240 GeV.

    Args:
        cat: Category ('ee', 'mumu', 'qq').
        ecm: Center-of-mass energy in GeV (240 or 365).

    Returns:
        List of signal sample names for training.

    Raises:
        ValueError: If cat is not in ['ee', 'mumu', 'qq'] or ecm is unsupported.
    '''
    if cat in ['ee', 'mumu']:
        return [f'wzp6_ee_{cat}H_ecm{ecm}']
    if cat == 'qq':
        if ecm == 240:
            return [f'wzp6_ee_qqH_ecm{ecm}']
        elif ecm == 365:
            return [f'wzp6_ee_{x}H_ecm{ecm}' for x in ['bb', 'cc', 'ss', 'qq']]
        else:
            raise ValueError(f'{ecm} is not supported for training. Use 240 or 365.')
    raise ValueError(f'{cat} is not a valid category. Use [ee, mumu, qq].')

def _build_background_dict(cat: str, ecm: int, train: bool, batch: bool = False) -> dict[str, dict]:
    '''Build category and mode-specific background process dictionary.

    Constructs background processes with appropriate event chunk counts.
    Training mode uses reduced backgrounds (category-specific only).
    Non-training mode includes all lepton-pair and rare processes.
    Chunk sizes scale with batch mode: larger chunks for batch processing.

    Args:
        cat: Category ('ee', 'mumu', 'qq').
        ecm: Center-of-mass energy in GeV.
        train: If True, use training-specific backgrounds (smaller sample).
        batch: If True, use larger chunk sizes for batch processing.

    Returns:
        Dictionary mapping process names to {'frac': fraction, 'nb': chunks}.

    Raises:
        ValueError: If cat is unsupported.
    '''

    if train:
        small  = 5  if batch else 1
        middle = 5  if batch else 5
        big    = 10 if batch else 10
    else:
        small  = 5  if batch else 1
        middle = 20 if batch else 5
        big    = 30 if batch else 10

    # Common diboson processes
    common: dict[str, dict[str, int]] = {}
    common[f'p8_ee_ZZ_ecm{ecm}'] = {'frac': 0.5 if cat=='qq' else 1, 'nb': middle}
    if not train or (cat == 'qq'):
        common[f'p8_ee_WW_ecm{ecm}'] = {'frac': 0.1 if cat=='qq' else 1, 'nb': big}

    category_specific = {
        'ee': {
            f'p8_ee_WW_ee_ecm{ecm}':           {'frac': 1, 'nb': middle},
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}': {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zee_ecm{ecm}':    {'frac': 1, 'nb': middle},
            f'wzp6_gaga_ee_60_ecm{ecm}':       {'frac': 1, 'nb': middle},
        },
        'mumu': {
            f'p8_ee_WW_mumu_ecm{ecm}':         {'frac': 1, 'nb': middle},
            f'wzp6_ee_mumu_ecm{ecm}':          {'frac': 1, 'nb': big},
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}':  {'frac': 1, 'nb': middle},
            f'wzp6_gammae_eZ_Zmumu_ecm{ecm}':  {'frac': 1, 'nb': middle},
            f'wzp6_gaga_mumu_60_ecm{ecm}':     {'frac': 1, 'nb': middle},
        },
        'qq': {
            f'wzp6_ee_qq_ecm{ecm}':            {'frac': 0.5, 'nb': big},
            f'wzp6_egamma_eZ_Zqq_ecm{ecm}':    {'frac': 1,   'nb': middle},
            f'wzp6_gammae_eZ_Zqq_ecm{ecm}':    {'frac': 1,   'nb': middle},
            # f'wzp6_gaga_qq_60_ecm{ecm}':       {'frac': 1, 'nb': middle},
        },
    }

    # Training mode: category-specific backgrounds (reduced sample size)
    if train:
        return {**common, **category_specific.get(cat, {})}

    # Non-training mode: comprehensive backgrounds (full sample)
    # Lepton-pair backgrounds (ee, mumu, tautau channels)
    nominal_bkgs = {
        f'wzp6_ee_tautau_ecm{ecm}':         {'frac': 1, 'nb': small},
        f'wzp6_gaga_tautau_60_ecm{ecm}':    {'frac': 1, 'nb': small},
        f'wzp6_ee_nuenueZ_ecm{ecm}':        {'frac': 1, 'nb': small},
    }

    bkgs = {**common, **category_specific.get(cat, {}), **nominal_bkgs}
    if cat in ['ee', 'mumu']:
        return bkgs
    elif cat == 'qq':
        bkgs[f'p8_ee_WW_ee_ecm{ecm}']   = {'frac': 1, 'nb': middle}
        bkgs[f'p8_ee_WW_mumu_ecm{ecm}'] = {'frac': 1, 'nb': middle}
        # Special case: top production at 365 GeV
        if ecm == 365:
            bkgs['p8_ee_tt_ecm365'] = {'frac': 1, 'nb': small}
        return bkgs

    return common


def get_process_list(
        cat: str,
        ecm: int,
        z_decays: tuple[str, ...] = Z_DECAYS,
        h_decays: tuple[str, ...] = H_DECAYS_ALL,
        train: bool = False,
        batch: bool = False,
        onlysig: bool = False,
        onlybkg: bool = False,
        frac: dict[str, float] | None = None,
        chunks: dict[str, int] | None = None,
        include: dict[str, dict] | None = None,
        exclude: set[str] | None = None,
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

    # Generate signal samples
    if train:
        # Training mode: category-specific signal only
        sigs = _get_training_signals(cat, ecm)
    else:
        # Non-training mode: all Z and Higgs decay combinations
        sigs = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays]

    # Generate background samples
    bkgs = _build_background_dict(cat, ecm, train, batch)

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
