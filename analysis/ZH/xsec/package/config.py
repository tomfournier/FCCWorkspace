'''Core configuration for the FCC-ee ZH cross-section analysis.

Provides:
- Feature set for BDT training: `input_vars`.
- Decay mode enumerations: `Z_DECAYS`, `H_DECAYS`, `H_DECAYS_WITH_INV`, `QUARKS`
    plus lowercase aliases for backward compatibility.
- Color palettes for ROOT and matplotlib: `colors`, `h_colors`, `modes_color`.
- Physics and axis labels (ROOT TLatex and LaTeX): `labels`, `h_labels`,
    `vars_label`, `vars_xlabel`, `modes_label`.
- Process builders for sample names: `mk_processes()` with cached defaults
    via `_default_processes()` and helpers `_build_processes()` and `_as_tuple()`.
- Small utilities: `warning()` for formatted exceptions and `timer()` for
    pretty end-of-code timing output.

Conventions:
- Process naming follows FCC patterns, e.g. ``wzp6_ee_{z}H_H{h}_ecm{ecm}``,
    ``p8_ee_WW_ecm{ecm}``.
- Labels use ROOT TLatex syntax where applicable.
- Units are appended in `vars_xlabel` (e.g., GeV, GeV^2).

Usage:
- Construct process maps for an energy with optional filtering:
    ``mk_processes(procs=['ZH','WW'], ecm=365)``.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

# Lazy import of ROOT - only load when colors are accessed
_ROOT = None

def _get_root():
    """Lazily import ROOT only when needed for color definitions."""
    global _ROOT
    if _ROOT is None:
        import ROOT
        _ROOT = ROOT
    return _ROOT

from time import time
from functools import lru_cache
from typing import Sequence, Union



#############################
##### VARIABLES FOR BDT #####
#############################

# Tuple of kinematic variables used as input features for BDT training
input_vars = (
    'leading_p',    'leading_theta', 
    'subleading_p', 'subleading_theta',
    'acolinearity', 'acoplanarity', 
    'zll_m', 'zll_p', 'zll_theta'
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
# ROOT object cache (lazily loaded if use_root=True)
_ZH_COLOR   = None
_WW_COLOR   = None
_ZZ_COLOR   = None
_ZG_COLOR   = None
_RARE_COLOR = None

def _init_colors(use_root: bool = False):
    """Initialize ROOT colors on first access.
    
    Args:
        use_root: If True, use ROOT.TColor.GetColor(); if False, use numeric constants.
    """
    global _ZH_COLOR, _WW_COLOR, _ZZ_COLOR, _ZG_COLOR, _RARE_COLOR
    if _ZH_COLOR is None:
        root = _get_root()
        _ZH_COLOR   = root.TColor.GetColor('#e42536')  # Red for ZH signal
        _WW_COLOR   = root.TColor.GetColor('#f89c20')  # Orange for WW background
        _ZZ_COLOR   = root.TColor.GetColor('#5790fc')  # Blue for ZZ background
        _ZG_COLOR   = root.TColor.GetColor('#964a8b')  # Purple for Z/gamma
        _RARE_COLOR = root.TColor.GetColor('#9c9ca1')  # Gray for rare processes

def _get_h_colors_dict(use_root: bool = False) -> dict:
    """Lazy-load h_colors with color constants.
    
    Args:
        use_root: If True, use ROOT color constants; if False, use numeric codes.
        
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

def _get_colors_dict(use_root: bool = False) -> dict:
    """Lazy-load colors dictionary with color constants.
    
    Args:
        use_root: If True, use ROOT.TColor.GetColor(); if False, use numeric codes.
        
    Returns:
        Dictionary mapping process names to color codes.
    """
    _init_colors(use_root=use_root)
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
        'Rare'     : _RARE_COLOR
    }

class LazyDict(dict):
    """Dictionary that loads from a lazy function on first access."""
    def __init__(self, lazy_func, use_root: bool = False):
        super().__init__()
        self._lazy_func = lazy_func
        self._use_root = use_root
        self._loaded = False
    
    def _ensure_loaded(self):
        if not self._loaded:
            data = self._lazy_func(use_root=self._use_root)
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

# Lazy-loaded color dictionaries - by default uses numeric codes (no ROOT import)
# Set use_root=True to use ROOT color constants instead
h_colors = LazyDict(_get_h_colors_dict, use_root=False)
colors   = LazyDict(_get_colors_dict, use_root=False)

# Matplotlib tab colors for different analysis modes by channel (no lazy loading needed)
modes_color = {
    f'ZmumuH':      'tab:blue',
    f'ZZ':          'tab:orange',
    f'Zmumu':       'tab:red',
    f'WWmumu':      'tab:green',
    f'egamma_mumu': 'tab:purple',
    f'gammae_mumu': 'tab:brown',
    f'gaga_mumu':   'tab:pink',
    
    f'ZeeH':        'tab:blue',
    f'Zee':         'tab:red',
    f'WWee':        'tab:green',
    f'egamma_ee':   'tab:purple',
    f'gammae_ee':   'tab:brown',
    f'gaga_ee':     'tab:pink'
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
    'Rare'   : 'Rare'
}

# LaTeX labels for kinematic variables (used in matplotlib importance plots)
vars_label = {
    'leading_p':        r'$p_{\ell,leading}$',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'leading_phi':      r'$\phi_{\ell, leading}$',

    'subleading_p':     r'$p_{\ell,subleading}$',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'subleading_phi':   r'$\phi_{\ell, subleading}$',
    
    'acolinearity':     r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'deltaR':           r'$\Delta R$',

    'zll_m':            r'$m_{\ell^{+}\ell^{-}}$',
    'zll_p':            r'$p_{\ell^{+}\ell^{-}}$',
    'zll_theta':        r'$\theta_{\ell^{+}\ell^{+}}$',
    'zll_phi':          r'$\phi_{\ell^{+}\ell^{-}}$',

    'zll_recoil_m':     r'$m_{recoil}$',
    'cosTheta_miss':    r'$\cos\theta_{miss}$',

    'visibleEnergy':    r'$E_{vis}$',
    'missingMass':      r'$m_{miss}$',
    
    'H':                r'$H$',
    'BDTscore':         r'BDT Score'
}

# LaTeX x-axis labels with units (used in histogram plots)
vars_xlabel = vars_label.copy()
for v in ['leading_p', 'subleading_p', 'zll_m', 'zll_p', 'zll_recoil_m', 'visibleEnergy', 'missingMass']:
    vars_xlabel[v] += ' [GeV]'
vars_xlabel['H'] += ' [GeV$^{2}$]'

# LaTeX labels for analysis modes (physics processes)
modes_label = {
    f'ZmumuH':      r'$e^+e^-\rightarrow Z(\mu^+\mu^-)H$',
    f'ZZ':          r'$e^+e^-\rightarrow ZZ$', 
    f'Zmumu':       r'$e^+e^-\rightarrow Z/\gamma^{*}\rightarrow\mu^+\mu^-$',
    f'WWmumu':      r'$e^+e^-\rightarrow W^{+}W^{-}[\nu_{\mu}\mu]$',
    f'egamma_mumu': r'$e^-\gamma\rightarrow e^-Z(\mu^+\mu^-)$',
    f'gammae_mumu': r'$e^+\gamma\rightarrow e^+Z(\mu^+\mu^-)$',
    f'gaga_mumu':   r'$\gamma\gamma\rightarrow\mu^+\mu^-$',
    
    f'ZeeH':        r'$e^+e^-\rightarrow Z(e^+e^-)H$',
    f'Zee':         r'$e^+e^-\rightarrow Z/\gamma^{*}\rightarrow e^+e^-$',
    f'WWee':        r'$e^+e^-\rightarrow W^{+}W^{-}[\nu_{e}e]$',
    f'egamma_ee':   r'$e^-\gamma\rightarrow e^-Z(e^+e^-)$',
    f'gammae_ee':   r'$e^+\gamma\rightarrow e^+Z(e^+e^-)$',
    f'gaga_ee':     r'$\gamma\gamma\rightarrow e^+e^-$'
}



########################
### CONFIG FUNCTIONS ###
########################

#______________________________
def warning(log_msg: str, 
            lenght: int = -1, 
            abort_msg: str = ''
            ) -> None:
    '''Print formatted error message and raise exception.
    
    Args:
        log_msg: Error message to display.
        lenght: Width of the message box. Auto-calculated if -1.
        abort_msg: Header text for the error box. Defaults to ' ERROR CODE '.
        
    Raises:
        Exception: Always raised with formatted error message.
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
    msg =  f'\n{abort_msg:=^{lenght}}\n'
    msg += f'{log_msg:^{lenght}}\n'
    sep = '=' * lenght
    msg += f'{sep:^{lenght}}\n'
    print(msg)

#___________________
def timer(t: float
          ) -> None:
    '''Print formatted elapsed time since timestamp.
    
    Calculates and displays elapsed time in human-readable format (hours, minutes,
    seconds, milliseconds) since the provided timestamp.
    
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

    print(f'\n{" CODE ENDED ":=^{lenght}}\n{elapsed:^{lenght}}\n{"="*lenght}\n')



##########################
### PROCESSES FUNCTION ###
##########################

def _as_tuple(seq: Union[Sequence[str], None], 
              fallback: tuple[str, ...]
              ) -> tuple[str, ...]:
    '''Convert sequence to tuple or return fallback if None.
    
    Args:
        seq: Input sequence to convert, or None.
        fallback: Default tuple to use if seq is None.
        
    Returns:
        Tuple from seq if provided, otherwise fallback.
    '''
    return tuple(seq) if seq is not None else fallback

def _build_processes(z_set: tuple[str, ...],
                     h_set: tuple[str, ...],
                     H_set: tuple[str, ...],
                     q_set: tuple[str, ...],
                     ecm: int) -> dict[str, 
                                       tuple[str, ...]]:
    '''Build process name dictionary from decay modes and center-of-mass energy.
    
    Constructs full process names following the FCC naming convention by combining
    Z decays, Higgs decays, and center-of-mass energy.
    
    Args:
        z_set: Z boson decay modes.
        h_set: Higgs decay modes (without invisible).
        H_set: Higgs decay modes (with invisible).
        q_set: Quark decay channels.
        ecm: Center-of-mass energy in GeV.
        
    Returns:
        Dictionary mapping process keys to tuples of full process names.
    '''
    return {
        'ZH':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set 
                                                       for y in h_set),
        'ZeeH':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_set),
        'ZmumuH': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_set),
        'ZqqH':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set 
                                                       for y in h_set),

        # Lowercase variants include H -> Inv decay
        'zh':     tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_set 
                                                       for y in H_set),
        'zeeh':   tuple(f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in H_set),
        'zmumuh': tuple(f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in H_set),
        'zqqh':   tuple(f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in q_set 
                                                       for y in H_set),

        'WW': (
            f'p8_ee_WW_ecm{ecm}',
            f'p8_ee_WW_mumu_ecm{ecm}',
            f'p8_ee_WW_ee_ecm{ecm}',
        ),
        'ZZ': (f'p8_ee_ZZ_ecm{ecm}',),
        'Zgamma': (
            f'wzp6_ee_tautau_ecm{ecm}',
            f'wzp6_ee_mumu_ecm{ecm}',
            f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
        ),
        'Rare': (
            f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',
            f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
            f'wzp6_gammae_eZ_Zee_ecm{ecm}',
            f'wzp6_egamma_eZ_Zee_ecm{ecm}',
            f'wzp6_gaga_ee_60_ecm{ecm}',
            f'wzp6_gaga_mumu_60_ecm{ecm}',
            f'wzp6_gaga_tautau_60_ecm{ecm}',
            f'wzp6_ee_nuenueZ_ecm{ecm}',
        ),
    }

@lru_cache(maxsize=None)
def _default_processes(ecm: int) -> dict[str, 
                                         tuple[str, ...]]:
    '''Generate default process dictionary with standard decay modes (cached).
    
    Uses default Z_DECAYS, H_DECAYS, H_DECAYS_WITH_INV, and QUARKS.
    Results are cached for performance.
    
    Args:
        ecm: Center-of-mass energy in GeV.
        
    Returns:
        Dictionary of process names with default decay modes.
    '''
    return _build_processes(Z_DECAYS, H_DECAYS, H_DECAYS_WITH_INV, QUARKS, ecm)

#____________________________________________________________
def mk_processes(procs:    Union[Sequence[str], None] = None,
                 z_decays: Union[Sequence[str], None] = None, 
                 h_decays: Union[Sequence[str], None] = None, 
                 H_decays: Union[Sequence[str], None] = None, 
                 quarks:   Union[Sequence[str], None] = None,
                 ecm: int = 240
                 ) -> dict[str, 
                           tuple[str, ...]]:
    '''Generate process dictionary with optional filtering and custom decay modes.
    
    Creates a dictionary mapping process keys to full process names. Can use default
    decay modes or custom ones. Optionally filters to specific processes.
    
    Args:
        procs: Process keys to include. If None, returns all processes.
        z_decays: Z decay modes. Uses Z_DECAYS if None.
        h_decays: Higgs decay modes (without invisible). Uses H_DECAYS if None.
        H_decays: Higgs decay modes (with invisible). Uses H_DECAYS_WITH_INV if None.
        quarks: Quark channels. Uses QUARKS if None.
        ecm: Center-of-mass energy in GeV. Default is 240.
        
    Returns:
        Dictionary mapping process keys to tuples of full process names.
        
    Examples:
        >>> mk_processes()  # All processes with default decays
        >>> mk_processes(procs=['ZH', 'WW'], ecm=365)  # Specific processes
        >>> mk_processes(h_decays=['bb', 'cc'])  # Custom Higgs decays
    '''
    # Use cached defaults if all decay mode parameters are None
    use_defaults = all(val is None for val in (z_decays, h_decays, H_decays, quarks))
    processes = _default_processes(ecm) if use_defaults else _build_processes(
        _as_tuple(z_decays, Z_DECAYS),
        _as_tuple(h_decays, H_DECAYS),
        _as_tuple(H_decays, H_DECAYS_WITH_INV),
        _as_tuple(quarks, QUARKS),
        ecm
    )
    # Filter to requested processes if specified
    if procs:
        requested = tuple(procs)
        return {proc: processes[proc] for proc in requested if proc in processes}
    return processes
            