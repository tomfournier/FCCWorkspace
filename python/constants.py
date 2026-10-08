'''Core configuration for the FCC-ee ZH cross-section analysis.

Provides:
- Decay mode enumerations: `Z_DECAYS`, `H_DECAYS`, `H_DECAYS_WITH_INV`, `H_DECAYS_ALL`, `QUARKS`
    plus lowercase aliases for backward compatibility.
- Color palettes for ROOT and matplotlib: `colors`, `h_colors`, `modes_color`.
- Physics and axis labels (ROOT TLatex and LaTeX): `labels`, `h_labels`,
    `vars_label`, `modes_label`, `process_label`.

Conventions:
- Labels use ROOT TLatex syntax for ROOT displays and LaTeX for matplotlib.
'''

####################################
### IMPORT MODULES AND FUNCTIONS ###
####################################

from logger import get_logger
from utilities import LazyColorDict, get_root
LOGGER = get_logger(__name__)



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
        root = get_root()
        _ZH_COLOR   = root.TColor.GetColor('#e42536')    # Red       for ZH signal
        _WW_COLOR   = root.TColor.GetColor('#f89c20')    # Orange    for WW background
        _ZZ_COLOR   = root.TColor.GetColor('#5790fc')    # Blue      for ZZ background
        _ZG_COLOR   = root.TColor.GetColor('#964a8b')    # Purple    for Z/gamma
        _RARE_COLOR = root.TColor.GetColor('#9c9ca1')    # Gray      for rare processes
        _TT_COLOR   = root.TColor.GetColor("#1414ad")    # Dark blue for tt processes


def _get_h_colors_dict() -> dict:
    """Lazy-load h_colors with color constants.

    Returns:
        Dictionary mapping decay modes to color codes.
    """

    root = get_root()
    return {'bb':     root.kViolet,
            'cc':     root.kBlue,
            'ss':     root.kRed,
            'gg':     root.kGreen+1,
            'mumu':   root.kOrange,
            'tautau': root.kCyan,
            'ZZ':     root.kGray,
            'WW':     root.kGray+2,
            'Za':     root.kGreen+2,
            'aa':     root.kRed+2,
            'inv':    root.kBlue+2}


def _get_colors_dict() -> dict:
    """Lazy-load colors dictionary with color constants.

    Returns:
        Dictionary mapping process names to color codes.
    """
    _init_colors()
    return {'ZH':     _ZH_COLOR,
            'ZeeH':   _ZH_COLOR,
            'ZmumuH': _ZH_COLOR,
            'ZqqH':   _ZH_COLOR,
            'ZnunuH': _ZH_COLOR,

            'zh':     _ZH_COLOR,
            'zeeh':   _ZH_COLOR,
            'zmumuh': _ZH_COLOR,
            'zqqh':   _ZH_COLOR,
            'znunuh': _ZH_COLOR,

            'WW':       _WW_COLOR,
            'ZZ':       _ZZ_COLOR,
            'Zgamma':   _ZG_COLOR,
            'Zqqgamma': _ZG_COLOR,
            'Rare':     _RARE_COLOR,
            'tt':       _TT_COLOR}


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
    'bb':     'Z#rightarrowb#bar{b}',
    'cc':     'Z#rightarrowc#bar{c}',
    'ss':     'Z#rightarrows#bar{s}',
    'qq':     'Z#rightarrowq#bar{q}',
    'ee':     'Z#rightarrowe^{#plus}e^{#minus}',
    'mumu':   'Z#rightarrow#mu^{#plus}#mu^{#minus}',
    'tautau': 'Z#rightarrow#tau^{#plus}#tau^{#minus}',
    'nunu':   'Z#rightarrow#nu#bar{#nu}',
}

# ROOT TLatex labels for Higgs decay modes
h_labels = {
    'bb':     'H#rightarrowb#bar{b}',
    'cc':     'H#rightarrowc#bar{c}',
    'ss':     'H#rightarrows#bar{s}',
    'gg':     'H#rightarrowgg',
    'mumu':   'H#rightarrow#mu^{#plus}#mu^{#minus}',
    'tautau': 'H#rightarrow#tau^{#plus}#tau^{#minus}',
    'ZZ':     'H#rightarrowZZ*',
    'WW':     'H#rightarrowWW*',
    'Za':     'H#rightarrowZ#gamma',
    'aa':     'H#rightarrow#gamma#gamma',
    'inv':    'H#rightarrowInv'
}

H_labels = {
    'bb':     r'$H\to b\bar{b}$',
    'cc':     r'$H\to c\bar{c}$',
    'ss':     r'$H\to s\bar{s}$',
    'gg':     r'$H\to gg$',
    'mumu':   r'$H\to \mu^+\mu^-$',
    'tautau': r'$H\to \tau^+\tau^-$',
    'ZZ':     r'$H\to ZZ^*$',
    'WW':     r'$H\to WW^*$',
    'Za':     r'$H\to Z\gamma$',
    'aa':     r'$H\to \gamma\gamma$',
    'inv':    r'$H\to$ Inv'
}

# ROOT TLatex labels for main physics processes
legend = {
    'ZH':     'ZH',
    'ZmumuH': 'Z(#mu^{+}#mu^{#minus})H',
    'ZeeH':   'Z(e^{+}e^{#minus})H',
    'ZqqH':   'Z(q#bar{q})H',

    'zh':     'ZH',
    'zmumuh': 'Z(#mu^{+}#mu^{#minus})H',
    'zeeh':   'Z(e^{+}e^{#minus})H',
    'zqqh':   'Z(q#bar{q})H',

    'WW':     'W^{+}W^{-}',
    'ZZ':     'ZZ',
    'Zgamma': 'Z/#gamma^{*} #rightarrow f#bar{f}+#gamma(#gamma)',
    'Rare':   'Rare',
    'tt':     't#bar{t}'
}

# Labels shared by the leptonic and hadronic variable sets.
_common_var_labels = {
    'leading_p':             r'$p_{leading}$ [GeV]',
    'leading_pT':            r'$p_{T,leading}$ [GeV]',
    'leading_theta':         r'$\theta_{leading}$',
    'leading_costheta':      r'$\cos\theta_{leading}$',

    'subleading_p':          r'$p_{subleading}$ [GeV]',
    'subleading_pT':         r'$p_{T,subleading}$ [GeV]',
    'subleading_theta':      r'$\theta_{subleading}$',
    'subleading_costheta':   r'$\cos\theta_{subleading}$',

    'deltaR':                r'$\Delta R$',
    'cosTheta_miss':         r'$\cos\theta_{miss}$',
    'visibleEnergy':         r'$E_{vis}$ [GeV]',
    'missingEnergy':         r'$E_{miss}$ [GeV]',
    'missingMass':           r'$m_{miss}$ [GeV]',
    'BDTscore':              r'BDT Score',
    'H':                     r'$H$ [GeV]',
}

# Leptonic kinematic labels.
vars_label_ll = {
    **_common_var_labels,

    'acolinearity':     r'$\pi - \Delta\alpha_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'acopolarity':      r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'zll_m':            r'$m_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_e':            r'$E_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_p':            r'$p_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_pT':           r'$p_{T, \ell^{+}\ell^{-}}$ [GeV]',
    'zll_theta':        r'$\theta_{\ell^{+}\ell^{+}}$',
    'zll_costheta':     r'$\cos\theta_{\ell^{+}\ell^{+}}$',
    'zll_phi':          r'$\phi_{\ell^{+}\ell^{-}}$',

    'zll_recoil_m':     r'$m_{recoil}$ [GeV]',
    'zll_recoil_m_tot': r'$m_{recoil}$ [GeV]',
    'leps_iso':         r'$I_{rel}$',
    'leps_iso_no':      r'Isolated leptons'
}

# Hadronic kinematic labels.
vars_label_qq = {
    **_common_var_labels,
    'acolinearity':          r'$\pi - \Delta\alpha_{jj}$',
    'acoplanarity':          r'$\pi - \Delta\phi_{jj}$',
    'acopolarity':           r'$\Delta\theta_{jj}$',
    'zqq_m':                 r'$m_{jj}$ [GeV]',
    'zqq_e':                 r'$E_{jj}$ [GeV]',
    'zqq_p':                 r'$p_{jj}$ [GeV]',
    'zqq_pT':                r'$p_{T,jj}$ [GeV]',
    'zqq_theta':             r'$\theta_{jj}$',
    'zqq_costheta':          r'$\cos\theta_{jj}$',
    'zqq_phi':               r'$\phi_{jj}$',

    'W1_m':                  r'$m_{W1}$ [GeV]',
    'W1_p':                  r'$p_{W1}$ [GeV]',
    'W1_theta':              r'$\theta_{W1}$',
    'W1_costheta':           r'$\cos\theta_{W1}$',

    'W2_m':                  r'$m_{W2}$ [GeV]',
    'W2_p':                  r'$p_{W2}$ [GeV]',
    'W2_theta':              r'$\theta_{W2}$',
    'W2_costheta':           r'$\cos\theta_{W2}$',

    'delta_mWW4':            r'$\Delta m_{WW}$ (4 jets algo) [GeV]',

    'thrust':                r'$T$',
    'thrust_costheta':       r'$\cos\theta_{T}$',

    'zqq_recoil_m':          r'$m_{recoil}$ [GeV]',
    'zqq_recoil_m_tot':      r'$m_{recoil}$ [GeV]',
    'best_clustering_idx':   'Best clustering algorithm',
    'best_cluster_idx':      'Best clustering algorithm',
    'njets_inclusive':       'Number of jets (inclusive)',
    'njets_incl':            'Number of jets (inclusive)',
    'njets':                 r'n_{jets}'
}

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
