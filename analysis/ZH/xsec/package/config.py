import time, ROOT

z_decays = ['bb', 'cc', 'ss', 'qq', 'ee', 'mumu', 'tautau', 'nunu']
h_decays = ['bb', 'cc', 'gg', 'ss', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa']
H_decays = ['bb', 'cc', 'gg', 'ss', 'mumu', 'tautau', 'ZZ', 'WW', 'Za', 'aa', 'inv']
quarks   = ['bb', 'cc', 'ss', 'qq']

h_labels = {
    'bb': 'H#rightarrowb#bar{b}', 
    'cc': 'H#rightarrowc#bar{c}', 
    'ss': 'H#rightarrows#bar{s}', 
    'gg': 'H#rightarrowgg', 
    'mumu': 'H#rightarrow#mu^{#plus}#mu^{#minus}', 
    'tautau': 'H#rightarrow#tau^{#plus}#tau^{#minus}', 
    'ZZ': 'H#rightarrowZZ*', 
    'WW': 'H#rightarrowWW*', 
    'Za': 'H#rightarrowZ#gamma', 
    'aa': 'H#rightarrow#gamma#gamma', 
    'inv': 'H#rightarrowInv'
}

h_colors = {
    'bb': ROOT.kViolet, 
    'cc': ROOT.kBlue , 
    'ss': ROOT.kRed, 
    'gg': ROOT.kGreen+1,
    'mumu': ROOT.kOrange, 
    'tautau': ROOT.kCyan, 
    'ZZ': ROOT.kGray, 
    'WW': ROOT.kGray+2, 
    'Za': ROOT.kGreen+2, 
    'aa': ROOT.kRed+2, 
    'inv': ROOT.kBlue+2
}

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

# colors from https://github.com/mpetroff/accessible-color-cycles
colors = {
    'ZH'       : ROOT.TColor.GetColor('#e42536'),
    'ZqqH'     : ROOT.TColor.GetColor('#e42536'),
    'ZmumuH'   : ROOT.TColor.GetColor('#e42536'),
    'ZnunuH'   : ROOT.TColor.GetColor('#e42536'),
    'ZeeH'     : ROOT.TColor.GetColor('#e42536'),
    
    'zh'       : ROOT.TColor.GetColor('#e42536'),
    'zqqh'     : ROOT.TColor.GetColor('#e42536'),
    'zmumuh'   : ROOT.TColor.GetColor('#e42536'),
    'znunuh'   : ROOT.TColor.GetColor('#e42536'),
    'zeeh'     : ROOT.TColor.GetColor('#e42536'),

    'WW'       : ROOT.TColor.GetColor('#f89c20'),
    'ZZ'       : ROOT.TColor.GetColor('#5790fc'),
    'Zgamma'   : ROOT.TColor.GetColor('#964a8b'),
    'Zqqgamma' : ROOT.TColor.GetColor('#964a8b'),
    'Rare'     : ROOT.TColor.GetColor('#9c9ca1')
}

#############################
##### VARIABLES FOR BDT #####
#############################

#First stage BDT including event-level vars
vars = [
    'leading_p', 'leading_theta', 'subleading_p', 'subleading_theta',
    'acolinearity', 'acoplanarity', 'zll_m', 'zll_p', 'zll_theta'
]

# Latex mapping for importance plot
vars_label = {
    'leading_p':        r'$p_{\ell,leading}$',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'leading_phi':      r'$\phi_{\ell, leading}$',

    'subleading_p':     r'$p_{\ell,subleading}$',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'subleading_phi':   r'$\phi_{\ell, subleading}$',
    
    'acolinearity':     r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'zll_deltaR':       r'$\Delta R$',

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

# Latex mapping for histcheck plot
vars_xlabel = {
    'leading_p':        r'$p_{\ell,leading}$ [GeV]',
    'leading_theta':    r'$\theta_{\ell,leading}$',
    'leading_phi':      r'$\phi_{\ell, leading}$',
    
    'subleading_p':     r'$p_{\ell,subleading}$ [GeV]',
    'subleading_theta': r'$\theta_{\ell,subleading}$',
    'subleading_phi':   r'$\phi_{\ell, subleading}$',
    
    'acolinearity':     r'$\Delta\theta_{\ell^{+}\ell^{-}}$',
    'acoplanarity':     r'$\pi - \Delta\phi_{\ell^{+}\ell^{-}}$',
    'zll_deltaR':       r'$\Delta R$',

    'zll_m':            r'$m_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_p':            r'$p_{\ell^{+}\ell^{-}}$ [GeV]',
    'zll_theta':        r'$\theta_{\ell^{+}\ell^{-}}$',
    'zll_phi':          r'$\phi_{\ell^{+}\ell^{-}}$',
    
    'zll_recoil_m':     r'$m_{recoil}$ [GeV]',
    
    'cosTheta_miss':    r'$\cos\theta_{miss}$',
    'visibleEnergy':    r'$E_{vis}$ [GeV]',
    'missingMass':      r'$m_{miss}$ [GeV]',

    'H':                r'$H$ [GeV$^{2}$]',
    'BDTscore':         r'BDT Score'
}

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

#______________________________
def warning(log_msg: str, 
            lenght: int = -1, 
            abort_msg: str = ''
            ) -> None:
    if not abort_msg:
        abort_msg = ' ERROR CODE '
    if lenght==-1:
        if len(log_msg) < len(abort_msg) + 6:
            lenght = len(abort_msg) + 6
        else:
            lenght = len(log_msg) + 6
    msg =  f'\n{abort_msg:=^{lenght}}\n'
    msg += f'{log_msg:^{lenght}}\n'
    sep = '=' * lenght
    msg += f'{sep:^{lenght}}\n'
    raise Exception(msg)

#___________________________
def timer(t: float) -> None:
    dt = time.time() - t
    h, m  = int(dt // 3600), int(dt // 60 % 60), 
    s, ms = int(dt % 60), int((dt % 1) * 1000) 

    time_parts = []
    if h>0:
        time_parts.append(f'{h} h')
    if m>0:
        time_parts.append(f'{m} min')
    if s>0:
        time_parts.append(f'{s} s')
    if ms>0:
        time_parts.append(f'{ms} ms')

    full_time = ' '.join(time_parts)

    header = ' CODE ENDED '
    elapsed = f'Elapsed time: {full_time}'

    lenght = len(elapsed) + 4
    sep = lenght * '='

    print(f'\n{header:=^{lenght}}\n{elapsed:^{lenght}}\n{sep}\n')

#____________________________________________
def mk_processes(procs:    list[str] = [],
                 z_decays: list[str] = z_decays, 
                 h_decays: list[str] = h_decays, 
                 H_decays: list[str] = H_decays, 
                 quarks:   list[str] = quarks,
                 ecm: int = 240
                 ) -> dict[str, list[str]]:
    processes = {
        'ZH'        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays 
                                                      for y in h_decays],
        'ZmumuH'    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
        'ZeeH'      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_decays],
        'ZqqH'      : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in quarks
                                                      for y in h_decays],

        # Include H -> Inv decay
        'zh'        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays 
                                                      for y in H_decays],
        'zmumuh'    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in H_decays],
        'zeeh'      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in H_decays],
        'zqqh'      : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in quarks
                                                      for y in H_decays],
        

        'WW'        : [f'p8_ee_WW_ecm{ecm}', 
                       f'p8_ee_WW_mumu_ecm{ecm}', 
                       f'p8_ee_WW_ee_ecm{ecm}'],

        'ZZ'        : [f'p8_ee_ZZ_ecm{ecm}'],

        'Zgamma'    : [f'wzp6_ee_tautau_ecm{ecm}', 
                       f'wzp6_ee_mumu_ecm{ecm}',
                       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],

        'Rare'      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
                       f'wzp6_gammae_eZ_Zee_ecm{ecm}',   f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
                       f'wzp6_gaga_ee_60_ecm{ecm}',      f'wzp6_gaga_mumu_60_ecm{ecm}', 
                       f'wzp6_gaga_tautau_60_ecm{ecm}',  f'wzp6_ee_nuenueZ_ecm{ecm}'],
    }
    if procs:
        new_processes = {proc: processes[proc] for proc in procs if proc in processes}
        return new_processes
    return processes
            