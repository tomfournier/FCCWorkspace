##########################
### BASELINE SELECTION ###
##########################

def Baseline_cut_ll(ecm: int, miss: bool = False) -> str:
    # c.o.m energy dependent cut
    p_dw, p_up = (20, 70) if ecm==240 else ((50, 150) if ecm==365 else (None, None))
    if None in (p_dw, p_up):
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    cut = ''
    # Di-lepton mass cut
    cut += ' && zll_m > 86 && zll_m < 96'

    # Di-lepton momentum cut
    cut += f' && zll_p > {p_dw} && zll_p < {p_up}'

    # Missing energy polar angle cut (optional)
    if miss: cut += ' && cosTheta_miss < 0.98'

    # Recoil mass cut (only for 365 GeV, negligible effect for 240 GeV)
    if ecm == 365: cut += ' && zll_recoil_m > 100 && zll_recoil_m < 150'

    return cut



#############################################
### CUSTOM HISTOGRAMS (FOR 3-MEASUREMENT) ###
#############################################

custom_hists_ll = {
    'leps_iso':    {'name':'ConeIsolation', 'title':'I_{rel}'},
    'leps_iso_no': {'name':'n_leptons', 'title':'Isolated leptons'}
}



######################################
### HISTOGRAMS FOR FINAL-SELECTION ###
######################################

# Output histogram definitions (name, title, binning)
histos_ll = {

    # Lepton kinematics: leading lepton
    'leading_p':           {'name':'leading_p',
                            'title':'p_{l,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_pT':          {'name':'leading_pT',
                            'title':'p_{T,l,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_costheta':    {'name':'leading_costheta',
                            'title':'cos#theta_{l,leading}',
                            'bin':1000, 'xmin':-1, 'xmax':1},

    # Lepton kinematics: subleading lepton
    'subleading_p':        {'name':'subleading_p',
                            'title':'p_{l,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_pT':       {'name':'subleading_pT',
                            'title':'p_{T,l,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_costheta': {'name':'subleading_costheta',
                            'title':'cos#theta_{l,subleading}',
                            'bin':1000, 'xmin':-1, 'xmax':1},

    # Angular separation between leptons
    'acolinearity':        {'name':'acolinearity',
                            'title':'#pi-#Delta#alpha_{l^{+}l^{-}}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'acopolarity':         {'name':'acopolarity',
                            'title':'#Delta#theta_{l^{+}l^{-}}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'acoplanarity':        {'name':'acoplanarity',
                            'title':'#pi-#Delta#phi_{l^{+}l^{-}}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'deltaR':              {'name':'deltaR',
                            'title':'#DeltaR',
                            'bin':1000,'xmin':1,'xmax':20},

    # Z boson properties
    'zll_m':               {'name':'zll_m',
                            'title':'m_{l^{+}l^{-}} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zll_p':               {'name':'zll_p',
                            'title':'p_{l^{+}l^{-}} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zll_pT':              {'name':'zll_pT',
                            'title':'p_{T,l^{+}l^{-}} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zll_costheta':        {'name':'zll_costheta',
                            'title':'cos#theta_{l^{+}l^{-}}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Recoil mass (Higgs candidate)
    'zll_recoil_m':        {'name':'zll_recoil_m',
                            'title':'m_{recoil} [GeV]',
                            'bin':200,'xmin':100,'xmax':150},

    'zll_recoil_m_tot':    {'name':'zll_recoil_m',
                            'title':'m_{recoil} [GeV]',
                            'bin':1400,'xmin':0,'xmax':350},

    # Visible and invisible information
    'cosTheta_miss':       {'name':'cosTheta_miss',
                            'title':'|cos#theta_{miss}|',
                            'bin':1000,'xmin':0,'xmax':1},

    'missingEnergy':       {'name':'missingEnergy',
                            'title':'E_{miss} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    'visibleEnergy':       {'name':'visibleEnergy',
                            'title':'E_{vis} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    'missingMass':         {'name':'missingMass',
                            'title':'m_{miss} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    # Higgsstrahlungness
    'H':                   {'name':'H',
                            'title':'Higgsstrahlungness [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    # Leptons properties
    'leps_iso_no':         {'name':'leps_iso_no',
                            'title':'Number of isolated leptons',
                            'bin':20,'xmin':0,'xmax':20}

}
