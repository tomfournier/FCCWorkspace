def Baseline_cut_ll(ecm: int) -> str:
    cut = 'zll_m > 86 && zll_m < 96'

    if ecm == 240:
        cut += ' && zll_p > 20 && zqq_p < 70'
    elif ecm == 365:
        cut += ' && zll_p > 50 && zll_p < 150'
        cut += ' && zll_recoil_m > 100 && zll_recoil_m < 150'
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    return cut


# Output histogram definitions (name, title, binning)
histos_ll = {

    # Lepton kinematics: leading lepton
    'leading_p':           {'name':'leading_p',
                            'title':'p_{l,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_pT':          {'name':'leading_pT',
                            'title':'p_{T,l,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_theta':       {'name':'leading_theta',
                            'title':'#theta_{l,leading}',
                            'bin':128, 'xmin':0, 'xmax':3.2},

    # Lepton kinematics: subleading lepton
    'subleading_p':        {'name':'subleading_p',
                            'title':'p_{l,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_pT':       {'name':'subleading_pT',
                            'title':'p_{T,l,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_theta':    {'name':'subleading_theta',
                            'title':'#theta_{l,subleading}',
                            'bin':128, 'xmin':0, 'xmax':3.2},

    # Angular separation between leptons
    'acolinearity':        {'name':'acolinearity',
                            'title':'#Delta#alpha_{l^{+}l^{-}}',
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
                            'bin':100,'xmin':86,'xmax':96},

    'zll_p':               {'name':'zll_p',
                            'title':'p_{l^{+}l^{-}} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zll_pT':              {'name':'zll_pT',
                            'title':'p_{T,l^{+}l^{-}} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zll_theta':           {'name':'zll_theta',
                            'title':'#theta_{l^{+}l^{-}}',
                            'bin':128,'xmin':0,'xmax':3.2},

    # Recoil mass (Higgs candidate)
    'zll_recoil_m':        {'name':'zll_recoil_m',
                            'title':'m_{recoil} [GeV]',
                            'bin':200,'xmin':100,'xmax':150},

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
                            'title':'Higgsstrahlungness [GeV^{2}]',
                            'bin':110,'xmin':0,'xmax':110},

}
