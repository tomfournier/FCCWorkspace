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

    'leading_phi':         {'name':'leading_phi',
                            'title':'#phi_{l,leading}',
                            'bin':64,'xmin':-3.2,'xmax':3.2},

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

    'subleading_phi':      {'name':'subleading_phi',
                            'title':'#phi_{l,subleading}',
                            'bin':64,'xmin':-3.2,'xmax':3.2},

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

    'zll_costheta':        {'name':'zll_costheta',
                            'title':'cos#theta_{l^{+}l^{-}}',
                            'bin':1000,'xmin':-1,'xmax':1},

    'zll_phi':             {'name':'zll_phi',
                            'title':'#phi_{l^{+}l^{-}}',
                            'bin':64,'xmin':-3.2,'xmax':3.2},

    'zll_category':        {'name':'zll_category',
                            'title':'Category',
                            'bin':11,'xmin':0,'xmax':10},

    # Recoil mass (Higgs candidate)
    'zll_recoil_m':        {'name':'zll_recoil_m',
                            'title':'m_{recoil} [GeV]',
                            'bin':200,'xmin':100,'xmax':150},

    'zll_recoil_p':        {'name':'zll_recoil_p',
                            'title':'p_{recoil} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    # Energy inbalance variables
    'e_long':              {'name':'e_long',
                            'title':'E_{long} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    'e_trans':             {'name':'e_trans',
                            'title':'E_{trans} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    'e_tan':               {'name':'e_tan',
                            'title':'E_{trans}/E_{long}',
                            'bin':500,'xmin':-5000,'xmax':5000},

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

    'visibleEnergy_tot':   {'name':'visibleEnergy_tot',
                            'title':'E_{vis, tot} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    'missingMass':         {'name':'missingMass',
                            'title':'m_{miss} [GeV]',
                            'bin':730,'xmin':0,'xmax':365},

    # Higgsstrahlungness
    'H':                   {'name':'H',
                            'title':'Higgsstrahlungness [GeV^{2}]',
                            'bin':110,'xmin':0,'xmax':110},

}
