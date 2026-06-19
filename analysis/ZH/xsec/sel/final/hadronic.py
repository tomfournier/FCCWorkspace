def Baseline_cut_qq(ecm: int) -> str:
    cut = ''
    if ecm == 240:
        cut += 'zqq_m > 20 && zqq_m < 140'
        cut += ' && zqq_p > 20 && zqq_p < 90'
    elif ecm == 365:
        cut += 'zqq_m > 60 && zqq_m < 200'
        cut += ' && zqq_p > 20 && zqq_p < 160'
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')
    cut += ' && zqq_costheta > -0.85 && zqq_costheta < 0.85'
    cut += ' && acolinearity > 0.35'
    cut += ' && delta_mWW > 6'
    cut += ' && cosTheta_miss < 0.995'
    cut += ' && thrust < 0.85' if ecm == 365 else ''

    return cut


custom_hists_qq = {
    'best_cluster_idx': {'name':'best_cluster_idx',
                         'title':'Best clustering algorithm'},

    'njets_inclusive':  {'name':'njets_inclusive',
                         'title':'Number of jets (inclusive)'},

    'njets_incl':       {'name':'njets_incl',
                         'title':'Number of jets (inclusive)'}
}


# Output histogram definitions (name, title, binning)
histos_qq = {

    # Lepton kinematics: leading lepton
    'leading_p':           {'name':'leading_p',
                            'title':'p_{jet,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_pT':          {'name':'leading_pT',
                            'title':'p_{T,jet,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

    'leading_theta':       {'name':'leading_theta',
                            'title':'#theta_{jet,leading}',
                            'bin':128, 'xmin':0, 'xmax':3.2},

    'leading_costheta':    {'name':'leading_costheta',
                            'title':'cos#theta_{jet,leading}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Lepton kinematics: subleading lepton
    'subleading_p':        {'name':'subleading_p',
                            'title':'p_{jet,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_pT':       {'name':'subleading_pT',
                            'title':'p_{T,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

    'subleading_theta':    {'name':'subleading_theta',
                            'title':'#theta_{jet,subleading}',
                            'bin':128, 'xmin':0, 'xmax':3.2},

    'subleading_costheta': {'name':'subleading_costheta',
                            'title':'cos#theta_{jet,subleading}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Angular separation between leptons
    'acolinearity':        {'name':'acolinearity',
                            'title':'#Delta#alpha_{jj}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'acopolarity':         {'name':'acopolarity',
                            'title':'#Delta#theta_{jj}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'acoplanarity':        {'name':'acoplanarity',
                            'title':'#pi-#Delta#phi_{jj}',
                            'bin':256,'xmin':0,'xmax':3.2},

    'deltaR':              {'name':'deltaR',
                            'title':'#DeltaR',
                            'bin':1000,'xmin':1,'xmax':20},

    # Z boson properties
    'zqq_m':               {'name':'zqq_m',
                            'title':'m_{jj} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'zqq_p':               {'name':'zqq_p',
                            'title':'p_{jj} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zqq_pT':              {'name':'zqq_pT',
                            'title':'p_{T,jj} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

    'zqq_theta':           {'name':'zqq_theta',
                            'title':'#theta_{jj}',
                            'bin':128,'xmin':0,'xmax':3.2},

    'zqq_costheta':        {'name':'zqq_costheta',
                            'title':'cos#theta_{jj}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Recoil mass (Higgs candidate)
    'zqq_recoil_m':        {'name':'zqq_recoil_m',
                            'title':'m_{recoil} [GeV]',
                            'bin':200,'xmin':100,'xmax':150},

    'zqq_m_recoil_m':      {'cols':['zqq_recoil_m', 'zqq_m'],
                            'title':'m_{jj} - m_{recoil} [GeV]',
                            'bins':[(100, 100, 150), (120, 60, 120)]},

    'zqq_m_recoil_m_test': {'cols':['zqq_recoil_m', 'zqq_m'],
                            'title':'m_{jj} - m_{recoil} [GeV]',
                            'bins':[(50, 100, 150), (60, 60, 120)]},

    'zqq_m_recoil_m_test1':{'cols':['zqq_recoil_m', 'zqq_m'],
                            'title':'m_{jj} - m_{recoil} [GeV]',
                            'bins':[(100, 100, 150), (200, 40, 140)]},

    'zqq_m_recoil_m_test2':{'cols':['zqq_recoil_m', 'zqq_m'],
                            'title':'m_{jj} - m_{recoil} [GeV]',
                            'bins':[(50, 100, 150), (100, 40, 140)]},

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

    # Thrust variables
    'thrust':              {'name':'thrust',
                            'title':'Thrust',
                            'bin':500,'xmin':0,'xmax':1},

    'thrust_costheta':     {'name':'thrust_costheta',
                            'title':'cos(Thrust)',
                            'bin':1000,'xmin':-1,'xmax':1},


    # WW pair variables (4 jets clustering): 1st W boson
    'W1_m':                {'name':'W1_m',
                            'title':'m_{W1} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'W1_p':                {'name':'W1_p',
                            'title':'p_{W1} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'W1_theta':            {'name':'W1_theta',
                            'title':'#theta_{W1}',
                            'bin':128,'xmin':0,'xmax':3.2},

    'W1_costheta':         {'name':'W1_costheta',
                            'title':'cos#theta_{W1}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # WW pair variables (4 jets clustering): 2nd W boson
    'W2_m':                {'name':'W2_m',
                            'title':'m_{W2} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'W2_p':                {'name':'W2_p',
                            'title':'p_{W2} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'W2_theta':            {'name':'W2_theta',
                            'title':'#theta_{W2}',
                            'bin':128,'xmin':0,'xmax':3.2},

    'W2_costheta':         {'name':'W2_costheta',
                            'title':'cos#theta_{W2}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # ZZ pair variables (4 jets clustering): 1st Z boson
    'Z1_m':                {'name':'Z1_m',
                            'title':'m_{Z1} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'Z1_p':                {'name':'Z1_p',
                            'title':'p_{Z1} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'Z1_theta':            {'name':'Z1_theta',
                            'title':'#theta_{Z1}',
                            'bin':128,'xmin':0,'xmax':3.2},

    'Z1_costheta':         {'name':'Z1_costheta',
                            'title':'cos#theta_{Z1}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # ZZ pair variables (4 jets clustering): 2nd Z boson
    'Z2_m':                {'name':'Z2_m',
                            'title':'m_{Z2} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'Z2_p':                {'name':'Z2_p',
                            'title':'p_{Z2} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'Z2_theta':            {'name':'Z2_theta',
                            'title':'#theta_{Z2}',
                            'bin':128,'xmin':0,'xmax':3.2},

    'Z2_costheta':         {'name':'Z2_costheta',
                            'title':'cos#theta_{Z2}',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Distance from WW or ZZ pair mass
    'delta_mWW':           {'name':'delta_mWW',
                            'title':'#Deltam_{WW} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'delta_mZZ':           {'name':'delta_mZZ',
                            'title':'#Deltam_{ZZ} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    # Jet clustering variables
    'best_clustering_idx': {'name':'best_clustering_idx',
                            'title':'Best clustering algorithm',
                            'bin':4,'xmin':0,'xmax':4},

    'njets':               {'name':'njets',
                            'title':'Number of jets',
                            'bin':20,'xmin':0,'xmax':20},

}
