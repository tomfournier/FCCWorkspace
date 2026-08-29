def Baseline_cut_qq(ecm: int, miss: bool = False, thrust: bool = False) -> str:
    cut = ''
    if ecm == 240:
        cut += 'zqq_m > 20 && zqq_m < 140'
        cut += ' && zqq_p > 20 && zqq_p < 90'
    elif ecm == 365:
        cut += 'zqq_m > 60 && zqq_m < 200'
        cut += ' && zqq_p > 20 && zqq_p < 160'
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')
    # cut += ' && zqq_costheta > -0.85 && zqq_costheta < 0.85'
    # cut += ' && acolinearity > 0.35'
    # cut += ' && delta_mWW > 6'
    if miss:
        cut += ' && cosTheta_miss < 0.995'
    if thrust and (ecm == 365):
        cut += ' && thrust < 0.85'

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
    'leading_e':           {'name':'leading_e',
                            'title':'E_{jet,leading} [GeV]',
                            'bin':1000,'xmin':0,'xmax':250},

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
    'subleading_e':        {'name':'subleading_e',
                            'title':'E_{jet,subleading} [GeV]',
                            'bin':800,'xmin':0,'xmax':200},

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

    'zqq_e':               {'name':'zqq_e',
                            'title':'E_{jj} [GeV]',
                            'bin':2500,'xmin':0,'xmax':250},

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

    'zqq_recoil_m_tot':    {'name':'zqq_recoil_m',
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

    # Thrust variables
    'thrust':              {'name':'thrust',
                            'title':'Thrust',
                            'bin':500,'xmin':0,'xmax':1},

    'thrust_costheta':     {'name':'thrust_costheta',
                            'title':'cos(Thrust)',
                            'bin':1000,'xmin':-1,'xmax':1},

    # Distance from WW pair mass
    'delta_mWW':           {'name':'delta_mWW',
                            'title':'#Deltam_{WW} [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    'delta_mWW4':          {'name':'delta_mWW4',
                            'title':'#Deltam_{WW} (4 jets algo) [GeV]',
                            'bin':2000,'xmin':0,'xmax':200},

    # Jet clustering variables
    'best_clustering_idx': {'name':'best_clustering_idx',
                            'title':'Best clustering algorithm',
                            'bin':4,'xmin':0,'xmax':4},

    'njets':               {'name':'njets',
                            'title':'Number of jets',
                            'bin':20,'xmin':0,'xmax':20},

    # Higgstrahlungness
    'H':                   {'name':'H',
                            'title':'Higgsstrahlungness [GeV]',
                            'bin':400,'xmin':0,'xmax':200}

}
