from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT

def exclusive_clustering(df: 'ROOT.ROOT.RDataFrame',
                         njets: int
                         ) -> 'ROOT.ROOT.RDataFrame':
    if njets==0:  # inclusive
        df = df.Define('clustered_jets_N0', 'JetClustering::clustering_kt(0.6, 0, 5, 1, 0)(pseudo_jets)')
    else:
        df = df.Define(f'clustered_jets_N{njets}', f'JetClustering::clustering_ee_kt(2, {njets}, 0, 10)(pseudo_jets)')

    df = df.Define(f'jets_N{njets}',            f'FCCAnalyses::JetClusteringUtils::get_pseudoJets(clustered_jets_N{njets})')
    df = df.Define(f'njets_N{njets}',           f'jets_N{njets}.size()')
    df = df.Define(f'jetconstituents_N{njets}', f'FCCAnalyses::JetClusteringUtils::get_constituents(clustered_jets_N{njets})')
    df = df.Define(f'jets_e_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_e(jets_N{njets})')
    df = df.Define(f'jets_px_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_px(jets_N{njets})')
    df = df.Define(f'jets_py_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_py(jets_N{njets})')
    df = df.Define(f'jets_pz_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_pz(jets_N{njets})')
    df = df.Define(f'jets_m_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_m(jets_N{njets})')
    df = df.Define(f'jets_rp_N{njets}',         f'FCCAnalyses::jets2rp(jets_px_N{njets}, jets_py_N{njets}, jets_pz_N{njets}, jets_e_N{njets}, jets_m_N{njets})')
    df = df.Define(f'jets_rp_cand_N{njets}',    f'FCCAnalyses::select_jets(jets_rp_N{njets}, jetconstituents_N{njets}, {njets}, ReconstructedParticles)')  # reduces potentially the jet multiplicity
    df = df.Define(f'njets_cand_N{njets}',      f'jets_rp_cand_N{njets}.size()')
    return df

def define_variables(df: 'ROOT.ROOT.RDataFrame',
                     njets: int
                     ) -> 'ROOT.ROOT.RDataFrame':
    df = df.Define(f'zbuilder_result_N{njets}', f'FCCAnalyses::resonanceBuilder_mass_recoil_hadronic(91.2, 125, 0.0, ecm)(jets_rp_cand_N{njets})')
    df = df.Define(f'zqq_N{njets}',             f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_N{njets}[0]}}')  # the Z
    df = df.Define(f'zqq_jets_N{njets}',        f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_N{njets}[1],zbuilder_result_N{njets}[2]}}')  # the leptons
    df = df.Define(f'zqq_m_N{njets}',           f'FCCAnalyses::ReconstructedParticle::get_mass(zqq_N{njets})[0]')
    df = df.Define(f'zqq_p_N{njets}',           f'FCCAnalyses::ReconstructedParticle::get_p(zqq_N{njets})[0]')
    df = df.Define(f'zqq_recoil_N{njets}',      f'FCCAnalyses::ReconstructedParticle::recoilBuilder(ecm)(zqq_N{njets})')
    df = df.Define(f'zqq_recoil_m_N{njets}',    f'FCCAnalyses::ReconstructedParticle::get_mass(zqq_recoil_N{njets})[0]')
    return df


def veto_leptonic(df: 'ROOT.ROOT.RDataFrame',
                  ecm: int,
                  cat: str
                  ) -> 'ROOT.ROOT.RDataFrame':

    if cat == 'mumu': df = df.Define(f'leps_{cat}', 'FCCAnalyses::ReconstructedParticle::sel_p(20)(muons_all)')
    elif cat == 'ee': df = df.Define(f'leps_{cat}', 'FCCAnalyses::ReconstructedParticle::sel_p(20)(electrons_all)')
    else: raise ValueError(f'cat = {cat} not supported')

    df = df.Define(f'leps_{cat}_q',       f'FCCAnalyses::ReconstructedParticle::get_charge(leps_{cat})')
    df = df.Define(f'leps_{cat}_no',      f'FCCAnalyses::ReconstructedParticle::get_n(leps_{cat})')
    df = df.Define(f'leps_{cat}_iso',     f'FCCAnalyses::coneIsolation(0.01, 0.5)(leps_{cat}, ReconstructedParticles)')
    df = df.Define(f'leps_{cat}_sel_iso', f'FCCAnalyses::sel_iso(0.25)(leps_{cat}, leps_{cat}_iso)')  # 0.25


    # Do not veto the H-ll candidates, it suppresses HZZ
    # Tighter selection in leptonic channel, can safely remove this here
    df = df.Define(f'zbuilder_result_H{cat}', f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, ecm, false)'
                   f'(leps_{cat}, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define(f'zll_H{cat}',             f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_H{cat}[0]}}')  # the Z
    df = df.Define(f'zll_H{cat}_m',           f'FCCAnalyses::ReconstructedParticle::get_mass(zll_H{cat})[0]')
    df = df.Define(f'zll_leps_H{cat}',        f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_H{cat}[1],zbuilder_result_H{cat}[2]}}')  # the leptons
    df = df.Define(f'zll_leps_dummy_{cat}',    'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{}}')  # the leptons
    df = df.Define(f'leps_to_remove_{cat}',   f'return (zll_H{cat}_m > (125-3) && zll_H{cat}_m < (125+3)) ? zll_leps_H{cat} : zll_leps_dummy_{cat}')
    df = df.Define(f'leps_good_{cat}',        f'FCCAnalyses::ReconstructedParticle::remove(leps_{cat}, leps_to_remove_{cat})')


    df = df.Define(f'zbuilder_result_{cat}', f'FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, ecm, false)'
                   f'(leps_{cat}, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define(f'zll_{cat}',             f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_{cat}[0]}}')  # the Z
    df = df.Define(f'zll_leps_{cat}',        f'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{{zbuilder_result_{cat}[1],zbuilder_result_{cat}[2]}}')  # the leptons
    df = df.Define(f'zll_m_{cat}',           f'FCCAnalyses::ReconstructedParticle::get_mass(zll_{cat})[0]')
    df = df.Define(f'zll_p_{cat}',           f'FCCAnalyses::ReconstructedParticle::get_p(zll_{cat})[0]')
    df = df.Define(f'zll_recoil_{cat}',      f'FCCAnalyses::ReconstructedParticle::recoilBuilder(ecm)(zll_{cat})')
    df = df.Define(f'zll_recoil_m_{cat}',    f'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil_{cat})[0]')

    sel_leps =   f'leps_{cat}_no >= 2 && leps_{cat}_sel_iso.size() > 0 && abs(Sum(leps_{cat}_q)) < leps_{cat}_q.size()'
    sel_mll =    f'zll_m_{cat} > 86 && zll_m_{cat} < 96'
    if ecm == 240:   sel_pll = f'zll_p_{cat} > 20 && zll_p_{cat} < 70'
    elif ecm == 365: sel_pll = f'zll_p_{cat} > 50 && zll_p_{cat} < 150'
    else: raise ValueError(f'ecm = {ecm} is not supported')
    sel_recoil = f'zll_recoil_m_{cat} > 100 && zll_recoil_m_{cat} < 150'

    df = df.Filter(f'!({sel_leps} && {sel_mll} && {sel_pll} && {sel_recoil})')
    return df

def presel_qq(df: 'ROOT.ROOT.RDataFrame',
              ecm: int,
              dataset: str
              ) -> tuple['ROOT.ROOT.RDataFrame',
                         list['ROOT.TH1',
                              'ROOT.TParameter']]:

    hists = []

    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')

    if 'p8_ee_WW_ecm' in dataset:  # remove muons/electrons from inclusive WW
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    df = df.Alias('Muon0',     'Muon#0.index')
    df = df.Alias('Electron0', 'Electron#0.index')
    df = df.Alias('Photon0',   'Photon#0.index')
    df = df.Define('muons_all',     'FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)')
    df = df.Define('electrons_all', 'FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)')
    df = df.Define('photons_all',   'FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)')

    ##########
    ### CUT 0: all events
    ##########

    ##########
    ### CUT 1: Veto w.r.t leptonic channel to ensure orthogonality
    ##########
    df = veto_leptonic(df, 'mumu')
    df = veto_leptonic(df, 'ee')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut1'))

    # Remove isolated photons and leptons from clustering
    # Ensure orthogonality with leptonic channel
    df = df.Define('muons_all_p',     'FCCAnalyses::ReconstructedParticle::get_p(muons_all)')
    df = df.Define('photons_all_p',   'FCCAnalyses::ReconstructedParticle::get_p(photons_all)')
    df = df.Define('electrons_all_p', 'FCCAnalyses::ReconstructedParticle::get_p(electrons_all)')
    if ecm == 240:
        df = df.Define('muons',     'FCCAnalyses::sel_range(40, 95, false)(muons_all, muons_all_p)')
        df = df.Define('photons',   'FCCAnalyses::sel_range(40, 95, false)(photons_all, photons_all_p)')
        df = df.Define('electrons', 'FCCAnalyses::sel_range(40, 95, false)(electrons_all, electrons_all_p)')
    else:
        df = df.Define('muons',     'FCCAnalyses::sel_range(20, 170, false)(muons_all, muons_all_p)')
        df = df.Define('photons',   'FCCAnalyses::sel_range(20, 170, false)(photons_all, photons_all_p)')
        df = df.Define('electrons', 'FCCAnalyses::sel_range(20, 170, false)(electrons_all, electrons_all_p)')

    df = df.Define('rps_no_photons',                 'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons)')
    df = df.Define('rps_no_photons_muons',           'FCCAnalyses::ReconstructedParticle::remove(rps_no_photons, muons)')
    df = df.Define('rps_no_photons_muons_electrons', 'FCCAnalyses::ReconstructedParticle::remove(rps_no_photons_muons, electrons)')

    # Define PF candidates collection by removing the muons
    df = df.Alias('rps_sel',      'rps_no_photons_muons_electrons')
    df = df.Define('RP_px',       'FCCAnalyses::ReconstructedParticle::get_px(rps_sel)')
    df = df.Define('RP_py',       'FCCAnalyses::ReconstructedParticle::get_py(rps_sel)')
    df = df.Define('RP_pz',       'FCCAnalyses::ReconstructedParticle::get_pz(rps_sel)')
    df = df.Define('RP_e',        'FCCAnalyses::ReconstructedParticle::get_e(rps_sel)')
    df = df.Define('RP_m',        'FCCAnalyses::ReconstructedParticle::get_mass(rps_sel)')
    df = df.Define('RP_q',        'FCCAnalyses::ReconstructedParticle::get_charge(rps_sel)')
    df = df.Define('pseudo_jets', 'FCCAnalyses::JetClusteringUtils::set_pseudoJets(RP_px, RP_py, RP_pz, RP_e)')

    # Perform possible clusterings
    for i in [0, 2, 4, 6]:
        df = exclusive_clustering(df, i)
        df = define_variables(df, i)

    df = df.Define('zqq',                 'std::vector<Vec_rp> r = {zqq_N0, zqq_N2, zqq_N4, zqq_N6}; return r;')
    df = df.Define('zqq_jets',            'std::vector<Vec_rp> r = {jets_rp_cand_N0, jets_rp_cand_N2, jets_rp_cand_N4, jets_rp_cand_N6}; return r;')
    df = df.Define('zqq_m',               'Vec_f r = {zqq_m_N0, zqq_m_N2, zqq_m_N4, zqq_m_N6}; return r;')
    df = df.Define('zqq_p',               'Vec_f r = {zqq_p_N0, zqq_p_N2, zqq_p_N4, zqq_p_N6}; return r;')
    df = df.Define('zqq_recoil_m',        'Vec_f r = {zqq_recoil_m_N0, zqq_recoil_m_N2, zqq_recoil_m_N4, zqq_recoil_m_N6}; return r;')
    df = df.Define('njets',               'Vec_i r = {(int)njets_cand_N0, (int)njets_cand_N2, (int)njets_cand_N4, (int)njets_cand_N6}; return r;')
    df = df.Define('njets_target',        'Vec_i r = {0, 2, 4, 6}; return r;')
    df = df.Define('best_clustering_idx', 'FCCAnalyses::best_clustering_idx(zqq_m, zqq_p, zqq_recoil_m, njets, njets_target, ecm)')

    hists.append(df.Histo1D(('best_clustering_idx_nosel', '', *(15, -5, 10)), 'best_clustering_idx'))

    # njets for inclusive clustering
    hists.append(df.Histo1D(('njets_inclusive', '', *(50, 0, 50)), 'njets_cand_N0'))
    df_incl = df.Filter('best_clustering_idx == 0')
    hists.append(df_incl.Histo1D(('njets_inclusive_sel', '', *(50, 0, 50)), 'njets_cand_N0'))

    ##########
    ### CUT 2: remove events failing clustering
    ##########
    df = df.Filter('best_clustering_idx >= 0')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut2'))  # after clustering

    df = df.Define('zqq_best',               'zqq[best_clustering_idx]')
    df = df.Define('zqq_jets_best',          'zqq_jets[best_clustering_idx]')
    df = df.Define('zqq_m_best',             'zqq_m[best_clustering_idx]')
    df = df.Define('zqq_recoil_m_best',      'zqq_recoil_m[best_clustering_idx]')
    df = df.Define('zqq_p_best',             'zqq_p[best_clustering_idx]')
    df = df.Define('zqq_jets_p_best',        'FCCAnalyses::ReconstructedParticle::get_p(zqq_jets_best)')
    df = df.Define('zqq_jets_theta_best',    'FCCAnalyses::ReconstructedParticle::get_theta(zqq_jets_best)')
    df = df.Define('zqq_jets_costheta_best', 'Vec_f ret; for(auto & theta: zqq_jets_theta_best) ret.push_back(std::abs(cos(theta))); return ret;')

    df = df.Define('z_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(zqq_best)')
    df = df.Define('z_costheta', 'std::abs(cos(z_theta[0]))')

    # Jet kinematics
    df = df.Define('leading_idx',             '(zqq_jets_p_best[0] > zqq_jets_p_best[1]) ? 0 : 1')
    df = df.Define('subleading_idx',          '(zqq_jets_p_best[0] > zqq_jets_p_best[1]) ? 1 : 0')
    df = df.Define('leading_jet_p',           'zqq_jets_p_best[leading_idx]')
    df = df.Define('subleading_jet_p',        'zqq_jets_p_best[subleading_idx]')
    df = df.Define('leading_jet_costheta',    'zqq_jets_costheta_best[leading_idx]')
    df = df.Define('subleading_jet_costheta', 'zqq_jets_costheta_best[subleading_idx]')

    # Attempt to reconstruct WW with 4 jets
    df = df.Define('pairs_WW_N4', 'FCCAnalyses::pair_WW_N4(jets_rp_cand_N4)')
    df = df.Define('W1', 'pairs_WW_N4[0]')
    df = df.Define('W2', 'pairs_WW_N4[1]')
    df = df.Define('W1_m', 'W1.M()')
    df = df.Define('W2_m', 'W2.M()')
    df = df.Define('W1_p', 'W1.P()')
    df = df.Define('W2_p', 'W2.P()')
    df = df.Define('W1_costheta', 'std::abs(W1.Theta())')
    df = df.Define('W2_costheta', 'std::abs(W1.Theta())')
    df = df.Define('delta_mWW',   'std::sqrt((W1_m-78)*(W1_m-78) + (W2_m-78)*(W2_m-78))')

    df = df.Define('acolinearity', 'FCCAnalyses::acolinearity(zqq_jets_best)')
    df = df.Define('acoplanarity', 'FCCAnalyses::acoplanarity(zqq_jets_best)')

    df = df.Define('missingEnergy_rp', f'FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)')
    df = df.Define('missingEnergy',    'missingEnergy_rp[0].energy')
    df = df.Define('cosTheta_Miss',    'FCCAnalyses::get_cosTheta_miss(missingEnergy_rp)')
    df = df.Define('missingMass',      f'FCCAnalyses::missingMass({ecm}, ReconstructedParticles)')

    # Thrust
    df = df.Define('rps_charged_idx', 'RP_q != 0')  # select only charged particles
    df = df.Define('rps_charged',     'ReconstructedParticles[rps_charged_idx]')
    df = df.Define('rps_charged_p',   'FCCAnalyses::ReconstructedParticle::get_p(rps_charged)')
    df = df.Define('rps_charged_n',   'FCCAnalyses::ReconstructedParticle::get_n(rps_charged)')
    df = df.Define('RP_px_q', 'RP_px[rps_charged_idx]')
    df = df.Define('RP_py_q', 'RP_py[rps_charged_idx]')
    df = df.Define('RP_pz_q', 'RP_pz[rps_charged_idx]')
    df = df.Define('max_p_idx', 'FCCAnalyses::get_max_idx(rps_charged_p)')

    df = df.Define('thrust', 'FCCAnalyses::Algorithms::calculate_thrust()(RP_px, RP_py, RP_pz)')
    df = df.Define('thrust_magn', 'thrust[0]')
    df = df.Define('thrust_costheta', 'abs(thrust[3])')



    ###########################
    ### KINEMATIC SELECTION ###
    ###########################

    ##########
    ### CUT 3: Z mass window
    ##########
    if ecm == 240: df = df.Filter('zqq_m_best > 20 && zqq_m_best < 140')  # loose
    else:          df = df.Filter('zqq_m_best > 20 && zqq_m_best < 200')  # loose
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut3'))


    ##########
    ### CUT 4: Z momentum (CoM dependent)
    ##########
    if ecm == 240: df = df.Filter('zqq_p_best < 90 && zqq_p_best > 20')
    else:          df = df.Filter('zqq_p_best > 60 && zqq_p_best < 160')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut4'))

    ##########
    ### CUT 5: Z Polar angle
    ##########
    df = df.Filter('z_costheta < 0.85')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut5'))

    ##########
    ### CUT 6: Acolinearity
    ##########
    df = df.Filter('acolinearity > 0.35')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut6'))

    ##########
    ### CUT 7: Acoplanarity
    ##########
    df = df.Filter('acoplanarity < 5')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut7'))

    ##########
    ### CUT 8: WW pair removal
    ##########
    df = df.Filter('delta_mWW > 6')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut7'))

    ##########
    ### CUT 9: Polar angle of the missing energy
    ##########
    df = df.Filter('cosTheta_Miss < 0.995')
    # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut8'))

    ###########
    ### CUT 10: Thrust (365 GeV)
    ###########
    if ecm == 365:
        df = df.Filter('thrust_magn < 0.85')
        # hists.append(df.Histo1D(('cutFlow', '', *bins_count), 'cut9'))

    return df, hists
