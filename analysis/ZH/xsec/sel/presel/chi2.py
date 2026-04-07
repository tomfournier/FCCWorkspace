from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT

from .helper import setup_alias
from .hadronic import veto_leptonic, make_pseudojets



#################################
### LEPTONIC HELPER FUNCTIONS ###
#################################

def get_leps(df: 'ROOT.ROOT.RDataFrame'
             ) -> 'ROOT.ROOT.RDataFrame':
    """Define baseline lepton collections and isolation selections.

    Computes lepton kinematics (p, theta, phi) for all leptons before and after
    momentum cut (p > 20 GeV). Applies cone isolation (0.01 < dR < 0.5) and
    selects isolated leptons (Irel < 0.25) for downstream analysis.

    Args:
        df: RDataFrame-like object with reconstructed particles.

    Returns:
        The dataframe with baseline lepton collections and kinematic properties:
        - All leptons: leps_all
        - Selected leptons: leps, leps_iso
        - Isolated leptons: leps_sel_iso with count leps_sel_no
    """
    # Define all lepton properties (before cuts)
    df = df.Define('leps_all', 'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')

    # Apply momentum cut (p > 20 GeV) to reduce soft backgrounds
    df = df.Define('leps', 'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')

    # Select isolated leptons: Irel < 0.25 (relative isolation ratio)
    df = df.Define('leps_q',       'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',      'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',     'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('leps_sel_iso', 'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    return df



#################################
### HADRONIC HELPER FUNCTIONS ###
#################################


def clustering(df: 'ROOT.ROOT.RDataFrame',
               njets: int
               ) -> 'ROOT.ROOT.RDataFrame':
    if njets==0:  # inclusive
        df = df.Define('clustered_jets_N0', 'JetClustering::clustering_kt(0.6, 0, 5, 1, 0)(pseudo_jets)')
    else:
        df = df.Define(f'clustered_jets_N{njets}', f'JetClustering::clustering_ee_kt(2, {njets}, 0, 10)(pseudo_jets)')

    df = df.Define(f'jets_N{njets}',            f'FCCAnalyses::JetClusteringUtils::get_pseudoJets(clustered_jets_N{njets})')
    # df = df.Define(f'njets_N{njets}',           f'jets_N{njets}.size()')
    # df = df.Define(f'jetconstituents_N{njets}', f'FCCAnalyses::JetClusteringUtils::get_constituents(clustered_jets_N{njets})')
    # df = df.Define(f'jets_e_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_e(jets_N{njets})')
    # df = df.Define(f'jets_px_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_px(jets_N{njets})')
    # df = df.Define(f'jets_py_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_py(jets_N{njets})')
    # df = df.Define(f'jets_pz_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_pz(jets_N{njets})')
    # df = df.Define(f'jets_m_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_m(jets_N{njets})')
    # df = df.Define(f'jets_rp_N{njets}',         f'FCCAnalyses::jets2rp(jets_px_N{njets}, jets_py_N{njets}, jets_pz_N{njets}, jets_e_N{njets}, jets_m_N{njets})')
    df = df.Define(f'jets_rp_N{njets}',         f'FCCAnalyses::jets2rp(jets_N{njets})')
    df = df.Define(f'jets_rp_cand_N{njets}',    f'FCCAnalyses::select_jets(jets_rp_N{njets}, {njets}, ReconstructedParticles)')  # reduces potentially the jet multiplicity
    return df



######################
### MAIN FUNCTIONS ###
######################

def optimize_ll(df: 'ROOT.ROOT.RDataFrame',
                cat: str,
                ecm: int,
                dataset: str) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)

    if 'p8_ee_WW_ecm' in dataset:  # remove muons/electrons from inclusive WW
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    df = get_leps(df)

    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps.size()')

    df = df.Define('all_pairs', f'FCCAnalyses::leptonicZBuilder({ecm}, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)')
    df = df.Define('trueZ',     f'FCCAnalyses::getTrueZ("{cat}", Particle, Particle0, Particle1)')

    df = df.Filter('trueZ.z_system.mass >= 0')

    # Extract serializable fields from all_pairs
    df = df.Define('z_rp',   'FCCAnalyses::getZPairsZSystem(all_pairs)')
    df = df.Define('l1',     'FCCAnalyses::getZPairsLepton1(all_pairs)')
    df = df.Define('l2',     'FCCAnalyses::getZPairsLepton2(all_pairs)')

    df = df.Define('z_e',     'FCCAnalyses::ReconstructedParticle::get_e(z_rp)')
    df = df.Define('z_p',     'FCCAnalyses::ReconstructedParticle::get_p(z_rp)')
    df = df.Define('z_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(z_rp)')
    df = df.Define('z_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(z_rp)')

    df = df.Define('l1_e',     'FCCAnalyses::ReconstructedParticle::get_e(l1)')
    df = df.Define('l1_p',     'FCCAnalyses::ReconstructedParticle::get_p(l1)')
    df = df.Define('l1_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(l1)')
    df = df.Define('l1_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(l1)')

    df = df.Define('l2_e',     'FCCAnalyses::ReconstructedParticle::get_e(l2)')
    df = df.Define('l2_p',     'FCCAnalyses::ReconstructedParticle::get_p(l2)')
    df = df.Define('l2_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(l2)')
    df = df.Define('l2_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(l2)')

    df = df.Define('mass',   'FCCAnalyses::getAllPairsZMass(all_pairs)')
    df = df.Define('recoil', 'FCCAnalyses::getAllPairsRecoil(all_pairs)')

    df = df.Define('l1_idx', 'FCCAnalyses::getAllPairsLegIdx1(all_pairs)')
    df = df.Define('l2_idx', 'FCCAnalyses::getAllPairsLegIdx2(all_pairs)')
    df = df.Define('mc1',    'FCCAnalyses::getAllPairsMCIdx1(all_pairs)')
    df = df.Define('mc2',    'FCCAnalyses::getAllPairsMCIdx2(all_pairs)')

    # Extract serializable fields from trueZ
    df = df.Define('trueZ_rp', 'FCCAnalyses::getTrueZRP(trueZ)')
    df = df.Define('lep1',     'FCCAnalyses::getTrueZLepton1(trueZ)')
    df = df.Define('lep2',     'FCCAnalyses::getTrueZLepton2(trueZ)')
    df = df.Define('MC1',      'FCCAnalyses::getTrueZMCIdx1(trueZ)')
    df = df.Define('MC2',      'FCCAnalyses::getTrueZMCIdx2(trueZ)')

    df = df.Define('Z_e',     'FCCAnalyses::ReconstructedParticle::get_e(trueZ_rp)[0]')
    df = df.Define('Z_p',     'FCCAnalyses::ReconstructedParticle::get_p(trueZ_rp)[0]')
    df = df.Define('Z_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(trueZ_rp)[0]')
    df = df.Define('Z_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(trueZ_rp)[0]')

    df = df.Define('Mass',      'FCCAnalyses::ReconstructedParticle::get_mass(trueZ_rp)[0]')
    df = df.Define('z_recoil', f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(trueZ_rp)')
    df = df.Define('Recoil',    'FCCAnalyses::ReconstructedParticle::get_mass(z_recoil)[0]')

    df = df.Define('lep1_e',     'FCCAnalyses::ReconstructedParticle::get_e(lep1)[0]')
    df = df.Define('lep1_p',     'FCCAnalyses::ReconstructedParticle::get_p(lep1)[0]')
    df = df.Define('lep1_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(lep1)[0]')
    df = df.Define('lep1_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(lep1)[0]')

    df = df.Define('lep2_e',     'FCCAnalyses::ReconstructedParticle::get_e(lep2)[0]')
    df = df.Define('lep2_p',     'FCCAnalyses::ReconstructedParticle::get_p(lep2)[0]')
    df = df.Define('lep2_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(lep2)[0]')
    df = df.Define('lep2_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(lep2)[0]')

    return df


def optimize_qq(df: 'ROOT.ROOT.RDataFrame',
                cat: str,
                ecm: int,
                dataset: str) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)

    if 'p8_ee_WW_ecm' in dataset:  # remove muons/electrons from inclusive WW
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    df = df.Define('muons_all',     'FCCAnalyses::ReconstructedParticle::get(Muon0,     ReconstructedParticles)')
    df = df.Define('photons_all',   'FCCAnalyses::ReconstructedParticle::get(Photon0,   ReconstructedParticles)')
    df = df.Define('electrons_all', 'FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)')

    df = veto_leptonic(df, ecm, 'mumu')
    df = veto_leptonic(df, ecm, 'ee')

    df = make_pseudojets(df, ecm)

    for i in [0, 2, 4, 6]:
        df = clustering(df, i)

    return df


reco_mc = ['MCRecoAssociations#0', 'MCRecoAssociations#1', 'ReconstructedParticles', 'Particle', 'Particle#0', 'Particle#1']
branch_list_ll = [
    'leps', 'mass', 'recoil', 'Mass', 'Recoil',
    'l1_idx', 'l2_idx', 'mc1', 'mc2', 'MC1', 'MC2',
    'z_e', 'z_p', 'z_pt', 'z_theta', 'Z_e', 'Z_p', 'Z_pt', 'Z_theta',
    'l1_e', 'l1_p', 'l1_pt', 'l1_theta', 'l2_e', 'l2_p', 'l2_pt', 'l2_theta',
    'lep1_e', 'lep1_p', 'lep1_pt', 'lep1_theta', 'lep2_e', 'lep2_p', 'lep2_pt', 'lep2_theta',
]
branch_list_qq = ['jets_rp_cand_N0', 'jets_rp_cand_N2', 'jets_rp_cand_N4', 'jets_rp_cand_N6',
                  'jets_N0', 'jets_N2', 'jets_N4', 'jets_N6',
                  'clustered_jets_N0', 'clustered_jets_N2', 'clustered_jets_N4', 'clustered_jets_N6',
                  'rps_sel', 'pseudo_jets'] + reco_mc
