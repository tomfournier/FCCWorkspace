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
    df = df.Define(f'njets_N{njets}',           f'jets_N{njets}.size()')
    df = df.Define(f'jetconstituents_N{njets}', f'FCCAnalyses::JetClusteringUtils::get_constituents(clustered_jets_N{njets})')
    df = df.Define(f'jets_e_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_e(jets_N{njets})')
    df = df.Define(f'jets_px_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_px(jets_N{njets})')
    df = df.Define(f'jets_py_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_py(jets_N{njets})')
    df = df.Define(f'jets_pz_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_pz(jets_N{njets})')
    df = df.Define(f'jets_m_N{njets}',          f'FCCAnalyses::JetClusteringUtils::get_m(jets_N{njets})')
    df = df.Define(f'jets_rp_N{njets}',         f'FCCAnalyses::jets2rp(jets_px_N{njets}, jets_py_N{njets}, jets_pz_N{njets}, jets_e_N{njets}, jets_m_N{njets})')
    df = df.Define(f'jets_rp_cand_N{njets}',    f'FCCAnalyses::select_jets(jets_rp_N{njets}, jetconstituents_N{njets}, {njets}, ReconstructedParticles)')  # reduces potentially the jet multiplicity
    return df



######################
### MAIN FUNCTIONS ###
######################

def optimize_ll(df: 'ROOT.ROOT.RDataFrame',
                cat: str,
                dataset: str) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)

    if 'p8_ee_WW_ecm' in dataset:  # remove muons/electrons from inclusive WW
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    df = get_leps(df)

    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps.size()')

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


branch_list_ll = ['leps', 'MCRecoAssociations0', 'MCRecoAssociations1', 'ReconstructedParticles', 'Particle', 'Particles0', 'Particle1']
branch_list_qq = ['jets_rp_cand_N0', 'jets_rp_cand_N2', 'jets_rp_cand_N4', 'jets_rp_cand_N6']
