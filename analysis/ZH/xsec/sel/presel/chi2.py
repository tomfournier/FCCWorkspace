from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT

from .helper import setup_alias
from .hadronic import veto_leptonic, make_pseudojets



#################################
### LEPTONIC HELPER FUNCTIONS ###
#################################

def get_leps(df: 'ROOT.ROOT.RDataFrame',
             ecm: int
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
    df = df.Define('leps_all',     'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('leps_all_iso', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)')

    # Recover the leptons' energy lost from FSR
    if ecm == 240:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps_all, Photon0, ReconstructedParticles, leps_all_iso, 1, 3.08, 1.80)')
    elif ecm == 365:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps_all, Photon0, ReconstructedParticles, leps_all_iso, 0, 0.25, 0.95)')
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    # Apply momentum cut (p > 20 GeV) to reduce soft backgrounds
    df = df.Define('leps', 'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_fsr)')

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
    if njets == 0:  # inclusive
        df = df.Define('clustered_jets_N0', 'JetClustering::clustering_kt(0.6, 0, 5, 1, 0)(pseudo_jets)')
    else:
        df = df.Define(f'clustered_jets_N{njets}', f'JetClustering::clustering_ee_kt(2, {njets}, 0, 10)(pseudo_jets)')

    df = df.Define(f'jets_N{njets}',         f'FCCAnalyses::JetClusteringUtils::get_pseudoJets(clustered_jets_N{njets})')
    df = df.Define(f'jets_rp_N{njets}',      f'FCCAnalyses::jets2rp(jets_N{njets})')
    df = df.Define(f'jets_rp_cand_N{njets}', f'FCCAnalyses::select_jets(jets_rp_N{njets}, {njets}, ReconstructedParticles)')  # reduces potentially the jet multiplicity
    return df



######################
### MAIN FUNCTIONS ###
######################

def fsr_recovery(
        df: 'ROOT.ROOT.RDataFrame',
        cat: str,
        ecm: int
         ) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)

    # Get the lepton and photons rps
    df = df.Define('leps',    'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('photons', 'FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)')
    if ecm == 240:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps, Photon0, ReconstructedParticles, leps_all_iso, 1, 3.08, 1.80)')
    elif ecm == 365:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps, Photon0, ReconstructedParticles, leps_all_iso, 0, 0.25, 0.95)')
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    # Get the true lepton and photon information
    df = df.Define('Leps_MC',    'FCCAnalyses::fromRP2MC(leps, MCRecoAssociations1, Particle)')
    df = df.Define('photons_MC', 'FCCAnalyses::fromRP2MC(Photon0, MCRecoAssociations1, Particle)')
    df = df.Define('leps_MC',    'FCCAnalyses::getParent(Leps_MC, Particle, Particle0)')  # Before FSR



    ########################################
    ### LEPTON-PHOTON ORIGIN DEFINITIONS ###
    ########################################

    # Get the origin of the leptons and photons
    df = df.Define('lepton_origin', 'FCCAnalyses::getLeptonOrigin(Leps_MC, Particle, Particle0, false)')
    df = df.Define('photon_origin', 'FCCAnalyses::getPhotonOrigin(photons_MC, Particle, Particle0)')

    # Determine if the photons come from ISR or FSR
    df = df.Define('fromISR', 'FCCAnalyses::fromISR(photon_origin)')
    df = df.Define('fromFSR', 'FCCAnalyses::fromFSR(photon_origin)')

    # Get the leptons and photons parent for merging comparison
    # Extract parent indices from MC particles directly
    df = df.Define('lep_parent', 'FCCAnalyses::getParentID(Leps_MC, Particle, Particle0)')
    df = df.Define('ph_parent',  'FCCAnalyses::getParentID(photons_MC, Particle, Particle0)')

    # Get the table of true association
    df = df.Define('n_radiated',  'FCCAnalyses::nRadiated(lep_parent, ph_parent)')
    df = df.Define('same_parent', 'FCCAnalyses::fromSameParent(lep_parent, ph_parent)')



    ############################################
    ### LEPTON-PHOTON KINEMATICS DEFINITIONS ###
    ############################################

    # Get the leptons isolation (Reco and MC)
    df = df.Define('leps_iso', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('LEPS_iso', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, MCRecoAssociations0, MCRecoAssociations1, Particle)')

    # Get the leptons and photons momentum (Reco and MC)
    df = df.Define('leps_p', 'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('ph_p',   'FCCAnalyses::ReconstructedParticle::get_p(photons)')
    df = df.Define('LEPS_p', 'FCCAnalyses::MCParticle::get_p(leps_MC)')
    df = df.Define('PH_p',   'FCCAnalyses::MCParticle::get_p(photons_MC)')
    df = df.Define('fsr_p',  'FCCAnalyses::ReconstructedParticle::get_p(leps_fsr)')

    # Get the leptons and photons transverse momentum (Reco and MC)
    df = df.Define('leps_pT', 'FCCAnalyses::ReconstructedParticle::get_pt(leps)')
    df = df.Define('ph_pT',   'FCCAnalyses::ReconstructedParticle::get_pt(photons)')
    df = df.Define('LEPS_pT', 'FCCAnalyses::MCParticle::get_pt(leps_MC)')
    df = df.Define('PH_pT',   'FCCAnalyses::MCParticle::get_pt(photons_MC)')
    df = df.Define('fsr_pT',  'FCCAnalyses::ReconstructedParticle::get_pt(leps_fsr)')

    # Get the leptons and photons polar angle (Reco and MC)
    df = df.Define('leps_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('ph_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(photons)')
    df = df.Define('LEPS_theta', 'FCCAnalyses::MCParticle::get_theta(leps_MC)')
    df = df.Define('PH_theta',   'FCCAnalyses::MCParticle::get_theta(photons_MC)')
    df = df.Define('fsr_theta',  'FCCAnalyses::ReconstructedParticle::get_theta(leps_fsr)')

    # Get the lepton isolation for each lepton pair
    df = df.Define('leps_iso_pair', 'FCCAnalyses::lepGaPair(leps_iso, ph_p, true)')
    df = df.Define('LEPS_iso_pair', 'FCCAnalyses::lepGaPair(LEPS_iso, PH_p, true)')



    #######################################
    ### ANGULAR CORRELATION DEFINITIONS ###
    #######################################

    # Convert rps to lorentz vector for angular correlation computations
    df = df.Define('leps_tlv',    'FCCAnalyses::makeLorentzVectors(leps)')
    df = df.Define('photons_tlv', 'FCCAnalyses::makeLorentzVectors(photons)')
    # Get the angular correlation between the leptons and photons
    df = df.Define('cosTheta',     'FCCAnalyses::getCosTheta(leps_tlv,     photons_tlv)')
    df = df.Define('acolinearity', 'FCCAnalyses::getAcolinearity(leps_tlv, photons_tlv)')
    df = df.Define('acoplanarity', 'FCCAnalyses::getAcoplanarity(leps_tlv, photons_tlv)')
    df = df.Define('acopolarity',  'FCCAnalyses::getAcopolarity(leps_tlv,  photons_tlv)')
    df = df.Define('deltaR',       'FCCAnalyses::getDeltaR(leps_tlv,       photons_tlv)')


    # Convert mc to lorentz vector for angular correlation computations
    df = df.Define('leps_MC_tlv',    'FCCAnalyses::MCParticle::get_tlv(leps_MC)')
    df = df.Define('photons_MC_tlv', 'FCCAnalyses::MCParticle::get_tlv(photons_MC)')
    # Get the MC angular correlation between the leptons and photons
    df = df.Define('CosTheta',     'FCCAnalyses::getCosTheta(leps_MC_tlv,     photons_MC_tlv)')
    df = df.Define('Acolinearity', 'FCCAnalyses::getAcolinearity(leps_MC_tlv, photons_MC_tlv)')
    df = df.Define('Acoplanarity', 'FCCAnalyses::getAcoplanarity(leps_MC_tlv, photons_MC_tlv)')
    df = df.Define('Acopolarity',  'FCCAnalyses::getAcopolarity(leps_MC_tlv,  photons_MC_tlv)')
    df = df.Define('DeltaR',       'FCCAnalyses::getDeltaR(leps_MC_tlv,       photons_MC_tlv)')

    return df


def optimize_ll(df: 'ROOT.ROOT.RDataFrame',
                cat: str,
                ecm: int,
                ) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)
    df = get_leps(df, ecm)

    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps.size()')

    df = df.Define('all_pairs', f'FCCAnalyses::leptonicZBuilder({ecm}, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)')
    df = df.Define('n_pair',    'all_pairs.size()')
    df = df.Define('trueZ',     f'FCCAnalyses::getTrueZ("{cat}", Particle, Particle0, Particle1)')

    df = df.Filter('trueZ.z_system.mass >= 0')

    # Extract serializable fields from all_pairs
    df = df.Define('z_rp', 'FCCAnalyses::getZPairsZSystem(all_pairs)')
    df = df.Define('l1',   'FCCAnalyses::getZPairsLepton1(all_pairs)')
    df = df.Define('l2',   'FCCAnalyses::getZPairsLepton2(all_pairs)')

    df = df.Define('zll_p',     'FCCAnalyses::ReconstructedParticle::get_p(z_rp)')
    df = df.Define('zll_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(z_rp)')
    df = df.Define('zll_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(z_rp)')

    df = df.Define('leading_p',     'FCCAnalyses::ReconstructedParticle::get_p(l1)')
    df = df.Define('leading_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(l1)')
    df = df.Define('leading_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(l1)')

    df = df.Define('subleading_p',     'FCCAnalyses::ReconstructedParticle::get_p(l2)')
    df = df.Define('subleading_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(l2)')
    df = df.Define('subleading_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(l2)')

    df = df.Define('mass',   'FCCAnalyses::getAllPairsZMass(all_pairs)')
    df = df.Define('recoil', 'FCCAnalyses::getAllPairsRecoil(all_pairs)')

    df = df.Define('leading_mc',    'FCCAnalyses::getAllPairsMCIdx1(all_pairs)')
    df = df.Define('subleading_mc', 'FCCAnalyses::getAllPairsMCIdx2(all_pairs)')

    # Extract serializable fields from trueZ
    df = df.Define('trueZ_rp',      'FCCAnalyses::getTrueZRP(trueZ)')
    df = df.Define('lep1',          'FCCAnalyses::getTrueZLepton1(trueZ)')
    df = df.Define('lep2',          'FCCAnalyses::getTrueZLepton2(trueZ)')
    df = df.Define('Leading_MC',    'FCCAnalyses::getTrueZMCIdx1(trueZ)')
    df = df.Define('Subleading_MC', 'FCCAnalyses::getTrueZMCIdx2(trueZ)')

    df = df.Define('Zll_p',     'FCCAnalyses::ReconstructedParticle::get_p(trueZ_rp)[0]')
    df = df.Define('Zll_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(trueZ_rp)[0]')
    df = df.Define('Zll_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(trueZ_rp)[0]')

    df = df.Define('Mass',        'FCCAnalyses::ReconstructedParticle::get_mass(trueZ_rp)[0]')
    df = df.Define('zll_recoil', f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(trueZ_rp)')
    df = df.Define('Recoil',      'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]')

    df = df.Define('Leading_p',     'FCCAnalyses::ReconstructedParticle::get_p(lep1)[0]')
    df = df.Define('Leading_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(lep1)[0]')
    df = df.Define('Leading_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(lep1)[0]')

    df = df.Define('Subleading_p',     'FCCAnalyses::ReconstructedParticle::get_p(lep2)[0]')
    df = df.Define('Subleading_pt',    'FCCAnalyses::ReconstructedParticle::get_pt(lep2)[0]')
    df = df.Define('Subleading_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(lep2)[0]')

    return df


def optimize_qq(df: 'ROOT.ROOT.RDataFrame',
                cat: str,
                ecm: int,
                ) -> 'ROOT.ROOT.RDataFrame':

    df = setup_alias(df, cat)

    df = df.Define('muons_all',     'FCCAnalyses::ReconstructedParticle::get(Muon0,     ReconstructedParticles)')
    df = df.Define('photons_all',   'FCCAnalyses::ReconstructedParticle::get(Photon0,   ReconstructedParticles)')
    df = df.Define('electrons_all', 'FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)')

    df = veto_leptonic(df, ecm, 'mumu')
    df = veto_leptonic(df, ecm, 'ee')

    df = make_pseudojets(df, ecm)

    for i in [0, 2, 4, 6]:
        df = clustering(df, i)

    return df


#######################
### OUTPUT BRANCHES ###
#######################

branch_list_ll = [
    'n_pair',             # Number of reconstructed pair
    'mass', 'recoil',     # Reco Z mass and recoil mass
    'Mass', 'Recoil',     # True Z mass and recoil mass
    'leading_mc', 'subleading_mc',      # Reco MC idx for leading and subleading lepton
    'Leading_MC', 'Subleading_MC',      # True MC idx for leading and subleading lepton
    'zll_p', 'zll_pt', 'zll_theta',     # Reco Z kinematics
    'Zll_p', 'Zll_pt', 'Zll_theta',     # True Z kinematics
    'leading_p', 'leading_pt', 'leading_theta',              # Reco leading lepton kinematics
    'Leading_p', 'Leading_pt', 'Leading_theta',              # True leading lepton kinematics
    'subleading_p', 'subleading_pt', 'subleading_theta',     # Reco subleading lepton kinematics
    'Subleading_p', 'Subleading_pt', 'Subleading_theta',     # True subleading lepton kinematics
]
branch_list_qq = [
    'jets_rp_cand_N0', 'jets_rp_cand_N2', 'jets_rp_cand_N4', 'jets_rp_cand_N6',
    'jets_N0', 'jets_N2', 'jets_N4', 'jets_N6',
    'clustered_jets_N0', 'clustered_jets_N2', 'clustered_jets_N4', 'clustered_jets_N6',
    'rps_sel', 'pseudo_jets'
]
branch_list_fsr = [
    'lep_parent', 'lepton_origin',          # Lepton origin (signal, tau, Z, W, H, jet)
    'ph_parent',  'photon_origin',          # Photon origin (lep, H, jet, etc.)
    'fromISR', 'fromFSR', 'same_parent',    # Photon origin (ISR, FSR) and check if pair comes from same parent
    'leps_iso', 'leps_iso_pair',            # Reco lepton isolation
    'LEPS_iso', 'LEPS_iso_pair',            # True lepton isolation
    'leps_p', 'leps_pT', 'leps_theta', 'ph_p', 'ph_pT', 'ph_theta',         # Reco leptons and photons kinematics
    'LEPS_p', 'LEPS_pT', 'LEPS_theta', 'PH_p', 'PH_pT', 'PH_theta',         # True leptons and photons kinematics
    'fsr_p',  'fsr_pT',  'fsr_theta',                                       # Leptons kinematics after FSR recovery
    'cosTheta', 'acolinearity', 'acoplanarity', 'acopolarity', 'deltaR',    # Reco angular correlation between for all the lepton-photon pair
    'CosTheta', 'Acolinearity', 'Acoplanarity', 'Acopolarity', 'DeltaR'     # True angular correlation between for all the lepton-photon pair
    'n_radiated',    # Number of radiated photon for each lepton
]
