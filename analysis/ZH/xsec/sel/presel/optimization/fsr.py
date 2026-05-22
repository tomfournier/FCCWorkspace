from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT

from ..helper import setup_alias



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
    df = df.Define('leps',     'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('leps_iso', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('photons',  'FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)')
    if ecm == 240:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps, Photon0, ReconstructedParticles, leps_iso, 1, 3.08, 1.80)')
    elif ecm == 365:
        df = df.Define('leps_fsr', 'FCCAnalyses::recoverFSR(leps, Photon0, ReconstructedParticles, leps_iso, 0, 0.25, 0.95)')
    else:
        raise ValueError(f'{ecm = } not supported, choose between [240, 365]')

    # Get the true lepton and photon information
    df = df.Define('Leps_MC',    'FCCAnalyses::fromRP2MC(leps, ReconstructedParticles, MCRecoAssociations0, MCRecoAssociations1, Particle)')
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

    # Get the leptons isolation (Reco)
    df = df.Define('rps_no_photons', 'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons)')
    df = df.Define('leps_iso_ne',    'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, 0)')
    df = df.Define('leps_iso_ch',    'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, 1)')
    df = df.Define('leps_iso_ph',    'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, rps_no_photons, 2)')
    df = df.Define('leps_iso_PH',    'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, rps_no_photons, 1)')

    # Get the leptons isolation (MC)
    df = df.Define('LEPS_iso',    'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, MCRecoAssociations0, MCRecoAssociations1, Particle)')
    df = df.Define('LEPS_iso_ne', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, MCRecoAssociations0, MCRecoAssociations1, Particle, 0)')
    df = df.Define('LEPS_iso_ch', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles, MCRecoAssociations0, MCRecoAssociations1, Particle, 1)')
    df = df.Define('LEPS_iso_ph', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, rps_no_photons, MCRecoAssociations0, MCRecoAssociations1, Particle, 2)')
    df = df.Define('LEPS_iso_PH', 'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, rps_no_photons, MCRecoAssociations0, MCRecoAssociations1, Particle, 1)')

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
    df = df.Define('leps_iso_pair',    'FCCAnalyses::lepGaPair(leps_iso, ph_p, true)')
    df = df.Define('LEPS_iso_pair',    'FCCAnalyses::lepGaPair(LEPS_iso, PH_p, true)')
    df = df.Define('leps_iso_ch_pair', 'FCCAnalyses::lepGaPair(leps_iso_ch, ph_p, true)')
    df = df.Define('LEPS_iso_ch_pair', 'FCCAnalyses::lepGaPair(LEPS_iso_ch, PH_p, true)')
    df = df.Define('leps_iso_ne_pair', 'FCCAnalyses::lepGaPair(leps_iso_ne, ph_p, true)')
    df = df.Define('LEPS_iso_ne_pair', 'FCCAnalyses::lepGaPair(LEPS_iso_ne, PH_p, true)')
    df = df.Define('leps_iso_ph_pair', 'FCCAnalyses::lepGaPair(leps_iso_ph, ph_p, true)')
    df = df.Define('LEPS_iso_ph_pair', 'FCCAnalyses::lepGaPair(LEPS_iso_ph, PH_p, true)')
    df = df.Define('leps_iso_PH_pair', 'FCCAnalyses::lepGaPair(leps_iso_PH, ph_p, true)')
    df = df.Define('LEPS_iso_PH_pair', 'FCCAnalyses::lepGaPair(LEPS_iso_PH, PH_p, true)')



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


#######################
### OUTPUT BRANCHES ###
#######################

branch_list = [
    'lep_parent', 'lepton_origin',          # Lepton origin (signal, tau, Z, W, H, jet)
    'ph_parent',  'photon_origin',          # Photon origin (lep, H, jet, etc.)
    'fromISR', 'fromFSR', 'same_parent',    # Photon origin (ISR, FSR) and check if pair comes from same parent
    'leps_iso', 'leps_iso_pair',            # Reco lepton isolation
    'LEPS_iso', 'LEPS_iso_pair',            # True lepton isolation

    'leps_iso_ch', 'LEPS_iso_ch', 'leps_iso_ch_pair', 'LEPS_iso_ch_pair',
    'leps_iso_ne', 'LEPS_iso_ne', 'leps_iso_ne_pair', 'LEPS_iso_ne_pair',
    'leps_iso_ph', 'LEPS_iso_ph', 'leps_iso_ph_pair', 'LEPS_iso_ph_pair',
    'leps_iso_PH', 'LEPS_iso_PH', 'leps_iso_PH_pair', 'LEPS_iso_PH_pair',

    'leps_p', 'leps_pT', 'leps_theta', 'ph_p', 'ph_pT', 'ph_theta',         # Reco leptons and photons kinematics
    'LEPS_p', 'LEPS_pT', 'LEPS_theta', 'PH_p', 'PH_pT', 'PH_theta',         # True leptons and photons kinematics
    'fsr_p',  'fsr_pT',  'fsr_theta',                                       # Leptons kinematics after FSR recovery
    'cosTheta', 'acolinearity', 'acoplanarity', 'acopolarity', 'deltaR',    # Reco angular correlation between for all the lepton-photon pair
    'CosTheta', 'Acolinearity', 'Acoplanarity', 'Acopolarity', 'DeltaR',    # True angular correlation between for all the lepton-photon pair
    'n_radiated',    # Number of radiated photon for each lepton
]
