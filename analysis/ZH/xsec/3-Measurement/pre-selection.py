import importlib

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, ecm, frac, nb, ww

from package.config import z_decays, h_decays

cat = input('Select channel [ee, mumu]: ')

# Output directory where the files produced at the pre-selection level will be put
outputDir = get_loc(loc.EVENTS, cat, ecm, '')

# include custom functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = 'FCCee/winter2023/IDEA/'

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# optional: ncpus, default is 4, 
# -1 uses all cores available
nCPUS = 20

# Process samples
samples_bkg = [
    f'p8_ee_ZZ_ecm{ecm}',
    f'wzp6_ee_WW_ecm{ecm}',            f'p8_ee_WW_ee_ecm{ecm}',       f'p8_ee_WW_mumu_ecm{ecm}',
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',      f'wzp6_ee_tautau_ecm{ecm}',
    f'wzp6_gaga_ee_60_ecm{ecm}',       f'wzp6_gaga_mumu_60_ecm{ecm}', f'wzp6_gaga_tautau_60_ecm{ecm}', 
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',  f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays + ['ZZ_noInv']]
samples_sig.extend([f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}', f'wzp6_ee_ZH_Hinv_ecm{ecm}'])

samples = samples_sig + samples_bkg
if ww: samples = f'p8_ee_WW_ecm{ecm}'

big_sample = [
    f'p8_ee_ZZ_ecm{ecm}', f'p8_ee_WW_ecm{ecm}', f'p8_ee_WW_{cat}_ecm{ecm}',
    f'wzp6_ee_mumu_ecm{ecm}' if cat=='mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}', f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]
processList = {i:{'fraction': frac, 'chunks': nb if i in big_sample else 1}  for i in samples}



def cutflow(df, cut: str):
    n  = df.Count().GetValue()
    df = df.Define(cut, str(n))
    return df

def build_graph_ll(df, cat):

    ################################################
    ### Alias for muon and MC truth informations ###
    ################################################
    
    if cat == 'mumu':
        df = df.Alias('Lepton0', 'Muon#0.index')
    elif cat == 'ee':
        df = df.Alias('Lepton0', 'Electron#0.index')
    else:
        raise ValueError(f'cat {cat} not supported')
    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')
    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('Photon0', 'Photon#0.index')

    if ww:
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df.Filter('!ww_leptonic')
     
    # Missing ET
    df = df.Define('missingEnergy', f'FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)')
    df = df.Define('cosTheta_miss', 'FCCAnalyses::get_cosTheta_miss(missingEnergy)')
    df = df.Define('missingMass',   f'FCCAnalyses::missingMass({ecm}, ReconstructedParticles)')

    # all leptons (bare)
    df = df.Define('leps_all',       'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('leps_all_p',     'FCCAnalyses::ReconstructedParticle::get_p(leps_all)')
    df = df.Define('leps_all_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(leps_all)')
    df = df.Define('leps_all_phi',   'FCCAnalyses::ReconstructedParticle::get_phi(leps_all)')
    df = df.Define('leps_all_q',     'FCCAnalyses::ReconstructedParticle::get_charge(leps_all)')
    df = df.Define('leps_all_no',    'FCCAnalyses::ReconstructedParticle::get_n(leps_all)')
    df = df.Define('leps_all_iso',   'FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)') 
    
    # cuts on leptons
    df = df.Define('leps',            'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')
    df = df.Define('leps_p',          'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('leps_theta',      'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('leps_phi',        'FCCAnalyses::ReconstructedParticle::get_phi(leps)')
    df = df.Define('leps_q',          'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',         'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',        'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('leps_sel_iso',    'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    df = df.Define('leps_sel_iso_no', 'leps_sel_iso.size()')

    #########
    ### CUT 0 all events
    #########
    # df = df.Define('cut0', '0')
    df = cutflow(df, 'cut0')

    #########
    ### CUT 1: at least a lepton and at least 1 one lepton isolated (I_rel < 0.25)
    #########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0', 'cut1')
    # df = df.Define('cut1', '1')
    df = cutflow(df, 'cut1')

    #########
    ### CUT 2 :at least 2 OS leptons, and build the resonance
    #########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()', 'cut2')
    # df = df.Define('cut2', '2')
    df = cutflow(df, 'cut2')

    # remove H->mumu/ee candidate leptons
    df = df.Define('zbuilder_Hll',   f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)'
                   '(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll_Hll',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[0]}') # the Z
    df = df.Define('zll_Hll_m',      'FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]')
    df = df.Define('zll_leps_Hll',   'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[1], zbuilder_Hll[2]}') # the leptons
    df = df.Define('zll_leps_dummy', 'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}') # the leptons
    df = df.Define('leps_to_remove', 'return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy')
    df = df.Define('leps_good',      'FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)')

    # build the Z resonance based on the available leptons. 
    # Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
    # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, 
    # index 1 and 2 the leptons of the pair
    df = df.Define('zbuilder_result', f'FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)'
                   '(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}') # the Z
    df = df.Define('zll_leps',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}') # the leptons
    df = df.Define('zll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]')
    df = df.Define('zll_p',           'FCCAnalyses::ReconstructedParticle::get_p(zll)[0]')
    df = df.Define('zll_theta',       'FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]')
    df = df.Define('zll_phi',         'FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]')
    df = df.Define('zll_deltaR',      'FCCAnalyses::deltaR(zll_leps)')

    # recoil
    df = df.Define('zll_recoil',   f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)')
    df = df.Define('zll_recoil_m', 'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]')
    df = df.Define('zll_category', 'FCCAnalyses::polarAngleCategorization(0.8, 2.34)(zll_leps)')

    # Z leptons informations
    df = df.Define('zll_leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(zll_leps)')
    df = df.Define('zll_leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)')
    df = df.Define('zll_leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)')
    df = df.Define('zll_leps_dR',      'FCCAnalyses::deltaR(zll_leps)')
    df = df.Define('leading_p_idx',    '(zll_leps_p[0] > zll_leps_p[1]) ? 0 : 1')
    df = df.Define('subleading_p_idx', '(zll_leps_p[0] > zll_leps_p[1]) ? 1 : 0')
    df = df.Define('leading_p',        'zll_leps_p[leading_p_idx]')
    df = df.Define('subleading_p',     'zll_leps_p[subleading_p_idx]')
    df = df.Define('leading_theta',    'zll_leps_theta[leading_p_idx]')
    df = df.Define('subleading_theta', 'zll_leps_theta[subleading_p_idx]')
    df = df.Define('leading_phi',      'zll_leps_phi[leading_p_idx]')
    df = df.Define('subleading_phi',   'zll_leps_phi[subleading_p_idx]')
    
    df = df.Define('acoplanarity', 'FCCAnalyses::acoplanarity(zll_leps)')
    df = df.Define('acolinearity', 'FCCAnalyses::acolinearity(zll_leps)')

    # Higgsstrahlungness
    df = df.Define('H', 'FCCAnalyses::Higgsstrahlungness(zll_m, zll_recoil_m)')

    # Visible energy
    df = df.Define('rps_no_leps',   'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)')
    df = df.Define('visibleEnergy', 'FCCAnalyses::visibleEnergy(rps_no_leps)')

    #########
    ### CUT 3: Z mass window
    #########
    # df = df.Filter('zll_m > 86 && zll_m < 96')
    # df = df.Define('cut3', '3')
    # df = cutflow(df, 'cut3')

    #########
    ### CUT 4: Z momentum
    #########
    # if   ecm == 240: df = df.Filter('zll_p > 20 && zll_p < 70')
    # elif ecm == 365: df = df.Filter('zll_p > 50 && zll_p < 150')
    # df = df.Define('cut4', '4')
    # df = cutflow(df, 'cut4')

    return df



class RDFanalysis():

    #_________________________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, 
    # please make sure you return the last dataframe, in this example it is df
    def analysers(df):
        df = build_graph_ll(df, cat)
        return df
    
    #________________________________________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            # Reconstructed Particle
            # leptons
            'leading_p',    'leading_theta',    'leading_phi',
            'subleading_p', 'subleading_theta', 'subleading_phi',
            'acolinearity', 'acoplanarity',
            # Zed
            'zll_m', 'zll_p', 'zll_theta', 'zll_phi', 'zll_deltaR',
            # Recoil
            'zll_recoil_m',
            # missing Information
            'visibleEnergy', 'cosTheta_miss', 'missingMass',
            # Higgsstrahlungness
            'H',
            # Cutflow
            'cut0', 'cut1', 'cut2',
            # 'cut3', 'cut4'
        ]
        return branchList
