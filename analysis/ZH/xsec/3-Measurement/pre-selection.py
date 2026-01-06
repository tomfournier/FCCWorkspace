###########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULES ###
###########################################################

# Import user configuration paths and parameters
from package.userConfig import (
    loc, get_loc, 
    ecm, frac, 
    nb, ww
)
from package.config import z_decays, H_decays
# Select Z decay channel from user input
cat = input('Select channel [ee, mumu]: ')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
outputDir = get_loc(loc.EVENTS, cat, ecm, '')

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023/IDEA/'

# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_IDEA.json'

# Optional: Number of CPUs for parallel processing 
# (default is 4,  -1 uses all cores available)
nCPUS = 20

# Run on HTCondor batch system (default is False)
runBatch = False

# Batch queue name for HTCondor (default is workday)
batchQueue = 'longlunch'

# Computing account for HTCondor (default is group_u_FCC.local_gen)
compGroup = 'group_u_FCC.local_gen'



################################
### SETUP SAMPLES TO PROCESS ###
################################

# Background samples:
samples_bkg = [
    # Diboson:  ee -> VV
    f'p8_ee_ZZ_ecm{ecm}',
    f'p8_ee_WW_ee_ecm{ecm}', 
    f'p8_ee_WW_mumu_ecm{ecm}',

    # ee -> Z+jets
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}', 
    f'wzp6_ee_mumu_ecm{ecm}', 
    f'wzp6_ee_tautau_ecm{ecm}',

    # Radiative: ey -> eZ(ll)
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',

    # Diphoton: yy -> ll
    f'wzp6_gaga_ee_60_ecm{ecm}', 
    f'wzp6_gaga_mumu_60_ecm{ecm}', 
    f'wzp6_gaga_tautau_60_ecm{ecm}', 
    
    # Invisible: ee -> nunuZ
    f'wzp6_ee_nuenueZ_ecm{ecm}'
]

# Signal samples: ee -> Z(ll)H with all Higgs decay modes
samples_sig = [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in H_decays + ('ZZ_noInv',)]
samples_sig.extend([f'wzp6_ee_eeH_ecm{ecm}', f'wzp6_ee_mumuH_ecm{ecm}', f'wzp6_ee_ZH_Hinv_ecm{ecm}'])

# Combine all samples (override with WW only if ww flag is set)
samples = samples_sig + samples_bkg
if ww: samples = [f'p8_ee_WW_ecm{ecm}']

# Large samples requiring chunked processing
big_sample = [
    f'p8_ee_ZZ_ecm{ecm}', 
    f'p8_ee_WW_ecm{ecm}', 
    f'p8_ee_WW_{cat}_ecm{ecm}',

    f'wzp6_ee_mumu_ecm{ecm}' if cat=='mumu' else f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',

    f'wzp6_egamma_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gammae_eZ_Z{cat}_ecm{ecm}',
    f'wzp6_gaga_{cat}_60_ecm{ecm}'
]
# Configure processing fraction and chunks for each sample
processList = {i:{'fraction': frac, 'chunks': nb if i in big_sample else 1}  for i in samples}



###################################################
### CUSTOM FUNCTION TO USE IN THE PRE-SELECTION ###
###################################################

def cutflow(df, cut: str):
    '''Record event count at each cutflow stage.'''
    n  = df.Count().GetValue()
    df = df.Define(cut, str(n))
    return df



##########################
### SELECTIOM FUNCTION ###
##########################

def build_graph_ll(df, cat):
    
    # Alias for lepton collections based on final state
    if cat == 'mumu':
        df = df.Alias('Lepton0', 'Muon#0.index')
    elif cat == 'ee':
        df = df.Alias('Lepton0', 'Electron#0.index')
    else:
        raise ValueError(f'cat {cat} not supported')
    
    # Alias for MC truth matching and particle collections
    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')
    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('Photon0', 'Photon#0.index')

    # Filter out leptonic WW events if WW flag is set
    if ww:
        df = df.Define('ww_leptonic', 'FCCAnalyses::is_ww_leptonic(Particle, Particle1)')
        df = df.Filter('!ww_leptonic')

    # Compute missing energy and missing mass variables
    df = df.Define('missingEnergy', f'FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)')
    df = df.Define('cosTheta_miss', 'FCCAnalyses::get_cosTheta_miss(missingEnergy)')
    df = df.Define('missingMass',   f'FCCAnalyses::missingMass({ecm}, ReconstructedParticles)')

    # Define all lepton properties (before cuts)
    df = df.Define('leps_all',       'FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)')
    df = df.Define('leps_all_p',     'FCCAnalyses::ReconstructedParticle::get_p(leps_all)')
    df = df.Define('leps_all_theta', 'FCCAnalyses::ReconstructedParticle::get_theta(leps_all)')
    df = df.Define('leps_all_phi',   'FCCAnalyses::ReconstructedParticle::get_phi(leps_all)')
    df = df.Define('leps_all_q',     'FCCAnalyses::ReconstructedParticle::get_charge(leps_all)')
    df = df.Define('leps_all_no',    'FCCAnalyses::ReconstructedParticle::get_n(leps_all)')
    df = df.Define('leps_all_iso',   'FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)') 
    
    # Apply momentum cut (p > 20 GeV) and isolation requirements
    df = df.Define('leps',            'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')
    df = df.Define('leps_p',          'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('leps_theta',      'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('leps_phi',        'FCCAnalyses::ReconstructedParticle::get_phi(leps)')
    df = df.Define('leps_q',          'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',         'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',        'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('leps_sel_iso',    'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    df = df.Define('leps_sel_iso_no', 'leps_sel_iso.size()')

    ##########
    ### CUT 0: all events
    ##########
    df = cutflow(df, 'cut0')

    ##########
    ### CUT 1: at least one lepton and at least one isolated lepton (I_rel < 0.25)
    ##########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0', 'cut1')
    df = cutflow(df, 'cut1')

    ##########
    ### CUT 2: at least 2 leptons with opposite-sign
    ##########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()', 'cut2')
    df = cutflow(df, 'cut2')

    # Veto H->mumu/ee candidate leptons (mass window: 125 ± 3 GeV)
    df = df.Define('zbuilder_Hll',   f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)'
                   '(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll_Hll',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[0]}') # the Z
    df = df.Define('zll_Hll_m',      'FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]')
    df = df.Define('zll_leps_Hll',   'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[1], zbuilder_Hll[2]}') # the leptons
    df = df.Define('zll_leps_dummy', 'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}') # the leptons
    df = df.Define('leps_to_remove', 'return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy')
    df = df.Define('leps_good',      'FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)')

    # Build Z resonance from good leptons (mass ~ 91.2 GeV, recoil ~ 125 GeV)
    # Returns the best lepton pair compatible with Z mass and recoil at 125 GeV
    # Returns: [0] di-lepton system, [1,2] individual leptons
    df = df.Define('zbuilder_result', f'FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)'
                   '(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}') # the Z
    df = df.Define('zll_leps',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}') # the leptons
    
    # Z boson kinematics
    df = df.Define('zll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]')
    df = df.Define('zll_p',           'FCCAnalyses::ReconstructedParticle::get_p(zll)[0]')
    df = df.Define('zll_theta',       'FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]')
    df = df.Define('zll_phi',         'FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]')

    # Recoil mass (Higgs candidate)
    df = df.Define('zll_recoil',   f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)')
    df = df.Define('zll_recoil_m', 'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]')
    df = df.Define('zll_category', 'FCCAnalyses::polarAngleCategorization(0.8, 2.34)(zll_leps)')

    # Individual Z lepton kinematics with leading/subleading ordering by momentum
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
    
    # Angular correlation variables
    df = df.Define('acoplanarity', 'FCCAnalyses::acoplanarity(zll_leps)')
    df = df.Define('acolinearity', 'FCCAnalyses::acolinearity(zll_leps)')
    df = df.Define('deltaR',       'FCCAnalyses::deltaR(zll_leps)')

    # Higgsstrahlungness: discriminant based on Z and recoil masses
    df = df.Define('H', 'FCCAnalyses::Higgsstrahlungness(zll_m, zll_recoil_m)')

    # Compute visible energy (excluding Z leptons)
    df = df.Define('rps_no_leps',   'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)')
    df = df.Define('visibleEnergy', 'FCCAnalyses::visibleEnergy(rps_no_leps)')

    ##########
    ### CUT 3: Z mass window
    ##########
    # df = df.Filter('zll_m > 86 && zll_m < 96')
    # df = cutflow(df, 'cut3')

    ##########
    ### CUT 4: Z momentum (CoM dependent)
    ##########
    # if   ecm == 240: df = df.Filter('zll_p > 20 && zll_p < 70')
    # elif ecm == 365: df = df.Filter('zll_p > 50 && zll_p < 150')
    # df = cutflow(df, 'cut4')

    return df



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    '''RDataFrame analysis class for pre-selection stage.'''

    #_________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        '''Apply analysis graph construction to the dataframe.'''
        df = build_graph_ll(df, cat)
        return df
    
    #_____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        '''Define output branches to save.'''
        branchList = [
            # Lepton kinematics (leading and subleading)
            'leading_p',    'leading_theta',    'leading_phi',
            'subleading_p', 'subleading_theta', 'subleading_phi',

            # Angular correlation
            'acolinearity', 'acoplanarity', 'deltaR',

            # Z boson kinematics
            'zll_m', 'zll_p', 'zll_theta', 'zll_phi',

            # Recoil mass (Higgs candidate)
            'zll_recoil_m',

            # Missing energy variables
            'visibleEnergy', 'cosTheta_miss', 'missingMass',

            # Higgsstrahlungness discriminant
            'H',
            
            # Cutflow counters
            'cut0', 'cut1', 'cut2',
            # 'cut3', 'cut4'
        ]
        return branchList
