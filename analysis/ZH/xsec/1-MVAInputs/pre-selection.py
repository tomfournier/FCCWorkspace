##########################################################
### IMPORT FUNCTIONS AND PARAMETERS FROM CUSTOM MODULE ###
##########################################################

import os

from package.userConfig import loc, get_params, frac, nb

# Load config from temporary JSON if running automated, else prompt
cat, ecm = get_params(os.environ.copy(), '1-run.json')
if cat not in ['ee', 'mumu']:
    raise ValueError(f'Invalid channel: {cat}. Must be "ee" or "mumu"')



#############################
### SETUP CONFIG SETTINGS ###
#############################

# Output directory for training events (default is local directory)
# outputDir = get_loc(loc.EVENTS_TRAINING, cat, ecm, '')
outputDir = loc.get('EVENTS_TRAINING', cat, ecm)

# Include custom C++ analysis functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag for EDM4Hep centrally produced events
# Points to YAML files for sample statistics
prodTag = 'FCCee/winter2023_training/IDEA/'

# Process dictionary containing cross section information
# Path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

eosType = 'eosuser'

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

# Process samples for BDT training (electron channel)
samples_ee = [
    # Signal: ZH production with H->ee
    f'wzp6_ee_eeH_ecm{ecm}',

    # Main backgrounds: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}', 
    f'p8_ee_WW_ee_ecm{ecm}', 
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    
    # Rare backgrounds: radiative and diphoton
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f'wzp6_gaga_ee_60_ecm{ecm}'
]

# Process samples for BDT training (muon channel)
samples_mumu = [
    # Signal: ZH production with H->mumu
    f'wzp6_ee_mumuH_ecm{ecm}',

    # Background: diboson and Z+jets
    f'p8_ee_ZZ_ecm{ecm}', 
    f'p8_ee_WW_mumu_ecm{ecm}', 
    f'wzp6_ee_mumu_ecm{ecm}',

    # Rare backgrounds: radiative and diphoton
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', 
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gaga_mumu_60_ecm{ecm}'
]

# Select samples based on final state
if   cat=='ee':   samples_BDT = samples_ee
elif cat=='mumu': samples_BDT = samples_mumu
else: raise ValueError(f'cat {cat} not supported')

# Process list with parameters for RDataFrame analysis
processList = {i:{'fraction': frac, 'chunks': nb} for i in samples_BDT}



##########################
### SELECTION FUNCTION ###
##########################

def build_graph_ll(df, cat):
    
    # Alias for lepton collections based on final state
    if   cat == 'mumu':
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
    df = df.Alias('Photon0',   'Photon#0.index')
    
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
    df = df.Define('leps',         'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')
    df = df.Define('leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(leps)')
    df = df.Define('leps_q',       'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',      'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',     'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('leps_sel_iso', 'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    df = df.Define('leps_sel_no',  'leps_sel_iso.size()')

    ##########
    ### CUT 0: all events
    ##########

    ##########
    ### CUT 1: at least one lepton and at least one isolated lepton (I_rel < 0.25)
    ##########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')

    ##########
    ### CUT 2: at least 2 leptons with opposite-sign
    ##########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()')

    # Veto H->mumu/ee candidate leptons (mass window: 125 ± 3 GeV)
    df = df.Define('zbuilder_result_Hll', f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)'
                    '(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll_Hll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}') # the Z
    df = df.Define('zll_Hll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]')
    df = df.Define('zll_leps_Hll',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}') # the leptons
    df = df.Define('zll_leps_dummy',      'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}') # the leptons
    df = df.Define('leps_to_remove',      'return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy')
    df = df.Define('leps_good',           'FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)') 

    # Build Z resonance from good leptons (mass ~ 91.2 GeV, recoil ~ 125 GeV)
    # Returns: [0] di-lepton system, [1,2] individual leptons
    df = df.Define('zbuilder_result', f'FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)'
                    '(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}') # the Z
    df = df.Define('zll_leps',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}') # the leptons
    
    # Z boson kinematics
    df = df.Define('zll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]')
    df = df.Define('zll_p',           'FCCAnalyses::ReconstructedParticle::get_p(zll)[0]')
    df = df.Define('zll_pT',          'FCCAnalyses::ReconstructedParticle::get_pt(zll)[0]')
    df = df.Define('zll_theta',       'FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]')
    df = df.Define('zll_phi',         'FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]')
    
    # Recoil mass (Higgs candidate)
    df = df.Define('zll_recoil',   f'FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)')
    df = df.Define('zll_recoil_m', 'FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]')
    
    # Individual Z lepton kinematics with leading/subleading ordering
    df = df.Define('zll_leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(zll_leps)')
    df = df.Define('zll_leps_pT',      'FCCAnalyses::ReconstructedParticle::get_pt(zll_leps)')
    df = df.Define('zll_leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)')
    df = df.Define('zll_leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)')
    df = df.Define('leading_p_idx',    '(zll_leps_p[0] > zll_leps_p[1]) ? 0 : 1')
    df = df.Define('subleading_p_idx', '(zll_leps_p[0] > zll_leps_p[1]) ? 1 : 0')

    df = df.Define('leading_p',        'zll_leps_p[leading_p_idx]')
    df = df.Define('leading_pT',       'zll_leps_pT[leading_p_idx]')
    df = df.Define('subleading_p',     'zll_leps_p[subleading_p_idx]')
    df = df.Define('subleading_pT',    'zll_leps_pT[subleading_p_idx]')

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

    # Visible energy (excluding Z candidate leptons)
    df = df.Define('rps_no_leps', 'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)')
    df = df.Define('visibleEnergy', 'FCCAnalyses::visibleEnergy(rps_no_leps)')

    ##########
    ### CUT 3: Z mass window
    ##########
    # df = df.Filter('zll_m > 86 && zll_m < 96') 

    ##########
    ### CUT 4: Z momentum (CoM dependent)
    ##########
    # if ecm == 240:
    #     df = df.Filter('zll_p > 20 && zll_p < 70')  # 240 GeV CoM
    # elif ecm == 365:
    #     df = df.Filter('zll_p > 50 && zll_p < 150')  # 365 GeV CoM
    
    return df



#####################################################
### CLASS AND OUTPUT DEFINITION FOR PRE-SELECTION ###
#####################################################

class RDFanalysis():
    """RDataFrame analysis class for pre-selection."""

    #_________________________________________________________________
    # Mandatory: analysers function to define the analysers to process
    def analysers(df):
        """Apply analysis graph construction to the dataframe."""
        df = build_graph_ll(df, cat)
        return df
    
    #_____________________________________________________
    # Mandatory: output function defining branches to save
    def output():
        """Define output branches to save."""
        branchList = [
            # Lepton kinematics (leading and subleading)
            'leading_p',    'leading_pT',    'leading_theta',    'leading_phi',
            'subleading_p', 'subleading_pT', 'subleading_theta', 'subleading_phi',

            # Angular correlation
            'acolinearity', 'acoplanarity', 'deltaR',

            # Z boson kinematics
            'zll_m', 'zll_p', 'zll_pT', 'zll_theta', 'zll_phi',

            # Recoil mass (Higgs candidate)
            'zll_recoil_m',

            # Missing energy variables
            'visibleEnergy', 'cosTheta_miss', 'missingMass',

            # Higgsstrahlungness discriminant
            'H'
        ]
        return branchList
