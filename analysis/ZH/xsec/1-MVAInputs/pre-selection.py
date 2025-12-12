import importlib

# Load userConfig 
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, ecm, param

# Select Z decay
cat = input('Select channel [ee, mumu]: ')

# Optional: output directory, 
# default is local dir
outputDir = get_loc(loc.EVENTS_TRAINING,  cat, ecm, '')

# include custom functions
includePaths = ['../../../../functions/functions.h']

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = 'FCCee/winter2023_training/IDEA/'

# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = 'FCCee_procDict_winter2023_training_IDEA.json'

eosType = 'eosuser'

# Optional: ncpus, default is 4
nCPUS = 10

# Optional running on HTCondor, 
# default is False
runBatch = False

# Optional batch queue name when running on HTCondor, 
# default is workday
batchQueue = 'longlunch'

# Optional computing account when running on HTCondor, 
# default is group_u_FCC.local_gen
compGroup = 'group_u_FCC.local_gen'



# Process samples for BDT
samples_ee = [
    #signal
    f'wzp6_ee_eeH_ecm{ecm}',
    #background: 
    f'p8_ee_ZZ_ecm{ecm}', f'p8_ee_WW_ee_ecm{ecm}', 
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}',
    #rare backgrounds:
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f'wzp6_gaga_ee_60_ecm{ecm}'
]

samples_mumu = [
    #signal
    f'wzp6_ee_mumuH_ecm{ecm}',
    #background: 
    f'p8_ee_ZZ_ecm{ecm}', f'p8_ee_WW_mumu_ecm{ecm}', 
    f'wzp6_ee_mumu_ecm{ecm}',
    #rare backgrounds:
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_gaga_mumu_60_ecm{ecm}'
]

if   cat=='ee':   samples_BDT = samples_ee
elif cat=='mumu': samples_BDT = samples_mumu
else: raise ValueError(f'cat {cat} not supported')

# Mandatory: List of processes
processList = {i:param for i in samples_BDT}



def build_graph_ll(df, cat):

    ################################################
    ## Alias for lepton and MC truth informations ##
    ################################################
    if   cat == 'mumu':
        df = df.Alias('Lepton0', 'Muon#0.index')
    elif cat == 'ee':
        df = df.Alias('Lepton0', 'Electron#0.index')

    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')
    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('Photon0',   'Photon#0.index')
    
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
    df = df.Define('leps',         'FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)')
    df = df.Define('leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(leps)')
    df = df.Define('leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(leps)')
    df = df.Define('leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(leps)')
    df = df.Define('leps_q',       'FCCAnalyses::ReconstructedParticle::get_charge(leps)')
    df = df.Define('leps_no',      'FCCAnalyses::ReconstructedParticle::get_n(leps)')
    df = df.Define('leps_iso',     'FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)')
    df = df.Define('leps_sel_iso', 'FCCAnalyses::sel_isol(0.25)(leps, leps_iso)')
    df = df.Define('leps_sel_no',  'leps_sel_iso.size()')

    #########
    ### CUT 0: no cut
    #########

    #########
    ### CUT 1: at least one lepton and at least one lepton isolated (I_rel < 0.25)
    #########
    df = df.Filter('leps_no >= 1 && leps_sel_iso.size() > 0')

    #########
    ### CUT 2: at least 2 leptons
    #########
    df = df.Filter('leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()')

    # remove H->mumu/ee candidate leptons
    df = df.Define('zbuilder_result_Hll', f'FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)'
                    '(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)')
    df = df.Define('zll_Hll',             'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}') # the Z
    df = df.Define('zll_Hll_m',           'FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]')
    df = df.Define('zll_leps_Hll',        'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}') # the leptons
    df = df.Define('zll_leps_dummy',      'ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}') # the leptons
    df = df.Define('leps_to_remove',      'return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy')
    df = df.Define('leps_good',           'FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)') 

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
    
    # Z leptons informations
    df = df.Define('zll_leps_p',       'FCCAnalyses::ReconstructedParticle::get_p(zll_leps)')
    df = df.Define('zll_leps_theta',   'FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)')
    df = df.Define('zll_leps_phi',     'FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)')
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
    df = df.Define('rps_no_leps', 'FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)')
    df = df.Define('visibleEnergy', 'FCCAnalyses::visibleEnergy(rps_no_leps)')

    #########
    ### CUT 3: Z mass window
    #########
    df = df.Filter('zll_m > 86 && zll_m < 96') 

    #########
    ### CUT 4: Z momentum 
    #########
    if ecm == 240:
        df = df.Filter('zll_p > 20 && zll_p < 70')
    elif ecm == 365:
        df = df.Filter('zll_p > 50 && zll_p < 150')
    
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
            # Leptons
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
            'H'
        ]
        return branchList
