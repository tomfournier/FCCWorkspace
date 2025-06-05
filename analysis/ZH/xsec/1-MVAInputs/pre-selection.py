import ROOT
import importlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cat', help='Final state (ee, mumu), qq is not available yet', choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--param', help='Select the fraction of the samples or the number of files to put the samples', choices=['frac', 'chunks'], default='frac')
arg = parser.parse_args()

if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)


# Load userConfig 
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, select, frac, nb

final_state, ecm = arg.cat, arg.ecm
sel = select(arg.recoil120, arg.miss)

# Output directory where the files produced at the pre-selection level will be put
outputDir = get_loc(loc.MVA_INPUTS, final_state, ecm, sel)

# include custom functions
includePaths = ["../../../../functions/functions.h"]

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = "FCCee/winter2023_training/IDEA/"
# Optional: output directory, default is local dir
eosType = "eosuser"
# Optional: ncpus, default is 4
nCPUS = 20
# Optional running on HTCondor, default is False
runBatch = False
# Optional batch queue name when running on HTCondor, default is workday
batchQueue = "longlunch"
# Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
compGroup = "group_u_FCC.local_gen"

# Process samples for BDT
ee_ll = f"wzp6_ee_ee_Mee_30_150_ecm{ecm}" if final_state=='ee' else f"wzp6_ee_mumu_ecm{ecm}"
samples_BDT = [
    #signal
    f"wzp6_ee_{final_state}H_ecm{ecm}",

    #background: 
    f"p8_ee_ZZ_ecm{ecm}", f"p8_ee_WW_{final_state}_ecm{ecm}", ee_ll,

    #rare backgrounds:
    f"wzp6_egamma_eZ_Z{final_state}_ecm{ecm}", f"wzp6_gammae_eZ_Z{final_state}_ecm{ecm}",
    f"wzp6_gaga_{final_state}_60_ecm{ecm}"
]

if arg.param=='frac':     param = {'fraction': frac} 
elif arg.param=='chunks': param = {'chunks':   nb}

# Mandatory: List of processes
processList = {i:param for i in samples_BDT}

class RDFanalysis():

    #__________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df
    def analysers(df):

        ################################################
        ## Alias for lepton and MC truth informations ##
        ################################################
        if final_state == "mumu":
            df = df.Alias("Lepton0", "Muon#0.index")
        elif final_state == "ee":
            df = df.Alias("Lepton0", "Electron#0.index")
        else:
            raise ValueError(f"final_state {final_state} not supported")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("Photon0", "Photon#0.index")
        
        # Missing ET
        df = df.Define("missingEnergy", f"FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)")
        df = df.Define("cosTheta_miss", "FCCAnalyses::get_cosTheta_miss(missingEnergy)")
        
        # all leptons (bare)
        df = df.Define("leps_all", "FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)")
        df = df.Define("leps_all_p", "FCCAnalyses::ReconstructedParticle::get_p(leps_all)")
        df = df.Define("leps_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps_all)")
        df = df.Define("leps_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps_all)")
        df = df.Define("leps_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps_all)")
        df = df.Define("leps_all_no", "FCCAnalyses::ReconstructedParticle::get_n(leps_all)")
        df = df.Define("leps_all_iso", "FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)") 
        
        # cuts on leptons
        df = df.Define("leps", "FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)")
        df = df.Define("leps_p", "FCCAnalyses::ReconstructedParticle::get_p(leps)")
        df = df.Define("leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps)")
        df = df.Define("leps_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps)")
        df = df.Define("leps_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps)")
        df = df.Define("leps_no", "FCCAnalyses::ReconstructedParticle::get_n(leps)")
        df = df.Define("leps_iso", "FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)")
        df = df.Define("leps_sel_iso", "FCCAnalyses::sel_isol(0.25)(leps, leps_iso)")

        #########
        ### CUT 0: no cut
        #########

        #########
        ### CUT 1: at least one lepton and at least one lepton isolated (I_rel < 0.25)
        #########
        df = df.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")

        #########
        ### CUT 2: at least 2 leptons
        #########
        df = df.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")

        # remove H->mumu/ee candidate leptons
        df = df.Define("zbuilder_result_Hll", f"FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
        df = df.Define("zll_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}") # the Z
        df = df.Define("zll_Hll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]")
        df = df.Define("zll_leps_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}") # the leptons
        df = df.Define("zll_leps_dummy", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}") # the leptons
        df = df.Define("leps_to_remove", "return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy")
        df = df.Define("leps_good", "FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)") 

        # build the Z resonance based on the available leptons. 
        # Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
        # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, 
        # index 1 and 2 the leptons of the pair
        df = df.Define("zbuilder_result", f"FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
        df = df.Define("zll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}") # the Z
        df = df.Define("zll_leps", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}") # the leptons
        df = df.Define("zll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]")
        df = df.Define("zll_p", "FCCAnalyses::ReconstructedParticle::get_p(zll)[0]")
        df = df.Define("zll_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]")
        df = df.Define("zll_phi", "FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]") 
        
        # recoil
        df = df.Define("zll_recoil", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)")
        df = df.Define("zll_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]")
        
        # Z leptons informations
        df = df.Define("zll_leps_p", "FCCAnalyses::ReconstructedParticle::get_p(zll_leps)")
        df = df.Define("zll_leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)")
        df = df.Define("zll_leps_phi", "FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)")
        df = df.Define("leading_p_idx", "(zll_leps_p[0] > zll_leps_p[1]) ? 0 : 1")
        df = df.Define("subleading_p_idx", "(zll_leps_p[0] > zll_leps_p[1]) ? 1 : 0")
        df = df.Define("leading_p", "zll_leps_p[leading_p_idx]")
        df = df.Define("subleading_p", "zll_leps_p[subleading_p_idx]")
        df = df.Define("leading_theta", "zll_leps_theta[leading_p_idx]")
        df = df.Define("subleading_theta", "zll_leps_theta[subleading_p_idx]")
        df = df.Define("leading_phi", "zll_leps_phi[leading_p_idx]")
        df = df.Define("subleading_phi", "zll_leps_phi[subleading_p_idx]")
        
        df = df.Define("acoplanarity", "FCCAnalyses::acoplanarity(zll_leps)")
        df = df.Define("acolinearity", "FCCAnalyses::acolinearity(zll_leps)")

        # Higgsstrahlungness
        df = df.Define("H", "FCCAnalyses::Higgsstrahlungness(zll_m, zll_recoil_m)")

        #########
        ### CUT 3: Z mass between 86 and 96 GeV
        #########
        df = df.Filter("zll_m > 86 && zll_m < 96") 

        #########
        ### CUT 4: Z momentum between 20 and 70 GeV (240 GeV) or > 20 GeV (365 GeV)
        #########
        if ecm == 240:
            df = df.Filter("zll_p > 20 && zll_p < 70")
        elif ecm == 365:
            df = df.Filter("zll_p > 50 && zll_p < 150")

        #########
        ### CUT 5: recoil mass cut
        #########
        if arg.recoil120:
            df = df.Filter("zll_recoil_m < 140 && zll_recoil_m > 120")
        else:
            df = df.Filter("zll_recoil_m < 150 && zll_recoil_m > 100")

        #########
        ### CUT 6: cos(theta_miss) cut
        #########
        if arg.miss:
            df = df.Filter("cosTheta_miss < 0.98")
        
        return df

    #__________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            # Reconstructed Particle
            # leptons
            "leading_p",    "leading_theta",    "leading_phi",
            "subleading_p", "subleading_theta", "subleading_phi",
            "acolinearity", "acoplanarity",
            # Zed
            "zll_m", "zll_p", "zll_theta", "zll_phi",
            # Recoil
            "zll_recoil_m",
            # missing Information
            "cosTheta_miss",
            # Higgsstrahlungness
            "H",
        ]
        return branchList