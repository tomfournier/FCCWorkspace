import importlib

# Load userConfig 
userConfig = importlib.import_module("userConfig")

# Output directory where the files produced at the pre-selection level will be put
outputDir = userConfig.loc.PRESEL

# Define final_state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

# Mandatory: List of processes
processList = {i:userConfig.paramList for i in userConfig.processList}

if final_state=="mumu":
    processList[f"ee_{final_state}_ecm{ecm}"] = userConfig.paramList
elif final_state=="ee":
    processList[f"wzp6_ee_{final_state}_Mee_30_150_ecm{ecm}"] = userConfig.paramList

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = "FCCee/winter2023/IDEA/"

# Optional: output directory, default is local dir
eosType = "eosuser"
# Optional: ncpus, default is 4
nCPUS = 4

# Optional running on HTCondor, default is False
# runBatch = True
runBatch = False

# Optional batch queue name when running on HTCondor, default is workday
batchQueue = "longlunch"

# Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
compGroup = "group_u_FCC.local_gen"

# userBatchConfig = userConfig.loc.BATCH

# USER DEFINED CODE
import ROOT
ROOT.gInterpreter.Declare("""
bool Selection(ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> in){
    //at least one muon + and one muon - in each event
    int n_muon_plus = 0;
	int n_muon_minus = 0;
	int n = in.size();
	for (int i = 0; i < n; ++i) {
	    if (in[i].charge == 1.0){
	        ++n_muon_plus;
	    }
	    else if (in[i].charge == -1.0){
	        ++n_muon_minus;
	    }
	}
	if (n_muon_plus >= 1 && n_muon_minus >= 1){
		return true;
	}
    else{
        return false;
    }
}
""")

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis():

    #__________________________________________________________
    # Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
    def analysers(df):
        df2 = df

        ################################################
        ## Alias for lepton and MC truth informations ##
        ################################################
        if final_state == "mumu":
            df2 = df2.Alias("Lepton0", "Muon#0.index")
        elif final_state == "ee":
            df2 = df2.Alias("Lepton0", "Electron#0.index")
        else:
            raise ValueError(f"final_state {final_state} not supported")
        df2 = df2.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df2 = df2.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df2 = df2.Alias("Particle0", "Particle#0.index")
        df2 = df2.Alias("Particle1", "Particle#1.index")
        df2 = df2.Alias("Photon0", "Photon#0.index")
        
        # photons
        df2 = df2.Define("photons", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df2 = df2.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons)")
        df2 = df2.Define("photons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(photons)")
        df2 = df2.Define("photons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(photons)")
        df2 = df2.Define("photons_no", "FCCAnalyses::ReconstructedParticle::get_n(photons)")
        
        df2 = df2.Define("gen_photons", "HiggsTools::get_photons(Particle)")
        df2 = df2.Define("gen_photons_p", "FCCAnalyses::MCParticle::get_p(gen_photons)")
        df2 = df2.Define("gen_photons_theta", "FCCAnalyses::MCParticle::get_theta(gen_photons)")
        df2 = df2.Define("gen_photons_phi", "FCCAnalyses::MCParticle::get_phi(gen_photons)")
        df2 = df2.Define("gen_photons_no", "FCCAnalyses::MCParticle::get_n(gen_photons)")
        
        # Missing ET
        df2 = df2.Define("cosTheta_miss", "HiggsTools::get_cosTheta(MissingET)") 
        
        # all leptons (bare)
        df2 = df2.Define("leps_all", "FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)")
        df2 = df2.Define("leps_all_p", "FCCAnalyses::ReconstructedParticle::get_p(leps_all)")
        df2 = df2.Define("leps_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps_all)")
        df2 = df2.Define("leps_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps_all)")
        df2 = df2.Define("leps_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps_all)")
        df2 = df2.Define("leps_all_no", "FCCAnalyses::ReconstructedParticle::get_n(leps_all)")
        df2 = df2.Define("leps_all_iso", "HiggsTools::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)") 
        df2 = df2.Define("leps_all_p_gen", "HiggsTools::gen_p_from_reco(leps_all, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
        
        # cuts on leptons
        df2 = df2.Define("leps", "FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)")
        df2 = df2.Define("leps_p", "FCCAnalyses::ReconstructedParticle::get_p(leps)")
        df2 = df2.Define("leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps)")
        df2 = df2.Define("leps_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps)")
        df2 = df2.Define("leps_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps)")
        df2 = df2.Define("leps_no", "FCCAnalyses::ReconstructedParticle::get_n(leps)")
        df2 = df2.Define("leps_iso", "HiggsTools::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)")
        df2 = df2.Define("leps_sel_iso", "HiggsTools::sel_isol(0.25)(leps, leps_iso)")

        # momentum resolution
        df2 = df2.Define("leps_all_reso_p", "HiggsTools::leptonResolution_p(leps_all, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
        df2 = df2.Define("leps_reso_p", "HiggsTools::leptonResolution_p(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
        
        # build the Z resonance and recoil using MC information from the selected leptons
        df2 = df2.Define("zed_leptonic_MC", "HiggsTools::resonanceZBuilder2(91, true)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
        df2 = df2.Define("zed_leptonic_m_MC", "FCCAnalyses::ReconstructedParticle::get_mass(zed_leptonic_MC)")
        df2 = df2.Define("zed_leptonic_recoil_MC", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zed_leptonic_MC)")
        df2 = df2.Define("zed_leptonic_recoil_m_MC", "FCCAnalyses::ReconstructedParticle::get_mass(zed_leptonic_recoil_MC)")

        #########
        ### CUT 0: no cut
        #########
        # df2 = df2.Define("cut0", "0")

        #########
        ### CUT 1: at least one lepton and at least one lepton isolated (I_rel < 0.25)
        #########
        df2 = df2.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")
        # df2 = df2.Define("cut1", "1")

        #########
        ### CUT 2: at least 2 leptons
        #########
        df2 = df2.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")
        # df2 = df2.Define("cut2", "2")

        # build the Z resonance based on the available leptons. 
        # Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
        # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, 
        # index 1 and 2 the leptons of the pair
        df2 = df2.Define("zbuilder_result_Hll", f"HiggsTools::resonanceBuilder_mass_recoil2(125, 91.2, 0.4, {ecm}, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
        df2 = df2.Define("zll_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}") # the Z
        df2 = df2.Define("zll_Hll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]")
        df2 = df2.Define("zll_leps_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}") # the leptons
        df2 = df2.Define("zll_leps_dummy", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}") # the leptons
        df2 = df2.Define("leps_to_remove", "return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy")
        df2 = df2.Define("leps_good", "FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)") 
        df2 = df2.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()") 
        df2 = df2.Define("zbuilder_result", f"HiggsTools::resonanceBuilder_mass_recoil2(91.2, 125, 0.4, {ecm}, false)(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
        df2 = df2.Define("zll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}") # the Z
        df2 = df2.Define("zll_leps", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}") # the leptons
        df2 = df2.Define("zll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]")
        df2 = df2.Define("zll_p", "FCCAnalyses::ReconstructedParticle::get_p(zll)[0]")
        df2 = df2.Define("zll_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]")
        df2 = df2.Define("zll_phi", "FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]") 
        
        # recoil
        df2 = df2.Define("zll_recoil", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)")
        df2 = df2.Define("zll_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]")
        
        # Z leptons informations
        df2 = df2.Define("sorted_leptons", "HiggsTools::sort_greater_p(zll_leps)")
        df2 = df2.Define("sorted_p", "FCCAnalyses::ReconstructedParticle::get_p(sorted_leptons)")
        df2 = df2.Define("sorted_m", "FCCAnalyses::ReconstructedParticle::get_mass(sorted_leptons)")
        df2 = df2.Define("sorted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(sorted_leptons)")
        df2 = df2.Define("sorted_phi", "FCCAnalyses::ReconstructedParticle::get_phi(sorted_leptons)")
        df2 = df2.Define("leading_p", "return sorted_p.at(0)")
        df2 = df2.Define("leading_m", "return sorted_m.at(0)")
        df2 = df2.Define("leading_theta", "return sorted_theta.at(0)")
        df2 = df2.Define("leading_phi", "return sorted_phi.at(0)")
        df2 = df2.Define("subleading_p", "return sorted_p.at(1)")
        df2 = df2.Define("subleading_m", "return sorted_m.at(1)")
        df2 = df2.Define("subleading_theta", "return sorted_theta.at(1)")
        df2 = df2.Define("subleading_phi", "return sorted_phi.at(1)")
        
        df2 = df2.Define("zll_acolinearity", "HiggsTools::acolinearity(sorted_leptons)")
        df2 = df2.Define("zll_acoplanarity", "HiggsTools::acoplanarity(sorted_leptons)") 
        df2 = df2.Define("acolinearity", "if(zll_acolinearity.size()>0) return zll_acolinearity.at(0); else return -std::numeric_limits<float>::max()") 
        df2 = df2.Define("acoplanarity", "if(zll_acoplanarity.size()>0) return zll_acoplanarity.at(0); else return -std::numeric_limits<float>::max()") 
        
        # Higgsstrahlungness
        df2 = df2.Define("H", "HiggsTools::Higgsstrahlungness(zll_m, zll_recoil_m)")

        #########
        ### CUT 3: Z mass between 86 and 96 GeV
        #########
        df2 = df2.Filter("zll_m > 86 && zll_m < 96") 
        # df2 = df2.Define("cut3", "3")

        #########
        ### CUT 4: Z momentum between 20 and 70 GeV (240 GeV) or > 20 GeV (365 GeV)
        #########
        if ecm == "240":
            df2 = df2.Filter("zll_p > 20 && zll_p < 70")
        elif ecm == "365":
            df2 = df2.Filter("zll_p > 20")
        # df2 = df2.Define("cut4", "4")

        #########
        ### CUT 5: recoil mass between 100 and 150 GeV
        #########
        df2 = df2.Filter("zll_recoil_m < 150 && zll_recoil_m > 100")
        # df2 = df2.Define("cut5", "5")
        
        return df2

    #__________________________________________________________
    # Mandatory: output function, please make sure you return the branchlist as a python list
    def output():
        branchList = [
            # Reconstructed Particle
            # leptons
            "leading_p", "leading_m",  
            "leading_theta", "leading_phi",
            "subleading_p", "subleading_m",
            "subleading_theta", "subleading_phi",
            "acolinearity", "acoplanarity",
            # Zed
            "zll_m", "zll_p",
            "zll_theta", "zll_phi",
            # Recoil
            "zll_recoil_m",
            # missing Information
            "cosTheta_miss",
            # Higgsstrahlungness
            "H",
            # Cutflow
            # "cut0", "cut1",
            # "cut2", "cut3",
            # "cut4", "cut5"
        ]
        return branchList
