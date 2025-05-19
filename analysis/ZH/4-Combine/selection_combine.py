import importlib
import numpy as np
import array

# Load userConfig
userConfig = importlib.import_module("userConfig")

# Define final state and ecm
final_state = userConfig.final_state
ecm = userConfig.ecm

samples = {
    #signal
    f"wzp6_ee_mumuH_ecm{ecm}", f"wzp6_ee_eeH_ecm{ecm}",
    #background: 
    f"p8_ee_WW_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
    f"wzp6_ee_mumu_ecm{ecm}", f"wzp6_ee_ee_Mee_30_150_ecm{ecm}",
    f"wzp6_egamma_eZ_Zmumu_ecm{ecm}", f"wzp6_egamma_eZ_Zee_ecm{ecm}",
    f"wzp6_gammae_eZ_Zmumu_ecm{ecm}", f"wzp6_gammae_eZ_Zee_ecm{ecm}",
    f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_ee_60_ecm{ecm}",
    f"wzp6_ee_tautau_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f"wzp6_ee_nuenueZ_ecm{ecm}"
}

# Parameter of processList
paramList = {'fraction':1}
processList = {i:paramList for i in samples}

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = "FCCee/winter2023/IDEA/"
procDict = "FCCee_procDict_winter2023_IDEA.json"

# Input directory where the files produced at the pre-selection will be put
outputDir = userConfig.loc.COMBINE_HIST
eosType = "eosuser"

#Optional: ncpus, default is 4
nCPUS = 40

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

ROOT.gInterpreter.ProcessLine('''
TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/mumu/xgb_bdt.root");
computeModel1 = TMVA::Experimental::Compute<9, float>(bdt);
''')

ROOT.gInterpreter.ProcessLine('''
TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/ee/xgb_bdt.root");
computeModel2 = TMVA::Experimental::Compute<9, float>(bdt);
''')

#__________________________________________________________
# Mandatory: analysers funtion to define the analysers to process, please make sure you return the last dataframe, in this example it is df2
def build_graph_ll(df, hists, dataset, final_state):
    df2 = df

    #############################################
    ## Alias for muon and MC truth informations##
    #############################################
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
    df2 = df2.Define("cosTheta_miss", "abs(HiggsTools::get_cosTheta(MissingET))") 
    
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
    
    #########
    ### CUT 0: no cut
    #########
    # df2 = df2.Define("cut0", "0")
    
    #########
    ### CUT 1: at least a lepton
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
    df2 = df2.Define("zbuilder_result", "HiggsTools::resonanceBuilder_mass_recoil(91.2, 125, 0.4, 240, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df2 = df2.Define("zll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}") # the Z
    df2 = df2.Define("zll_leps", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}") # the leptons
    df2 = df2.Define("zll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]")
    df2 = df2.Define("zll_p", "FCCAnalyses::ReconstructedParticle::get_p(zll)[0]")
    df2 = df2.Define("zll_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]")
    df2 = df2.Define("zll_phi", "FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]")

    # recoil
    df2 = df2.Define("zll_recoil", "FCCAnalyses::ReconstructedParticle::recoilBuilder(240)(zll)")
    df2 = df2.Define("zll_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]")
    df2 = df2.Define("zll_category", "HiggsTools::polarAngleCategorization(0.8, 2.34)(zll_leps)")
    
    df2 = df2.Define("zll_leps_p", "FCCAnalyses::ReconstructedParticle::get_p(zll_leps)")
    df2 = df2.Define("zll_leps_dR", "HiggsTools::deltaR(zll_leps)")
    df2 = df2.Define("zll_leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)")
    
    df2 = df2.Define("prompt_muons", "HiggsTools::whizard_zh_select_prompt_leptons(zll_leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df2 = df2.Define("prompt_muons_no", "prompt_muons.size()")
    # df2 = df2.Filter("prompt_muons.size() == 2") 

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
    ### CUT 3: Zll mass between 73 and 120 GeV
    #########
    df2 = df2.Filter("zll_m > 86 && zll_m < 96") 
    # df2 = df2.Define("cut3", "3")

    #########
    ### CUT 4: zll momentum > 5 GeV
    #########
    df2 = df2.Filter("zll_p > 20 && zll_p < 70")
    # df2 = df2.Define("cut4", "4")

    #########
    ### CUT 5: recoil mass between 120 and 140 GeV
    #########
    df2 = df2.Filter("zll_recoil_m < 140 && zll_recoil_m > 120")
    # df2 = df2.Define("cut5", "5")

    #########
    ### CUT 6: cos(theta_miss) < 0.98
    #########
    # df2 = df2.Filter("cosTheta_miss.size() >= 1 && cosTheta_miss[0] < 0.98")
    # df2 = df2.Define("cut6", "6")
    
    ##############
    ### Define MVA 
    ##############
    if final_state=='mumu':
        df2 = df2.Define("MVAVec", ROOT.computeModel1, userConfig.train_vars)
    if final_state=='ee':
        df2 = df2.Define("MVAVec", ROOT.computeModel2, userConfig.train_vars)
    df2 = df2.Define("BDTscore", "MVAVec.at(0)")

    if final_state=='mumu':
        bdtcut = 0.84
    elif final_state=='ee':
        bdtcut = 0.83

    bins_mva_ = [0, bdtcut, 1]
    bins_mrec_ = list(np.arange(120, 140.5, 0.5))
    bins_mva = array.array('d', bins_mva_)
    bins_mrec = array.array('d', bins_mrec_)
    model = ROOT.RDF.TH2DModel(f"{final_state}_recoil_m_mva", "", len(bins_mrec_)-1, bins_mrec, len(bins_mva_)-1, bins_mva)
    hists.append(df2.Histo2D(model, "zll_recoil_m", "BDTscore"))

    return hists


def build_graph(df, dataset):

    hists = []

    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    build_graph_ll(df, hists, dataset, "mumu")
    build_graph_ll(df, hists, dataset, "ee")
    return hists, weightsum
