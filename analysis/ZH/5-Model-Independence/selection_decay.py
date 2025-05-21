import ROOT
import array
import importlib
import numpy as np

ROOT.TH1.SetDefaultSumw2(ROOT.kTRUE)
from addons.TMVAHelper.TMVAHelper import TMVAHelperXGB # type: ignore

userConfig = importlib.import_module('userConfig')
ecm = userConfig.ecm

fraction = 1

processListBkg = {

    f'p8_ee_WW_ecm{ecm}':                  {'fraction':fraction},
    f'p8_ee_ZZ_ecm{ecm}':                  {'fraction':fraction},
    f'wzp6_ee_tautau_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_mumu_ecm{ecm}':             {'fraction':fraction},
    f'wzp6_ee_ee_Mee_30_150_ecm{ecm}':    {'fraction':fraction},

    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}':      {'fraction':fraction},
    f'wzp6_gammae_eZ_Zmumu_ecm{ecm}':      {'fraction':fraction},
    f'wzp6_egamma_eZ_Zee_ecm{ecm}':        {'fraction':fraction},
    f'wzp6_gammae_eZ_Zee_ecm{ecm}':        {'fraction':fraction},

    f'wzp6_gaga_ee_60_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_gaga_mumu_60_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_gaga_tautau_60_ecm{ecm}':       {'fraction':fraction},
    f'wzp6_ee_nuenueZ_ecm{ecm}':           {'fraction':fraction},
}

processListSignal = {

    f'wzp6_ee_eeH_Hbb_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hcc_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hss_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hgg_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Haa_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HZa_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HWW_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_HZZ_ecm{ecm}':           {'fraction':fraction},
    f'wzp6_ee_eeH_Hmumu_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_eeH_Htautau_ecm{ecm}':       {'fraction':fraction},

    f'wzp6_ee_mumuH_Hbb_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hcc_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hss_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hgg_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Haa_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HZa_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HWW_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_HZZ_ecm{ecm}':         {'fraction':fraction},
    f'wzp6_ee_mumuH_Hmumu_ecm{ecm}':       {'fraction':fraction},
    f'wzp6_ee_mumuH_Htautau_ecm{ecm}':     {'fraction':fraction},
    
    f'wzp6_ee_ZH_Hinv_ecm{ecm}':           {'fraction':fraction},

}

processList = processListSignal | processListBkg

prodTag = "FCCee/winter2023/IDEA/"
procDict = "FCCee_procDict_winter2023_IDEA.json"

# output directory
outputDir = userConfig.loc.MODEL

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = 20

# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = userConfig.intLumi * 1e6

bins_count = (50, 0, 50)

ROOT.gInterpreter.ProcessLine('''
TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/mumu/xgb_bdt.root");
tmva_mumu = TMVA::Experimental::Compute<9, float>(bdt);
''')

ROOT.gInterpreter.ProcessLine('''
TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/ee/xgb_bdt.root");
tmva_ee = TMVA::Experimental::Compute<9, float>(bdt);
''')

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
    ### CUT 0 all events
    #########
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut0"))

    #########
    ### CUT 1: at least a lepton with at least 1 isolated one
    #########
    df2 = df2.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut1"))

    #########
    ### CUT 2 :at least 2 OS leptons, and build the resonance
    #########
    df2 = df2.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut2"))

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
    ### CUT 3: Z mass window
    #########
    df2 = df2.Filter("zll_m > 86 && zll_m < 96")
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut3"))

    #########
    ### CUT 4: Z momentum
    #########
    if ecm == 240:
        df2 = df2.Filter("zll_p > 20 && zll_p < 70")
    if ecm == 365:
        df2 = df2.Filter("zll_p > 50 && zll_p < 150")
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut4"))

    #########
    ### CUT 5: recoil cut
    #########
    df2 = df2.Filter("zll_recoil_m < 140 && zll_recoil_m > 120")
    hists.append(df2.Histo1D((f"{final_state}_recoil_m", "", *bins_count), "zll_recoil_m"))
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut5"))

    ##########
    ### MVA
    ##########

    if final_state=='mumu':
        df2 = df2.Define("MVAVec", ROOT.tmva_mumu, userConfig.train_vars)
    if final_state=='ee':
        df2 = df2.Define("MVAVec", ROOT.tmva_ee, userConfig.train_vars)
    df2 = df2.Define("BDTscore", "MVAVec.at(0)")
    hists.append(df2.Histo1D((f"{final_state}_mva_score", "", *(1000, 0, 1)), "BDTscore"))

    ########################
    # Final  (Baseline)
    ########################

    mva_sign = 0.84 if final_state=='mumu' else 0.83
    bins_mva_ = [0, mva_sign, 1]
    bins_mrec_ = list(np.arange(100, 150.5, 0.5))
    bins_mva = array.array('d', bins_mva_)
    bins_mrec = array.array('d', bins_mrec_)
    model = ROOT.RDF.TH2DModel(f"{final_state}_recoil_m_mva", "", len(bins_mrec_)-1, bins_mrec, len(bins_mva_)-1, bins_mva)
    hists.append(df2.Histo2D(model, "zll_recoil_m", "BDTscore"))


    #########
    ### CUT 6: cosThetaMiss, for mass analysis
    #########
    df = df2.Filter("cosTheta_miss < 0.98")
    hists.append(df2.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut6"))

    ########################
    # Final  (Baseline)
    ########################

    mva_sign = 0.84 if final_state=='mumu' else 0.83
    bins_mva_ = [0, mva_sign, 1]
    bins_mrec_ = list(np.arange(100, 150.5, 0.5))
    bins_mva = array.array('d', bins_mva_)
    bins_mrec = array.array('d', bins_mrec_)
    model = ROOT.RDF.TH2DModel(f"{final_state}_recoil_m_mva_miss", "", len(bins_mrec_)-1, bins_mrec, len(bins_mva_)-1, bins_mva)
    hists.append(df2.Histo2D(model, "zll_recoil_m", "BDTscore"))

    return hists


def build_graph(df, dataset):

    df2 = df
    hists = []

    df2 = df2.Define("weight", "1.0")
    weightsum = df2.Sum("weight")

    df2 = df2.Define("cut0", "0")
    df2 = df2.Define("cut1", "1")
    df2 = df2.Define("cut2", "2")
    df2 = df2.Define("cut3", "3")
    df2 = df2.Define("cut4", "4")
    df2 = df2.Define("cut5", "5")
    df2 = df2.Define("cut6", "6")

    build_graph_ll(df2, hists, dataset, "mumu")
    build_graph_ll(df2, hists, dataset, "ee")

    return hists, weightsum