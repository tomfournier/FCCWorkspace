import ROOT
import array
import importlib
import numpy as np

ROOT.TH1.SetDefaultSumw2(ROOT.kTRUE)
from addons.TMVAHelper.TMVAHelper import TMVAHelperXGB # type: ignore

userConfig = importlib.import_module('userConfig')
from userConfig import ecm, loc, train_vars, miss, selection, List_meas

prodTag = "FCCee/winter2023/IDEA/"
procDict = "FCCee_procDict_winter2023_IDEA.json"

processList = List_meas

# output directory
outputDir = loc.HIST_PREPROCESSED

# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = 20

# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = userConfig.intLumi * 1e6

# define histograms
bins_p_mu        = (2000, 0, 200)  # 100 MeV bins
bins_m_ll        = (2000, 0, 200)  # 100 MeV bins
bins_p_ll        = (200, 0, 200)   # 1 GeV bins
bins_recoil      = (20000, 0, 200) # 10 MeV bins 
bins_recoil_fine = (500, 100, 150) # 100 MeV bins 
bins_miss        = (10000, 0, 1)

bins_theta = (400, 0, 4)
bins_phi   = (400, -4, 4)
bins_aco   = (400, 0, 4)

bins_iso   = (500, 0, 5)
bins_count = (50, 0, 50)

ROOT.EnableImplicitMT(nCPUS) # hack to deal correctly with TMVAHelperXGB

tmva_mumu = TMVAHelperXGB(f"{loc.MVA}/{ecm}/mumu/{selection}/BDT/xgb_bdt.root", "ZH_Recoil_BDT", variables=train_vars)
tmva_ee   = TMVAHelperXGB(f"{loc.MVA}/{ecm}/ee/{selection}/BDT/xgb_bdt.root", "ZH_Recoil_BDT", variables=train_vars)

# ROOT.gInterpreter.ProcessLine('''
# TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/mumu/xgb_bdt.root");
# tmva_mumu = TMVA::Experimental::Compute<9, float>(bdt);
# ''')

# ROOT.gInterpreter.ProcessLine('''
# TMVA::Experimental::RBDT<> bdt("ZH_Recoil_BDT", "/eos/user/t/tofourni/public/FCC/FCCWorkspace/analysis/ZH/output/BDT/ee/xgb_bdt.root");
# tmva_ee = TMVA::Experimental::Compute<9, float>(bdt);
# ''')

def build_graph_ll(df, hists, dataset, final_state):

    #############################################
    ## Alias for muon and MC truth informations##
    #############################################
    if final_state == "mumu":
        df = df.Alias("Lepton0", "Muon#0.index")
    elif final_state == "ee":
        df = df.Alias("Lepton0", "Electron#0.index")
    else:
        raise ValueError(f"final_state {final_state} not supported")
    df = df.Alias("Photon0", "Photon#0.index")
    
    # photons
    df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
    df = df.Define("photons_p", "FCCAnalyses::ReconstructedParticle::get_p(photons)")
    df = df.Define("photons_theta", "FCCAnalyses::ReconstructedParticle::get_theta(photons)")
    df = df.Define("photons_phi", "FCCAnalyses::ReconstructedParticle::get_phi(photons)")
    df = df.Define("photons_no", "FCCAnalyses::ReconstructedParticle::get_n(photons)")
    
    df = df.Define("gen_photons", "HiggsTools::get_photons(Particle)")
    df = df.Define("gen_photons_p", "FCCAnalyses::MCParticle::get_p(gen_photons)")
    df = df.Define("gen_photons_theta", "FCCAnalyses::MCParticle::get_theta(gen_photons)")
    df = df.Define("gen_photons_phi", "FCCAnalyses::MCParticle::get_phi(gen_photons)")
    df = df.Define("gen_photons_no", "FCCAnalyses::MCParticle::get_n(gen_photons)")
        
    # Missing ET
    df = df.Define("cosTheta_miss", "abs(HiggsTools::get_cosTheta(MissingET))") 

    # all leptons (bare)
    df = df.Define("leps_all", "FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)")
    df = df.Define("leps_all_p", "FCCAnalyses::ReconstructedParticle::get_p(leps_all)")
    df = df.Define("leps_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps_all)")
    df = df.Define("leps_all_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps_all)")
    df = df.Define("leps_all_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps_all)")
    df = df.Define("leps_all_no", "FCCAnalyses::ReconstructedParticle::get_n(leps_all)")
    df = df.Define("leps_all_iso", "HiggsTools::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)") 
    df = df.Define("leps_all_p_gen", "HiggsTools::gen_p_from_reco(leps_all, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
    
    # cuts on leptons
    df = df.Define("leps", "FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)")
    df = df.Define("leps_p", "FCCAnalyses::ReconstructedParticle::get_p(leps)")
    df = df.Define("leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps)")
    df = df.Define("leps_phi", "FCCAnalyses::ReconstructedParticle::get_phi(leps)")
    df = df.Define("leps_q", "FCCAnalyses::ReconstructedParticle::get_charge(leps)")
    df = df.Define("leps_no", "FCCAnalyses::ReconstructedParticle::get_n(leps)")
    df = df.Define("leps_iso", "HiggsTools::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)")
    df = df.Define("leps_sel_iso", "HiggsTools::sel_isol(0.25)(leps, leps_iso)")

    # momentum resolution
    df = df.Define("leps_all_reso_p", "HiggsTools::leptonResolution_p(leps_all, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")
    df = df.Define("leps_reso_p", "HiggsTools::leptonResolution_p(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle)")

    # baseline selections and histograms
    hists.append(df.Histo1D((f"{final_state}_leps_all_p_noSel", "",     *bins_p_mu),  "leps_all_p"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_p_gen_noSel", "", *bins_p_mu),  "leps_all_p_gen"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_theta_noSel", "", *bins_theta), "leps_all_theta"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_phi_noSel", "",   *bins_phi),   "leps_all_phi"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_no_noSel", "",    *bins_count), "leps_all_no"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_iso_noSel", "",   *bins_iso),   "leps_all_iso"))

    hists.append(df.Histo1D((f"{final_state}_leps_p_noSel", "",     *bins_p_mu),  "leps_p"))
    hists.append(df.Histo1D((f"{final_state}_leps_theta_noSel", "", *bins_theta), "leps_theta"))
    hists.append(df.Histo1D((f"{final_state}_leps_phi_noSel", "",   *bins_phi),   "leps_phi"))
    hists.append(df.Histo1D((f"{final_state}_leps_no_noSel", "",    *bins_count), "leps_no"))
    hists.append(df.Histo1D((f"{final_state}_leps_iso_noSel", "",   *bins_iso),   "leps_iso"))

    #########
    ### CUT 0 all events
    #########
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut0"))

    #########
    ### CUT 1: at least a lepton with at least 1 isolated one
    #########
    df = df.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut1"))

    #########
    ### CUT 2 :at least 2 OS leptons, and build the resonance
    #########
    df = df.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut2"))

    # remove H->mumu/ee candidate leptons
    df = df.Define("zbuilder_result_Hll", f"HiggsTools::resonanceBuilder_mass_recoil2(125, 91.2, 0.4, {ecm}, false)(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("zll_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}") # the Z
    df = df.Define("zll_Hll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]")
    df = df.Define("zll_leps_Hll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1], zbuilder_result_Hll[2]}") # the leptons
    df = df.Define("zll_leps_dummy", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}") # the leptons
    df = df.Define("leps_to_remove", "return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy")
    df = df.Define("leps_good", "FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)")
    df = df.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()") 

    # build the Z resonance based on the available leptons. 
    # Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
    # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, 
    # index 1 and 2 the leptons of the pair
    df = df.Define("zbuilder_result", f"HiggsTools::resonanceBuilder_mass_recoil2(91.2, 125, 0.4, {ecm}, false)(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("zll", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}") # the Z
    df = df.Define("zll_leps", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}") # the leptons
    df = df.Define("zll_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]")
    df = df.Define("zll_p", "FCCAnalyses::ReconstructedParticle::get_p(zll)[0]")
    df = df.Define("zll_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]")
    df = df.Define("zll_phi", "FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]")

    # recoil
    df = df.Define("zll_recoil", f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)")
    df = df.Define("zll_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]")
    df = df.Define("zll_category", "HiggsTools::polarAngleCategorization(0.8, 2.34)(zll_leps)")
    
    df = df.Define("zll_leps_p", "FCCAnalyses::ReconstructedParticle::get_p(zll_leps)")
    df = df.Define("zll_leps_dR", "HiggsTools::deltaR(zll_leps)")
    df = df.Define("zll_leps_theta", "FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)")
    
    df = df.Define("prompt_muons", "HiggsTools::whizard_zh_select_prompt_leptons(zll_leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("prompt_muons_no", "prompt_muons.size()")

    # Z leptons informations
    df = df.Define("sorted_leptons", "HiggsTools::sort_greater_p(zll_leps)")
    df = df.Define("sorted_p", "FCCAnalyses::ReconstructedParticle::get_p(sorted_leptons)")
    df = df.Define("sorted_m", "FCCAnalyses::ReconstructedParticle::get_mass(sorted_leptons)")
    df = df.Define("sorted_theta", "FCCAnalyses::ReconstructedParticle::get_theta(sorted_leptons)")
    df = df.Define("sorted_phi", "FCCAnalyses::ReconstructedParticle::get_phi(sorted_leptons)")
    df = df.Define("leading_p", "return sorted_p.at(0)")
    df = df.Define("leading_m", "return sorted_m.at(0)")
    df = df.Define("leading_theta", "return sorted_theta.at(0)")
    df = df.Define("leading_phi", "return sorted_phi.at(0)")
    df = df.Define("subleading_p", "return sorted_p.at(1)")
    df = df.Define("subleading_m", "return sorted_m.at(1)")
    df = df.Define("subleading_theta", "return sorted_theta.at(1)")
    df = df.Define("subleading_phi", "return sorted_phi.at(1)")
    
    df = df.Define("zll_acolinearity", "HiggsTools::acolinearity(sorted_leptons)")
    df = df.Define("zll_acoplanarity", "HiggsTools::acoplanarity(sorted_leptons)") 
    df = df.Define("acolinearity", "if(zll_acolinearity.size()>0) return zll_acolinearity.at(0); else return -std::numeric_limits<float>::max()") 
    df = df.Define("acoplanarity", "if(zll_acoplanarity.size()>0) return zll_acoplanarity.at(0); else return -std::numeric_limits<float>::max()") 
    
    # Higgsstrahlungness
    df = df.Define("H", "HiggsTools::Higgsstrahlungness(zll_m, zll_recoil_m)")

    #########
    ### CUT 3: Z mass window
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_m_nOne", "", *bins_m_ll), "zll_m"))
    df = df.Filter("zll_m > 86 && zll_m < 96")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut3"))

    #########
    ### CUT 4: Z momentum
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_p_nOne", "", *bins_p_mu), "zll_p"))
    if ecm == 240:
        df = df.Filter("zll_p > 20 && zll_p < 70")
    if ecm == 365:
        df = df.Filter("zll_p > 50 && zll_p < 150")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut4"))

    #########
    ### CUT 5: recoil cut
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_recoil_nOne", "", *bins_recoil), "zll_recoil_m"))
    df = df.Filter("zll_recoil_m < 150 && zll_recoil_m > 100")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut5"))

    #########
    ### CUT 6: cosThetaMiss cut
    #########
    hists.append(df.Histo1D((f"{final_state}_cosThetaMiss_nOne", "", *bins_miss), "cosTheta_miss"))
    if miss:
        df = df.Filter("cosTheta_miss.size() >= 1 && cosTheta_miss[0] < 0.98")
        hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut6"))

    ##########
    ### MVA
    ##########

    vars_str = ', (float)'.join(train_vars)
    if final_state == "mumu":
        df = df.Define("MVAVec", f"ROOT::VecOps::RVec<float>{{{vars_str}}}")
        df = df.Define("mva_score", tmva_mumu.tmva_helper, ["MVAVec"])
        df = df.Define("BDTscore", "mva_score.at(0)")
    elif final_state == "ee":
        df = df.Define("MVAVec", f"ROOT::VecOps::RVec<float>{{{vars_str}}}")
        df = df.Define("mva_score", tmva_ee.tmva_helper, ["MVAVec"])
        df = df.Define("BDTscore", "mva_score.at(0)")
    hists.append(df.Histo1D((f"{final_state}_mva_score", "", *(1000, 0, 1)), "BDTscore"))

    # MVA cut
    mva_sign = 0.84 if final_state=='mumu' else 0.83

    # separate recoil plots
    df_low = df.Filter(f"BDTscore < {mva_sign}")
    hists.append(df_low.Histo1D((f"{final_state}_zll_recoil_m_mva_low", "", *(bins_recoil_fine)), "zll_recoil_m"))

    df_high = df.Filter(f"BDTscore > {mva_sign}")
    hists.append(df_high.Histo1D((f"{final_state}_zll_recoil_m_mva_high", "", *(bins_recoil_fine)), "zll_recoil_m"))

    ##########
    # Final
    ##########

    bins_mva_ = [0, mva_sign, 1]
    bins_mrec_ = list(np.arange(100, 150.5, 0.5))
    bins_mva = array.array('d', bins_mva_)
    bins_mrec = array.array('d', bins_mrec_)
    model = ROOT.RDF.TH2DModel(f"{final_state}_recoil_m_mva", "", len(bins_mrec_)-1, bins_mrec, len(bins_mva_)-1, bins_mva)
    hists.append(df.Histo2D(model, "zll_recoil_m", "BDTscore"))

    # final histograms
    hists.append(df.Histo1D((f"{final_state}_leps_p", "",     *bins_p_mu),   "leps_p"))
    hists.append(df.Histo1D((f"{final_state}_zll_p", "",      *bins_p_mu),   "zll_p"))
    hists.append(df.Histo1D((f"{final_state}_zll_m", "",      *bins_m_ll),   "zll_m"))
    hists.append(df.Histo1D((f"{final_state}_zll_recoil", "", *bins_recoil), "zll_recoil_m"))

    hists.append(df.Histo1D((f"{final_state}_cosThetaMiss", "", *bins_miss), "cosTheta_miss"))
    hists.append(df.Histo1D((f"{final_state}_acoplanarity", "", *bins_aco),  "acoplanarity"))
    hists.append(df.Histo1D((f"{final_state}_acolinearity", "", *bins_aco),  "acolinearity"))

    hists.append(df.Histo1D(("leading_p", "",        *bins_p_ll),  "leading_p"))
    hists.append(df.Histo1D(("leading_theta", "",    *bins_theta), "leading_theta"))
    hists.append(df.Histo1D(("subleading_p", "",     *bins_p_ll),  "subleading_p"))
    hists.append(df.Histo1D(("subleading_theta", "", *bins_theta), "subleading_theta"))

    return hists



def build_graph(df, dataset):

    df = df
    hists = []

    df = df.Define("weight", "1.0")
    weightsum = df.Sum("weight")

    df = df.Define("cut0", "0")
    df = df.Define("cut1", "1")
    df = df.Define("cut2", "2")
    df = df.Define("cut3", "3")
    df = df.Define("cut4", "4")
    df = df.Define("cut5", "5")
    if miss:
        df = df.Define("cut6", "6")

    df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
    df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Alias("Particle1", "Particle#1.index")

    build_graph_ll(df, hists, dataset, 'mumu')
    build_graph_ll(df, hists, dataset, 'ee')

    return hists, weightsum
