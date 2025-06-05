import ROOT
import array
import importlib
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', choices=[10.8, 3.1], type=float, default=10.8)
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--param', help='Select the fraction of the samples or the number of files to put the samples', choices=['frac', 'chunks'], default='frac')
arg = parser.parse_args()

ROOT.TH1.SetDefaultSumw2(ROOT.kTRUE)
from addons.TMVAHelper.TMVAHelper import TMVAHelperXGB # type: ignore

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, train_vars, frac, nb, z_decays, h_decays

ecm, sel = arg.ecm, select(arg.recoil120, arg.miss, arg.bdt)

# Output directory where the files produced at the pre-selection level will be put
outputDir = get_loc(loc.HIST_PREPROCESSED, '', ecm, sel)

# include custom functions
includePaths = ["../../../../functions/functions.h"]

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = "FCCee/winter2023/IDEA/"
# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = "FCCee_procDict_winter2023_IDEA.json"
# optional: ncpus, default is 4, -1 uses all cores available
nCPUS = 20
# scale the histograms with the cross-section and integrated luminosity
doScale = True
intLumi = arg.lumi * 1e6 # in pb-1

# Process samples
samples_bkg = [
    f"p8_ee_WW_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
    f"wzp6_ee_ee_Mee_30_150_ecm{ecm}", f"wzp6_ee_mumu_ecm{ecm}", f"wzp6_ee_tautau_ecm{ecm}",
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}', f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f"wzp6_gaga_ee_60_ecm{ecm}", f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f"wzp6_ee_nuenueZ_ecm{ecm}"
]
samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
for i in ['ee', 'mumu']:
    samples_sig.append(f"wzp6_ee_{i}H_ecm{ecm}")
samples_sig.append(f'wzp6_ee_ZH_Hinv_ecm{ecm}')

samples = samples_sig + samples_bkg

if arg.param=='frac':     param = {'fraction': frac} 
elif arg.param=='chunks': param = {'chunks':   nb}
processList = {i:param for i in samples}



# define histograms
bins_p_mu        = (2000,  0,   200) # 100 MeV bins
bins_m_ll        = (2000,  0,   200) # 100 MeV bins
bins_p_ll        = (200,   0,   200) # 1 GeV bins
bins_recoil      = (20000, 0,   200) # 10 MeV bins 
bins_recoil_fine = (500,   100, 150) # 100 MeV bins 
bins_miss        = (10000, 0,   1)

bins_theta = (400,  0, 4)
bins_phi   = (400, -4, 4)
bins_aco   = (400,  0, 4)

bins_iso   = (500, 0, 5)
bins_count = (50,  0, 50)

vars_list = train_vars.copy()
if arg.bdt: vars_list.append("cosTheta_miss")

ROOT.EnableImplicitMT(nCPUS) # hack to deal correctly with TMVAHelperXGB
tmva_mumu = TMVAHelperXGB(f"{loc.MVA}/{ecm}/mumu/{sel}/BDT/xgb_bdt.root", "ZH_Recoil_BDT", variables=vars_list)
tmva_ee   = TMVAHelperXGB(f"{loc.MVA}/{ecm}/ee/{sel}/BDT/xgb_bdt.root", "ZH_Recoil_BDT",   variables=vars_list)

mva_mumu = float(np.loadtxt(f"{loc.MVA}/{ecm}/mumu/{sel}/BDT/BDT_cut.txt"))
mva_ee   = float(np.loadtxt(f"{loc.MVA}/{ecm}/ee/{sel}/BDT/BDT_cut.txt"))

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
     
    # Missing ET
    df = df.Define("missingEnergy", f"FCCAnalyses::missingEnergy({ecm}, ReconstructedParticles)")
    df = df.Define("cosTheta_miss", "FCCAnalyses::get_cosTheta_miss(missingEnergy)")

    # all leptons (bare)
    df = df.Define("leps_all",       "FCCAnalyses::ReconstructedParticle::get(Lepton0, ReconstructedParticles)")
    df = df.Define("leps_all_p",     "FCCAnalyses::ReconstructedParticle::get_p(leps_all)")
    df = df.Define("leps_all_theta", "FCCAnalyses::ReconstructedParticle::get_theta(leps_all)")
    df = df.Define("leps_all_phi",   "FCCAnalyses::ReconstructedParticle::get_phi(leps_all)")
    df = df.Define("leps_all_q",     "FCCAnalyses::ReconstructedParticle::get_charge(leps_all)")
    df = df.Define("leps_all_no",    "FCCAnalyses::ReconstructedParticle::get_n(leps_all)")
    df = df.Define("leps_all_iso",   "FCCAnalyses::coneIsolation(0.01, 0.5)(leps_all, ReconstructedParticles)") 
    
    # cuts on leptons
    df = df.Define("leps",         "FCCAnalyses::ReconstructedParticle::sel_p(20)(leps_all)")
    df = df.Define("leps_p",       "FCCAnalyses::ReconstructedParticle::get_p(leps)")
    df = df.Define("leps_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(leps)")
    df = df.Define("leps_phi",     "FCCAnalyses::ReconstructedParticle::get_phi(leps)")
    df = df.Define("leps_q",       "FCCAnalyses::ReconstructedParticle::get_charge(leps)")
    df = df.Define("leps_no",      "FCCAnalyses::ReconstructedParticle::get_n(leps)")
    df = df.Define("leps_iso",     "FCCAnalyses::coneIsolation(0.01, 0.5)(leps, ReconstructedParticles)")
    df = df.Define("leps_sel_iso", "FCCAnalyses::sel_isol(0.25)(leps, leps_iso)")

    # baseline selections and histograms
    hists.append(df.Histo1D((f"{final_state}_leps_all_p_noSel",     "", *bins_p_mu),  "leps_all_p"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_theta_noSel", "", *bins_theta), "leps_all_theta"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_phi_noSel",   "", *bins_phi),   "leps_all_phi"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_no_noSel",    "", *bins_count), "leps_all_no"))
    hists.append(df.Histo1D((f"{final_state}_leps_all_iso_noSel",   "", *bins_iso),   "leps_all_iso"))

    hists.append(df.Histo1D((f"{final_state}_leps_p_noSel",     "", *bins_p_mu),  "leps_p"))
    hists.append(df.Histo1D((f"{final_state}_leps_theta_noSel", "", *bins_theta), "leps_theta"))
    hists.append(df.Histo1D((f"{final_state}_leps_phi_noSel",   "", *bins_phi),   "leps_phi"))
    hists.append(df.Histo1D((f"{final_state}_leps_no_noSel",    "", *bins_count), "leps_no"))
    hists.append(df.Histo1D((f"{final_state}_leps_iso_noSel",   "", *bins_iso),   "leps_iso"))

    #########
    ### CUT 0 all events
    #########
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut0"))

    #########
    ### CUT 1: at least a lepton and at least 1 one lepton isolated (I_rel < 0.25)
    #########
    df = df.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut1"))

    #########
    ### CUT 2 :at least 2 OS leptons, and build the resonance
    #########
    df = df.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut2"))

    # remove H->mumu/ee candidate leptons
    df = df.Define("zbuilder_Hll",   f"FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)"
                   "(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("zll_Hll",        "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[0]}") # the Z
    df = df.Define("zll_Hll_m",      "FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]")
    df = df.Define("zll_leps_Hll",   "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_Hll[1], zbuilder_Hll[2]}") # the leptons
    df = df.Define("zll_leps_dummy", "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}") # the leptons
    df = df.Define("leps_to_remove", "return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy")
    df = df.Define("leps_good",      "FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)")

    # build the Z resonance based on the available leptons. 
    # Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
    # technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, 
    # index 1 and 2 the leptons of the pair
    df = df.Define("zbuilder_result", f"FCCAnalyses::resonanceBuilder_mass_recoil(91.2, 125, 0.4, {ecm}, false)"
                   "(leps_good, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("zll",             "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[0]}") # the Z
    df = df.Define("zll_leps",        "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result[1],zbuilder_result[2]}") # the leptons
    df = df.Define("zll_m",           "FCCAnalyses::ReconstructedParticle::get_mass(zll)[0]")
    df = df.Define("zll_p",           "FCCAnalyses::ReconstructedParticle::get_p(zll)[0]")
    df = df.Define("zll_theta",       "FCCAnalyses::ReconstructedParticle::get_theta(zll)[0]")
    df = df.Define("zll_phi",         "FCCAnalyses::ReconstructedParticle::get_phi(zll)[0]")

    # recoil
    df = df.Define("zll_recoil",   f"FCCAnalyses::ReconstructedParticle::recoilBuilder({ecm})(zll)")
    df = df.Define("zll_recoil_m", "FCCAnalyses::ReconstructedParticle::get_mass(zll_recoil)[0]")
    df = df.Define("zll_category", "FCCAnalyses::polarAngleCategorization(0.8, 2.34)(zll_leps)")

    # Z leptons informations
    df = df.Define("zll_leps_p",       "FCCAnalyses::ReconstructedParticle::get_p(zll_leps)")
    df = df.Define("zll_leps_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)")
    df = df.Define("zll_leps_phi",     "FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)")
    df = df.Define("zll_leps_dR",      "FCCAnalyses::deltaR(zll_leps)")
    df = df.Define("leading_p_idx",    "(zll_leps_p[0] > zll_leps_p[1]) ? 0 : 1")
    df = df.Define("subleading_p_idx", "(zll_leps_p[0] > zll_leps_p[1]) ? 1 : 0")
    df = df.Define("leading_p",        "zll_leps_p[leading_p_idx]")
    df = df.Define("subleading_p",     "zll_leps_p[subleading_p_idx]")
    df = df.Define("leading_theta",    "zll_leps_theta[leading_p_idx]")
    df = df.Define("subleading_theta", "zll_leps_theta[subleading_p_idx]")
    df = df.Define("leading_phi",      "zll_leps_phi[leading_p_idx]")
    df = df.Define("subleading_phi",   "zll_leps_phi[subleading_p_idx]")
    
    df = df.Define("acoplanarity", "FCCAnalyses::acoplanarity(zll_leps)")
    df = df.Define("acolinearity", "FCCAnalyses::acolinearity(zll_leps)")

    # Higgsstrahlungness
    df = df.Define("H", "FCCAnalyses::Higgsstrahlungness(zll_m, zll_recoil_m)")

    #########
    ### CUT 3: Z mass window
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_m_nOne", "", *bins_m_ll),  "zll_m"))
    df = df.Filter("zll_m > 86 && zll_m < 96")
    hists.append(df.Histo1D((f"{final_state}_cutFlow",    "", *bins_count), "cut3"))

    #########
    ### CUT 4: Z momentum
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_p_nOne", "", *bins_p_mu), "zll_p"))
    if ecm == 240:
        df = df.Filter("zll_p > 20 && zll_p < 70")
    if ecm == 365:
        df = df.Filter("zll_p > 50 && zll_p < 150")
    hists.append(df.Histo1D((f"{final_state}_cutFlow",    "", *bins_count), "cut4"))

    #########
    ### CUT 5: recoil cut
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_recoil_nOne", "", *bins_recoil), "zll_recoil_m"))
    if arg.recoil120:
        df = df.Filter(f"zll_recoil_m < 140 && zll_recoil_m > 120")
    else:
        df = df.Filter(f"zll_recoil_m < 150 && zll_recoil_m > 100")
    hists.append(df.Histo1D((f"{final_state}_cutFlow",         "", *bins_count),  "cut5"))

    #########
    ### CUT 6: cosThetaMiss cut
    #########
    hists.append(df.Histo1D((f"{final_state}_cosThetaMiss_nOne", "", *bins_miss), "cosTheta_miss"))
    if arg.miss:
        df = df.Filter("cosTheta_miss < 0.98")
        hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut6"))

    ##########
    ### MVA
    ##########

    vars_str = ', (float)'.join(train_vars)
    if final_state == "mumu":
        df = df.Define("MVAVec",    f"ROOT::VecOps::RVec<float>{{{vars_str}}}")
        df = df.Define("mva_score", tmva_mumu.tmva_helper, ["MVAVec"])
        df = df.Define("BDTscore",  "mva_score.at(0)")
    elif final_state == "ee":
        df = df.Define("MVAVec",    f"ROOT::VecOps::RVec<float>{{{vars_str}}}")
        df = df.Define("mva_score", tmva_ee.tmva_helper, ["MVAVec"])
        df = df.Define("BDTscore",  "mva_score.at(0)")
    hists.append(df.Histo1D((f"{final_state}_mva_score", "", *(1000, 0, 1)), "BDTscore"))

    # MVA cut
    mva_sign = mva_mumu if final_state=='mumu' else mva_ee

    # separate recoil plots
    df_low = df.Filter(f"BDTscore < {mva_sign}")
    hists.append(df_low.Histo1D((f"{final_state}_zll_recoil_m_mva_low",   "", *(bins_recoil_fine)), "zll_recoil_m"))

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
    hists.append(df.Histo1D((f"{final_state}_leps_p",     "", *bins_p_mu),   "leps_p"))
    hists.append(df.Histo1D((f"{final_state}_zll_p",      "", *bins_p_mu),   "zll_p"))
    hists.append(df.Histo1D((f"{final_state}_zll_m",      "", *bins_m_ll),   "zll_m"))
    hists.append(df.Histo1D((f"{final_state}_zll_recoil", "", *bins_recoil), "zll_recoil_m"))

    hists.append(df.Histo1D((f"{final_state}_cosThetaMiss", "", *bins_miss), "cosTheta_miss"))
    hists.append(df.Histo1D((f"{final_state}_acoplanarity", "", *bins_aco),  "acoplanarity"))
    hists.append(df.Histo1D((f"{final_state}_acolinearity", "", *bins_aco),  "acolinearity"))

    hists.append(df.Histo1D((f"{final_state}_leading_p",        "", *bins_p_mu),  "leading_p"))
    hists.append(df.Histo1D((f"{final_state}_leading_theta",    "", *bins_theta), "leading_theta"))
    hists.append(df.Histo1D((f"{final_state}_subleading_p",     "", *bins_p_mu),  "subleading_p"))
    hists.append(df.Histo1D((f"{final_state}_subleading_theta", "", *bins_theta), "subleading_theta"))

    # final histograms for high mass events
    hists.append(df_high.Histo1D((f"{final_state}_leps_p_high",     "", *bins_p_mu),   "leps_p"))
    hists.append(df_high.Histo1D((f"{final_state}_zll_p_high",      "", *bins_p_mu),   "zll_p"))
    hists.append(df_high.Histo1D((f"{final_state}_zll_m_high",      "", *bins_m_ll),   "zll_m"))
    hists.append(df_high.Histo1D((f"{final_state}_zll_recoil_high", "", *bins_recoil), "zll_recoil_m"))

    hists.append(df_high.Histo1D((f"{final_state}_cosThetaMiss_high", "", *bins_miss), "cosTheta_miss"))
    hists.append(df_high.Histo1D((f"{final_state}_acoplanarity_high", "", *bins_aco),  "acoplanarity"))
    hists.append(df_high.Histo1D((f"{final_state}_acolinearity_high", "", *bins_aco),  "acolinearity"))

    hists.append(df_high.Histo1D((f"{final_state}_leading_p_high",        "", *bins_p_mu),  "leading_p"))
    hists.append(df_high.Histo1D((f"{final_state}_leading_theta_high",    "", *bins_theta), "leading_theta"))
    hists.append(df_high.Histo1D((f"{final_state}_subleading_p_high",     "", *bins_p_mu),  "subleading_p"))
    hists.append(df_high.Histo1D((f"{final_state}_subleading_theta_high", "", *bins_theta), "subleading_theta"))

    # final histograms for high mass events
    hists.append(df_low.Histo1D((f"{final_state}_leps_p_low",     "", *bins_p_mu),   "leps_p"))
    hists.append(df_low.Histo1D((f"{final_state}_zll_p_low",      "", *bins_p_mu),   "zll_p"))
    hists.append(df_low.Histo1D((f"{final_state}_zll_m_low",      "", *bins_m_ll),   "zll_m"))
    hists.append(df_low.Histo1D((f"{final_state}_zll_recoil_low", "", *bins_recoil), "zll_recoil_m"))

    hists.append(df_low.Histo1D((f"{final_state}_cosThetaMiss_low", "", *bins_miss), "cosTheta_miss"))
    hists.append(df_low.Histo1D((f"{final_state}_acoplanarity_low", "", *bins_aco),  "acoplanarity"))
    hists.append(df_low.Histo1D((f"{final_state}_acolinearity_low", "", *bins_aco),  "acolinearity"))

    hists.append(df_low.Histo1D((f"{final_state}_leading_p_low",        "", *bins_p_mu),  "leading_p"))
    hists.append(df_low.Histo1D((f"{final_state}_leading_theta_low",    "", *bins_theta), "leading_theta"))
    hists.append(df_low.Histo1D((f"{final_state}_subleading_p_low",     "", *bins_p_mu),  "subleading_p"))
    hists.append(df_low.Histo1D((f"{final_state}_subleading_theta_low", "", *bins_theta), "subleading_theta"))

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
    if arg.miss:
        df = df.Define("cut6", "6")

    df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
    df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Alias("Particle1", "Particle#1.index")

    build_graph_ll(df, hists, dataset, 'mumu')
    build_graph_ll(df, hists, dataset, 'ee')

    return hists, weightsum
