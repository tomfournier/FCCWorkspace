# import ROOT
import importlib


# Load userConfig 
userConfig = importlib.import_module("userConfig")
from userConfig import loc, get_loc, ecm, sel, param, treemaker

# Output directory where the files produced at the pre-selection level will be put
if treemaker:
    final_state = input('Select channel [ee, mumu]: ')
    outputDir = get_loc(loc.MVA_INPUTS, final_state, ecm, sel)
else:
    outputDir = get_loc(loc.HIST_MVAINPUTS, "", ecm, sel)

# include custom functions
includePaths = ["../../../../functions/functions.h"]

# Mandatory: Production tag when running over EDM4Hep centrally produced events, 
# this points to the yaml files for getting sample statistics
prodTag = "FCCee/winter2023_training/IDEA/"
# Link to the dictonary that contains all the cross section informations etc...
# path to procDict: /cvmfs/fcc.cern.ch/FCCDicts
procDict = "FCCee_procDict_winter2023_training_IDEA.json"
# Optional: output directory, default is local dir
eosType = "eosuser"
# Optional: ncpus, default is 4
nCPUS = 10
# Optional running on HTCondor, default is False
runBatch = False
# Optional batch queue name when running on HTCondor, default is workday
batchQueue = "longlunch"
# Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
compGroup = "group_u_FCC.local_gen"



# Process samples for BDT
samples_ee = [
    #signal
    f"wzp6_ee_eeH_ecm{ecm}",
    #background: 
    f"p8_ee_ZZ_ecm{ecm}", f"p8_ee_WW_ee_ecm{ecm}", 
    f"wzp6_ee_ee_Mee_30_150_ecm{ecm}",
    #rare backgrounds:
    f"wzp6_egamma_eZ_Zee_ecm{ecm}", f"wzp6_gammae_eZ_Zee_ecm{ecm}",
    f"wzp6_gaga_ee_60_ecm{ecm}"
]

samples_mumu = [
    #signal
    f"wzp6_ee_mumuH_ecm{ecm}",
    #background: 
    f"p8_ee_ZZ_ecm{ecm}", f"p8_ee_WW_mumu_ecm{ecm}", 
    f"wzp6_ee_mumu_ecm{ecm}",
    #rare backgrounds:
    f"wzp6_egamma_eZ_Zmumu_ecm{ecm}", f"wzp6_gammae_eZ_Zmumu_ecm{ecm}",
    f"wzp6_gaga_mumu_60_ecm{ecm}"
]

if treemaker:
    if   final_state=='ee':   samples_BDT = samples_ee
    elif final_state=='mumu': samples_BDT = samples_mumu
    else: raise ValueError(f"final_state {final_state} not supported")
else: samples_BDT = samples_mumu + samples_ee

# Mandatory: List of processes
processList = {i:param for i in samples_BDT}



# bins for histograms
bins_p_mu        = (2000,  0,   200) # 100 MeV bins
bins_m_ll        = (2000,  0,   200) # 100 MeV bins
bins_p_ll        = (200,   0,   200) # 1   GeV bins
bins_recoil      = (20000, 0,   200) # 10  MeV bins 
bins_recoil_fine = (500,   100, 150) # 100 MeV bins 
bins_miss        = (10000, 0,   1)

bins_theta = (400,  0, 4)
bins_phi   = (400, -4, 4)
bins_aco   = (400,  0, 4)

bins_iso   = (500, 0, 5)
bins_count = (50,  0, 50)



def build_graph_ll(df, hists, dataset, final_state):

    ################################################
    ## Alias for lepton and MC truth informations ##
    ################################################
    if final_state == "mumu":
        df = df.Alias("Lepton0", "Muon#0.index")
    elif final_state == "ee":
        df = df.Alias("Lepton0", "Electron#0.index")
    if treemaker:
        df = df.Define("cut0", "0")
        df = df.Define("cut1", "1")
        df = df.Define("cut2", "2")
        df = df.Define("cut3", "3")
        df = df.Define("cut4", "4")
        df = df.Define("cut5", "5")
        df = df.Define("cut6", "6")
        df = df.Define("cut7", "7")

        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
    df = df.Alias("Photon0",   "Photon#0.index")
    
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
    ### CUT 0: no cut
    #########
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut0"))

    #########
    ### CUT 1: at least one lepton and at least one lepton isolated (I_rel < 0.25)
    #########
    df = df.Filter("leps_no >= 1 && leps_sel_iso.size() > 0")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut1"))

    #########
    ### CUT 2: at least 2 leptons
    #########
    df = df.Filter("leps_no >= 2 && abs(Sum(leps_q)) < leps_q.size()")
    hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), "cut2"))

    # remove H->mumu/ee candidate leptons
    df = df.Define("zbuilder_result_Hll", f"FCCAnalyses::resonanceBuilder_mass_recoil(125, 91.2, 0.4, {ecm}, false)"
                    "(leps, MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles, Particle, Particle0, Particle1)")
    df = df.Define("zll_Hll",             "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[0]}") # the Z
    df = df.Define("zll_Hll_m",           "FCCAnalyses::ReconstructedParticle::get_mass(zll_Hll)[0]")
    df = df.Define("zll_leps_Hll",        "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{zbuilder_result_Hll[1],zbuilder_result_Hll[2]}") # the leptons
    df = df.Define("zll_leps_dummy",      "ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>{}") # the leptons
    df = df.Define("leps_to_remove",      "return (zll_Hll_m > (125-3) && zll_Hll_m < (125+3)) ? zll_leps_Hll : zll_leps_dummy")
    df = df.Define("leps_good",           "FCCAnalyses::ReconstructedParticle::remove(leps, leps_to_remove)") 

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
    
    # Z leptons informations
    df = df.Define("zll_leps_p",       "FCCAnalyses::ReconstructedParticle::get_p(zll_leps)")
    df = df.Define("zll_leps_theta",   "FCCAnalyses::ReconstructedParticle::get_theta(zll_leps)")
    df = df.Define("zll_leps_phi",     "FCCAnalyses::ReconstructedParticle::get_phi(zll_leps)")
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

    # Visible energy
    df = df.Define("rps_no_leps", "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, zll_leps)")
    df = df.Define("visibleEnergy", "FCCAnalyses::visibleEnergy(rps_no_leps)")

    #########
    ### CUT 3: Z mass window
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_m_nOne",      "", *bins_m_ll),   "zll_m"))
    hists.append(df.Histo1D((f"{final_state}_zll_recoil_cut3", "", *bins_recoil), "zll_recoil_m"))
    df = df.Filter("zll_m > 86 && zll_m < 96") 
    hists.append(df.Histo1D((f"{final_state}_cutFlow",    "", *bins_count), "cut3"))

    #########
    ### CUT 4: Z momentum 
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_p_nOne", "", *bins_p_mu), "zll_p"))
    hists.append(df.Histo1D((f"{final_state}_zll_recoil_cut4", "", *bins_recoil), "zll_recoil_m"))
    if ecm == 240:
        df = df.Filter("zll_p > 20 && zll_p < 70")
    elif ecm == 365:
        df = df.Filter("zll_p > 50 && zll_p < 150")
    hists.append(df.Histo1D((f"{final_state}_cutFlow",    "", *bins_count), "cut4"))

    #########
    ### CUT 4bis: p_leading between 50 GeV and 80 GeV (240GeV)
    #########
    if userConfig.leading:
        df = df.Filter("leading_p < 80 && leading_p > 50")

    #########
    ### CUT 4ter: p_subleading <  53 GeV (240GeV)
    #########
    if userConfig.leading:
        df = df.Filter("subleading_p < 53")

    #########
    ### CUT 5: recoil mass cut
    #########
    hists.append(df.Histo1D((f"{final_state}_zll_recoil_nOne", "", *bins_recoil), "zll_recoil_m"))
    if userConfig.recoil120:
        df = df.Filter("zll_recoil_m < 140 && zll_recoil_m > 120")
    else:
        df = df.Filter("zll_recoil_m < 150 && zll_recoil_m > 100")
    hists.append(df.Histo1D((f"{final_state}_cutFlow",         "", *bins_count),  "cut5"))

    ############
    ### CUT TEST
    ############
    if userConfig.sep:
        sel_vis = '(visibleEnergy > 100 && cosTheta_miss < 0.995)'
        sel_inv = '(visibleEnergy <= 100 && zll_theta < 2.85 && zll_theta > 0.25 && acoplanarity > 0.05 && cosTheta_miss < 0.998)'
        df = df.Filter(sel_vis+' || '+sel_inv)
        # df = df.Filter('leading_p < 80 && subleading_p < 53 && subleading_p > 23')

    ############
    ### CUT 5bis: visible energy cut
    ############
    hists.append(df.Histo1D((f"{final_state}_visibleEnergy_nOne", "", *bins_recoil), "zll_recoil_m"))
    if userConfig.vis:
        df = df.Filter("visibleEnergy > 10")
        hists.append(df.Histo1D((f"{final_state}_cutFlow",        "", *bins_count),  "cut6"))

    #########
    ### CUT 6: cos(theta_miss) cut
    #########
    hists.append(df.Histo1D((f"{final_state}_cosThetaMiss_nOne", "", *bins_miss), "cosTheta_miss"))
    if userConfig.miss:
        df = df.Filter("cosTheta_miss < 0.98")
        cut = 'cut7' if userConfig.vis else 'cut6'
        hists.append(df.Histo1D((f"{final_state}_cutFlow", "", *bins_count), cut))

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

    hists.append(df.Histo1D((f"{final_state}_visibleEnergy", "", *bins_p_mu), "visibleEnergy"))
    
    return df



if treemaker:

    class RDFanalysis():

        #__________________________________________________________
        # Mandatory: analysers funtion to define the analysers to process, 
        # please make sure you return the last dataframe, in this example it is df
        def analysers(df):
            df = build_graph_ll(df, [], "", final_state)
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
                "visibleEnergy", "cosTheta_miss",
                # Higgsstrahlungness
                "H",
            ]
            return branchList

else:

    def build_graph(df, dataset):

        hists = []

        df = df.Define("weight", "1.0")
        weightsum = df.Sum("weight")

        df = df.Define("cut0", "0")
        df = df.Define("cut1", "1")
        df = df.Define("cut2", "2")
        df = df.Define("cut3", "3")
        df = df.Define("cut4", "4")
        df = df.Define("cut5", "5")
        df = df.Define("cut6", "6")
        df = df.Define("cut7", "7")

        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")

        build_graph_ll(df, hists, dataset, 'mumu')
        build_graph_ll(df, hists, dataset, 'ee')

        return hists, weightsum



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
