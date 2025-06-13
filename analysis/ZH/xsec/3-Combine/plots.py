import time, argparse
t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--lumi', help='Integrated luminosity in attobarns', choices=[10.8, 3.1], type=float, default=10.8)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
arg = parser.parse_args()

import importlib
userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, z_decays, h_decays

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from tools.plotting import makePlot, CutFlow, PlotDecays, CutFlowDecays, significance

ecm, lumi = arg.ecm, arg.lumi

procs_cfg = {
"ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}'  for x in z_decays for y in h_decays],
"ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
"ZeeH"      : [f'wzp6_ee_eeH_H{y}_ecm{ecm}'   for y in h_decays],
"WW"        : [f'p8_ee_WW_ecm{ecm}'],
"ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
"Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
               f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
"Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
               f'wzp6_gaga_mumu_60_ecm{ecm}',    f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
               f'wzp6_gammae_eZ_Zee_ecm{ecm}',   f'wzp6_gaga_ee_60_ecm{ecm}', 
               f'wzp6_gaga_tautau_60_ecm{ecm}',  f'wzp6_ee_nuenueZ_ecm{ecm}'],
}

for final_state in ['ee', 'mumu']:
        print(f'\n----->[Info] Making plots for {final_state} channel\n')

        sel      = select(arg.recoil120, arg.miss, arg.bdt)
        inputDir = get_loc(loc.HIST_PREPROCESSED, final_state, ecm, sel)
        outDir   = get_loc(loc.PLOTS_MEASUREMENT, final_state, ecm, sel)

        procs = [f"Z{final_state}H", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal

        cuts = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]
        lep  = '#mu' if final_state=="mumu" else "e"
        m_dw, m_up = '120' if arg.recoil120 else '100', '140' if arg.recoil120 else '150'
        cut_labels = ["All events", "#geq 1 "+lep+"^{#pm} + ISO", "#geq 2 "+lep+"^{#pm} + OS", 
                      "86 < m_{"+lep+"^{+}"+lep+"^{#minus}} < 96", "20 < p_{"+lep+"^{+}"+lep+"^{#minus}} < 70", 
                      m_dw+" < m_{rec} < "+m_up]
        if arg.miss:
                cuts.append("cut6")
                cut_labels = cut_labels.append(["|cos#theta_{miss}| < 0.98"])

        CutFlow(inputDir, outDir, procs, procs_cfg, hName=f"{final_state}_cutFlow", cuts=cuts, 
                labels=cut_labels, outName='cutflow', sig_scale=10, yMin=1e4, yMax=1e10)
        CutFlowDecays(inputDir, outDir, final_state, hName=f"{final_state}_cutFlow", outName="cutFlow", 
                      cuts=cuts, cut_labels=cut_labels, yMin=40, yMax=150, z_decays=[final_state], h_decays=h_decays)

        if False:
                significance(f"{final_state}_cosThetaMiss_nOne",     inputDir, outDir, procs, procs_cfg, 0.95, 1, reverse=True)
                significance(f"{final_state}_mva_score",             inputDir, outDir, procs, procs_cfg, 0, 0.99)
                significance(f"{final_state}_mva_score",             inputDir, outDir, procs, procs_cfg, 0, 0.99, reverse=True)
                significance(f"{final_state}_zll_p_nOne",            inputDir, outDir, procs, procs_cfg, 0, 100)
                significance(f"{final_state}_zll_p_nOne",            inputDir, outDir, procs, procs_cfg, 0, 100,  reverse=True)
                significance(f"{final_state}_zll_m_nOne",            inputDir, outDir, procs, procs_cfg, 50, 150)
                significance(f"{final_state}_zll_m_nOne",            inputDir, outDir, procs, procs_cfg, 50, 150, reverse=True)
                significance(f"{final_state}_zll_recoil_nOne",       inputDir, outDir, procs, procs_cfg, 50, 150)
                significance(f"{final_state}_zll_recoil_nOne",       inputDir, outDir, procs, procs_cfg, 50, 150, reverse=True)
                significance(f"{final_state}_leading_p",             inputDir, outDir, procs, procs_cfg, 20, 100)
                significance(f"{final_state}_leading_p",             inputDir, outDir, procs, procs_cfg, 20, 100, reverse=True)
                significance(f"{final_state}_subleading_p",          inputDir, outDir, procs, procs_cfg, 20, 100)
                significance(f"{final_state}_subleading_p",          inputDir, outDir, procs, procs_cfg, 20, 100, reverse=True)
                significance(f"{final_state}_leading_theta",         inputDir, outDir, procs, procs_cfg, 0, 3.2)
                significance(f"{final_state}_leading_theta",         inputDir, outDir, procs, procs_cfg, 0, 3.2,  reverse=True)
                significance(f"{final_state}_subleading_theta",      inputDir, outDir, procs, procs_cfg, 0, 3.2)
                significance(f"{final_state}_subleading_theta",      inputDir, outDir, procs, procs_cfg, 0, 3.2,  reverse=True)
                significance(f"{final_state}_leps_all_p_noSel",      inputDir, outDir, procs, procs_cfg, 20, 100)
                significance(f"{final_state}_leps_all_p_noSel",      inputDir, outDir, procs, procs_cfg, 20, 100, reverse=True)
                significance(f"{final_state}_leps_all_theta_noSel",  inputDir, outDir, procs, procs_cfg, 0, 3.2)
                significance(f"{final_state}_leps_all_theta_noSel",  inputDir, outDir, procs, procs_cfg, 0, 3.2,  reverse=True)
                significance(f"{final_state}_leps_p",                inputDir, outDir, procs, procs_cfg, 20, 100)
                significance(f"{final_state}_leps_p",                inputDir, outDir, procs, procs_cfg, 20, 100, reverse=True)

        PlotDecays(f"{final_state}_zll_m_nOne",            inputDir, outDir, [final_state], h_decays, outName="zll_m_nOne", 
                xMin=15, xMax=130, yMin=1e-5, yMax=1, xLabel="m_{ll} [GeV]", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_zll_p_nOne",            inputDir, outDir, [final_state], h_decays, outName="zll_p_nOne", 
                xMin=0, xMax=80, yMin=1e-5, yMax=1, xLabel="p_{ll} [GeV]", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_cosThetaMiss_nOne",     inputDir, outDir, [final_state], h_decays, outName="cosThetaMiss_nOne", 
                xMin=0.9, xMax=1, yMin=1e-5, yMax=1e1, xLabel="|cos#theta_{miss}|", yLabel="Events", logY=True, rebin=8)
        PlotDecays(f"{final_state}_zll_recoil_m_mva_low",  inputDir, outDir, [final_state], h_decays, outName="zll_recoil_m_mva_low", 
                xMin=int(m_dw), xMax=int(m_up), yMin=1e-5, yMax=1, xLabel="Recoil [GeV]", yLabel="Events", logY=True, rebin=2)
        PlotDecays(f"{final_state}_zll_recoil_m_mva_high", inputDir, outDir, [final_state], h_decays, outName="zll_recoil_m_mva_high", 
                xMin=122, xMax=134, yMin=1e-5, yMax=1e1, xLabel="Recoil [GeV]", yLabel="Events", logY=True, rebin=1)
        PlotDecays(f"{final_state}_mva_score",             inputDir, outDir, [final_state], h_decays, outName="mva_score", 
                xMin=0, xMax=1, yMin=1e-4, yMax=1, xLabel="MVA score", yLabel="Events", logY=True, rebin=10)
        
        makePlot(f"{final_state}_zll_m_nOne",            inputDir, outDir, procs, procs_cfg, outName="zll_m_nOne", 
                xMin=50, xMax=120, yMin=1e2, yMax=1e8, xLabel="m_{ll} [GeV]", yLabel="Events", logY=True, rebin=1)
        makePlot(f"{final_state}_zll_p_nOne",            inputDir, outDir, procs, procs_cfg, outName="zll_p_nOne", 
                xMin=0, xMax=120, yMin=1e1, yMax=1e8, xLabel="p_{ll} [GeV]", yLabel="Events", logY=True, rebin=2)
        makePlot(f"{final_state}_cosThetaMiss_nOne",     inputDir, outDir, procs, procs_cfg, outName="cosThetaMiss_nOne", 
                xMin=0.9, xMax=1, yMin=1e1, yMax=1e7, xLabel="|cos#theta_{miss}|", yLabel="Events", logY=True, rebin=8)
        makePlot(f"{final_state}_zll_recoil_m_mva_low",  inputDir, outDir, procs, procs_cfg, outName="zll_recoil_m_mva_low", 
                xMin=int(m_dw), xMax=int(m_up), yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=2)
        makePlot(f"{final_state}_zll_recoil_m_mva_high", inputDir, outDir, procs, procs_cfg, outName="zll_recoil_m_mva_high", 
                xMin=122, xMax=134, yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=1)
        makePlot(f"{final_state}_zll_recoil_nOne",       inputDir, outDir, procs, procs_cfg, outName="zll_recoil_nOne", 
                xMin=100, xMax=150, yMin=1, yMax=1e6, xLabel="Recoil [GeV]", yLabel="Events", logY=True, rebin=8)
        

        for ind in ['', '_low', '_high']:
                PlotDecays(f"{final_state}_zll_m{ind}",                 inputDir, outDir, [final_state], h_decays, outName="zll_m", 
                        xMin=86, xMax=96, yMin=1e-3, yMax=1, xLabel="m_{ll} [GeV]", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_zll_p{ind}",                 inputDir, outDir, [final_state], h_decays, outName="zll_p", 
                        xMin=20, xMax=70, yMin=1e-5, yMax=1, xLabel="p_{ll} [GeV]", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_zll_recoil{ind}",            inputDir, outDir, [final_state], h_decays, outName="zll_recoil", 
                        xMin=int(m_dw), xMax=int(m_up), yMin=1e-5, yMax=1, xLabel="Recoil [GeV]", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_acoplanarity{ind}",          inputDir, outDir, [final_state], h_decays, outName="acoplanarity", 
                        xMin=0, xMax=3.2, yMin=1e-5, yMax=1, xLabel="#pi-#Delta#phi_{ll}", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_acolinearity{ind}",          inputDir, outDir, [final_state], h_decays, outName="acolinearity", 
                        xMin=0, xMax=3, yMin=1e-5, yMax=1, xLabel="#Delta#theta_{ll}", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_leading_p{ind}",             inputDir, outDir, [final_state], h_decays, outName="leading_p", 
                        xMin=40, xMax=90, yMin=1e-5, yMax=10, xLabel="p_{l,leading} [GeV]", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_subleading_p{ind}",          inputDir, outDir, [final_state], h_decays, outName="subleading_p", 
                        xMin=20, xMax=60, yMin=1e-5, yMax=1e1, xLabel="p_{l,subleading} [GeV]", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_leading_theta{ind}",         inputDir, outDir, [final_state], h_decays, outName="leading_theta", 
                        xMin=0, xMax=3.2, yMin=1e-5, yMax=1, xLabel="#theta_{l,leading}", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_subleading_theta{ind}",      inputDir, outDir, [final_state], h_decays, outName="subleading_theta", 
                        xMin=0, xMax=3.2, yMin=1e-5, yMax=1, xLabel="#theta_{l,subleading}", yLabel="Events", logY=True)
                PlotDecays(f"{final_state}_leps_p{ind}",                inputDir, outDir, [final_state], h_decays, outName="leps_p", 
                        xMin=10, xMax=90, yMin=1e-5, yMax=1, xLabel="p_{leptons} [GeV]", yLabel="Events", logY=True)
                

                makePlot(f"{final_state}_zll_recoil{ind}",            inputDir, outDir, procs, procs_cfg, outName="zll_recoil", 
                        xMin=int(m_dw), xMax=int(m_up), yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=16)
                makePlot(f"{final_state}_acoplanarity{ind}",          inputDir, outDir, procs, procs_cfg, outName="acoplanarity", 
                        xMin=0, xMax=3.2, yMin=1e-2, yMax=1e7, xLabel="#pi-#Delta#phi_{ll}", yLabel="Events", logY=True)
                makePlot(f"{final_state}_acolinearity{ind}",          inputDir, outDir, procs, procs_cfg, outName="acolinearity", 
                        xMin=0, xMax=3, yMin=1e-2, yMax=1e7, xLabel="#Delta#theta_{ll}", yLabel="Events", logY=True)
                makePlot(f"{final_state}_leading_p{ind}",             inputDir, outDir, procs, procs_cfg, outName="leading_p", 
                        xMin=40, xMax=100, yMin=1, yMax=-1, xLabel="p_{l,leading} [GeV]", yLabel="Events", logY=True, rebin=4)
                makePlot(f"{final_state}_subleading_p{ind}",          inputDir, outDir, procs, procs_cfg, outName="subleading_p", 
                        xMin=10, xMax=70, yMin=1, yMax=-1, xLabel="p_{l,subleading} [GeV]", yLabel="Events", logY=True, rebin=4)
                makePlot(f"{final_state}_leading_theta{ind}",         inputDir, outDir, procs, procs_cfg, outName="leading_theta", 
                        xMin=0, xMax=3.2, yMin=1e1, yMax=1e7, xLabel="#theta_{l,leading}", yLabel="Events", logY=True, rebin=4)
                makePlot(f"{final_state}_subleading_theta{ind}",      inputDir, outDir, procs, procs_cfg, outName="subleading_theta", 
                        xMin=0, xMax=3.2, yMin=1e1, yMax=1e6, xLabel="#theta_{l,subleading}", yLabel="Events", logY=True, rebin=4)
                makePlot(f"{final_state}_leps_p{ind}",                inputDir, outDir, procs, procs_cfg, outName="leps_p", 
                        xMin=20, xMax=100, yMin=1e-2, yMax=1e9, xLabel="p_{leptons} [GeV]", yLabel="Events", logY=True, rebin=4)


print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')