import time
t1 = time.time()

import importlib
userConfig = importlib.import_module('userConfig')
from userConfig import loc, intLumi, procs_cfg, z_decays, h_decays, miss, recoil_120

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from tools.plotting import makePlot, CutFlow, PlotDecays, CutFlowDecays, significance

lumi = intLumi

for final_state in ['ee', 'mumu']:
        print(f'\n----->[Info] Making plots for {final_state} channel\n')

        inputDir = loc.HIST_PREPROCESSED
        outDir   = loc.PLOTS_MEASUREMENT.replace('final_state',final_state)

        procs = [f"Z{final_state}H", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal

        cuts = ["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]
        lep = '#mu' if final_state=="mumu" else "e"
        m_dw, m_up = '120' if recoil_120 else '100', '140' if recoil_120 else '150'
        cut_labels = ["All events", "#geq 1 "+lep+"^{#pm} + ISO", "#geq 2 "+lep+"^{#pm} + OS", 
                      "86 < m_{"+lep+"^{+}"+lep+"^{#minus}} < 96", "20 < p_{"+lep+"^{+}"+lep+"^{#minus}} < 70", 
                      m_dw+" < m_{rec} < "+m_up]
        if miss:
                cuts.append("cut6")
                cut_labels = cut_labels.append(["|cos#theta_{miss}| < 0.98"])

        # Cutflow without cos(theta_miss) cut
        CutFlow(inputDir, outDir, procs, procs_cfg, hName=f"{final_state}_cutFlow", cuts=cuts, 
                labels=cut_labels, outName='cutflow', sig_scale=10, yMin=1e4, yMax=1e10)
        CutFlowDecays(inputDir, outDir, final_state, hName=f"{final_state}_cutFlow", outName="cutFlow", 
                      cuts=cuts, cut_labels=cut_labels, yMin=40, yMax=150, z_decays=[final_state], h_decays=h_decays)

        if True:
                significance(f"{final_state}_cosThetaMiss_nOne", inputDir, outDir, procs, procs_cfg, 0.95, 1, reverse=True)
                significance(f"{final_state}_mva_score",         inputDir, outDir, procs, procs_cfg, 0, 0.99)
                significance(f"{final_state}_mva_score",         inputDir, outDir, procs, procs_cfg, 0, 0.99, reverse=True)
                significance(f"{final_state}_zll_p_nOne",        inputDir, outDir, procs, procs_cfg, 0, 100)
                significance(f"{final_state}_zll_p_nOne",        inputDir, outDir, procs, procs_cfg, 0, 100, reverse=True)
                significance(f"{final_state}_zll_m_nOne",        inputDir, outDir, procs, procs_cfg, 50, 150)
                significance(f"{final_state}_zll_m_nOne",        inputDir, outDir, procs, procs_cfg, 50, 150, reverse=True)
                significance(f"{final_state}_zll_recoil_nOne",   inputDir, outDir, procs, procs_cfg, 50, 150)
                significance(f"{final_state}_zll_recoil_nOne",   inputDir, outDir, procs, procs_cfg, 50, 150, reverse=True)

        PlotDecays(f"{final_state}_zll_m_nOne",        inputDir, outDir, [final_state], h_decays, outName="zll_m_nOne", 
                   xMin=15, xMax=130, yMin=1e-5, yMax=1, xLabel="m_{ll} [GeV]", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_zll_p_nOne",        inputDir, outDir, [final_state], h_decays, outName="zll_p_nOne", 
                   xMin=0, xMax=100, yMin=1e-5, yMax=1, xLabel="p_{ll} [GeV]", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_zll_recoil",        inputDir, outDir, [final_state], h_decays, outName="zll_recoil", 
                   xMin=120, xMax=140, yMin=1e-5, yMax=1, xLabel="Recoil [GeV]", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_cosThetaMiss_nOne", inputDir, outDir, [final_state], h_decays, outName="cosThetaMiss_nOne", 
                   xMin=0.9, xMax=1, yMin=1e-5, yMax=1e1, xLabel="|cos#theta_{miss}|", yLabel="Events", logY=True, rebin=8)
        PlotDecays(f"{final_state}_mva_score",         inputDir, outDir, [final_state], h_decays, outName="mva_score", 
                   xMin=0, xMax=1, yMin=1e-4, yMax=1, xLabel="MVA score", yLabel="Events", logY=True, rebin=10)
        PlotDecays(f"{final_state}_acoplanarity",      inputDir, outDir, [final_state], h_decays, outName="acoplanarity", 
                   xMin=0, xMax=3.2, yMin=1e-5, yMax=1, xLabel="#pi-#Delta#phi_{ll}", yLabel="Events", logY=True)
        PlotDecays(f"{final_state}_acolinearity",      inputDir, outDir, [final_state], h_decays, outName="acolinearity", 
                   xMin=0, xMax=3, yMin=1e-5, yMax=1, xLabel="#Delta#theta_{ll}", yLabel="Events", logY=True)

        makePlot(f"{final_state}_zll_m_nOne", inputDir, outDir, procs, procs_cfg, outName="zll_m_nOne", 
                 xMin=50, xMax=120, yMin=1e2, yMax=1e8, xLabel="m_{ll} [GeV]", yLabel="Events", logY=True, rebin=1)
        makePlot(f"{final_state}_zll_p_nOne", inputDir, outDir, procs, procs_cfg, outName="zll_p_nOne", 
                 xMin=0, xMax=120, yMin=1e1, yMax=1e8, xLabel="p_{ll} [GeV]", yLabel="Events", logY=True, rebin=2)
        makePlot(f"{final_state}_zll_recoil_nOne", inputDir, outDir, procs, procs_cfg, outName="zll_recoil_nOne", 
                 xMin=100, xMax=150, yMin=1, yMax=1e6, xLabel="Recoil [GeV]", yLabel="Events", logY=True, rebin=8)
        makePlot(f"{final_state}_zll_recoil", inputDir, outDir, procs, procs_cfg, outName="zll_recoil", 
                 xMin=120, xMax=130, yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=8)
        makePlot(f"{final_state}_mva_score", inputDir, outDir, procs, procs_cfg, outName="mva_score", 
                 xMin=0, xMax=1, yMin=1e0, yMax=1e5, xLabel="MVA score", yLabel="Events", logY=True, rebin=5)
        makePlot(f"{final_state}_zll_recoil_m_mva_low", inputDir, outDir, procs, procs_cfg, outName="zll_recoil_m_mva_low", 
                 xMin=100, xMax=150, yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=1)
        makePlot(f"{final_state}_zll_recoil_m_mva_high", inputDir, outDir, procs, procs_cfg, outName="zll_recoil_m_mva_high", 
                 xMin=122, xMax=130, yMin=0, yMax=-1, xLabel="Recoil [GeV]", yLabel="Events", logY=False, rebin=1)

t2 = time.time()

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {t2-t1:.1f} s')
print('\n------------------------------------\n\n')