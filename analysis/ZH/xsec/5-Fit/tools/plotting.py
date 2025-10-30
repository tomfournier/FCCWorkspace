import os, json
import numpy as np
import pandas as pd

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from . import plotter
from .bias import make_pseudosignal

h_decays_labels = {
    "bb": "H#rightarrow b#bar{b}", "cc": "H#rightarrow c#bar{c}", "ss": "H#rightarrow s#bar{s}", 
    "gg": "H#rightarrow gg", "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
    "ZZ": "H#rightarrow ZZ*", "WW": "H#rightarrow WW*", "Za": "H#rightarrow Z#gamma", 
    "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrow Inv"
}

h_decays_colors = {
    "bb": ROOT.kViolet, "cc": ROOT.kBlue , "ss": ROOT.kRed, "gg": ROOT.kGreen+1, "mumu": ROOT.kOrange, 
    "tautau": ROOT.kCyan, "ZZ": ROOT.kGray, "WW": ROOT.kGray+2, "Za": ROOT.kGreen+2, "aa": ROOT.kRed+2, 
    "inv": ROOT.kBlue+2
}

procs_labels = {
    "ZH"        : "ZH",
    "ZmumuH"    : "Z(#mu^{+}#mu^{#minus})H",
    "ZeeH"      : "Z(e^{+}e^{#minus})H",
    "WW"        : "WW",
    "ZZ"        : "ZZ",
    "Zgamma"    : "Z/#gamma^{*} #rightarrow f#bar{f}+#gamma(#gamma)",
    "Rare"      : "Rare"
}

# colors from https://github.com/mpetroff/accessible-color-cycles
procs_colors = {
    "ZH"        : ROOT.TColor.GetColor("#e42536"),
    "ZqqH"      : ROOT.TColor.GetColor("#e42536"),
    "ZmumuH"    : ROOT.TColor.GetColor("#e42536"),
    "ZnunuH"    : ROOT.TColor.GetColor("#e42536"),
    "ZeeH"      : ROOT.TColor.GetColor("#e42536"),
    "WW"        : ROOT.TColor.GetColor("#f89c20"),
    "ZZ"        : ROOT.TColor.GetColor("#5790fc"),
    "Zgamma"    : ROOT.TColor.GetColor("#964a8b"),
    "Zqqgamma"  : ROOT.TColor.GetColor("#964a8b"),
    "Rare"      : ROOT.TColor.GetColor("#9c9ca1")
}

#__________________________________________________________
def get_range(hists: list, logY=False, strict=False,
              xmin=None, xmax=None, ymin=None, ymax=None):
    
    hist = hists[0].Clone()
    if len(hists)>1:
        for h in hists[1:]: hist.Add(h)

    vals_min, vals_max = [], []
    if xmin==None:
        xmin = hist.GetBinLowEdge(0)
    if xmax==None:
        xmax = hist.GetBinLowEdge(hist.GetNbinsX()+1)
    for i in range(hist.GetNbinsX()+1):
        if  (hist.GetBinLowEdge(i) > xmin) or (hist.GetBinLowEdge(i+1) < xmax):
            if not strict:
                if hist.GetBinContent(i) !=0:
                    vals_min.append(hist.GetBinLowEdge(i))
                    vals_max.append(hist.GetBinLowEdge(i+1))
            else: 
                vals_min.append(hist.GetBinLowEdge(i))
                vals_max.append(hist.GetBinLowEdge(i+1))
    xMin, xMax = min(vals_min), max(vals_max)

    if logY:
        yMin = min( [h.GetMinimum() for h in hists] )*1e-1
        yMax = max( [h.GetMaximum() for h in hists] )*1e4
    else: 
        yMin = 0
        yMax = max( [h.GetMaximum() for h in hists] )*1.6

    if (ymin!=None) and ymin>yMin: yMin = ymin
    if (ymax!=None) and ymax<yMax: yMax = ymax

    return xMin, xMax, yMin, yMax


#__________________________________________________________
def getMetaInfo(proc, info='crossSection', remove=False,
                fcc='/cvmfs/fcc.cern.ch/FCCDicts',
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    procFile = os.path.join(fcc, '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
    if remove:
        if 'HZZ' in proc:
            xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
            xsec = getMetaInfo(proc) - xsec_inv
        if 'p8_ee_WW_ecm' in proc:
            xsec_ee   = getMetaInfo(proc.replace('WW', 'WW_ee'))
            xsec_mumu = getMetaInfo(proc.replace('WW', 'WW_mumu'))
            xsec      = getMetaInfo(proc) - xsec_ee - xsec_mumu
    return xsec

#__________________________________________________________
def getHist(hName, procs, inputDir, suffix='', rebin=1, lazy=True):
    hist = None
    for proc in procs:
        fInName = f"{inputDir}/{proc}{suffix}.root"
        if os.path.exists(fInName):
            fIn = ROOT.TFile(fInName)
        else:
            if lazy: continue
            else:
                print(f"ERROR: input file {fInName} not found")
                quit()
        h = fIn.Get(hName)
        h.SetDirectory(0)
        if 'HZZ' in proc:
            xsec_old = getMetaInfo(proc)
            xsec_new = getMetaInfo(proc, remove=True)
            h.Scale( xsec_new/xsec_old )
        if 'p8_ee_WW_ecm' in proc:
            xsec_old = getMetaInfo(proc)
            xsec_new = getMetaInfo(proc, remove=True)
            h.Scale( xsec_new/xsec_old )
        if hist == None: hist = h
        else: hist.Add(h)
        fIn.Close()
    hist.Rebin(rebin)
    return hist

#__________________________________________________________
def Bias(biasDir, nomDir, outDir, h_decays, suffix='', ecm=240, lumi=10.8, 
         outName="Bias", plot_file=['png']):

    df = pd.read_csv(f'{biasDir}/bias_results.csv')
    mode, bias = df['mode'], df['bias']*100
    unc = float(np.loadtxt(f'{nomDir}/results.txt')[-1]*1e4)

    leg = ROOT.TLegend(.2, .925-(len(h_decays)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    if unc> bias.max(): xMin, xMax = -int(unc*1.2), int(unc*1.2)
    else:               xMin, xMax = -int(bias.max()*1.2), int(bias.max()*1.2)

    In, Out = bias < unc, bias >= unc
    In, Out = int(In.astype(int).sum()), int(Out.astype(int).sum())

    h_pulls = ROOT.TH2F("pulls", "pulls", (xMax-xMin)*10, xMin, xMax, len(h_decays), 0, len(h_decays))
    g_in, g_out = ROOT.TGraphErrors(In), ROOT.TGraphErrors(Out)

    i, j = 0, 0
    for k, h_decay in enumerate(h_decays):
        b = float(bias.loc[mode==h_decay].iloc[0])
        if b < unc: 
            g_in.SetPoint(i, b, float(k) + 0.5)
            h_pulls.GetYaxis().SetBinLabel(k+1, h_decays_labels[h_decay])
            i += 1
        else:
            g_out.SetPoint(j, b, float(k) + 0.5)
            h_pulls.GetYaxis().SetBinLabel(k+1, h_decays_labels[h_decay])
            j += 1

    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000) # transparency?
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    xTitle = "Bias (#times 100) [%]"

    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitle(xTitle)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw("HIST 0")

    maxx = len(h_decays)
    lineup = ROOT.TLine(unc, 0, unc, maxx)
    lineup.SetLineColor(ROOT.kBlack)
    lineup.SetLineWidth(3)
    lineup.Draw("SAME")

    linedw = ROOT.TLine(-unc, 0, -unc, maxx)
    linedw.SetLineColor(ROOT.kBlack)
    linedw.SetLineWidth(3)
    linedw.Draw("SAME")

    line = ROOT.TLine(0, 0, 0, maxx)
    line.SetLineColor(ROOT.kGray)
    line.SetLineWidth(2)
    line.Draw("SAME")

    g_in.SetMarkerSize(1.2)
    g_in.SetMarkerStyle(20)
    g_in.SetLineWidth(2)
    g_in.SetMarkerColor(ROOT.kBlack)
    g_in.Draw('P0 SAME')

    g_out.SetMarkerSize(1.2)
    g_out.SetMarkerStyle(20)
    g_out.SetLineWidth(2)
    g_out.SetMarkerColor(ROOT.kRed)
    g_out.Draw('P0 SAME')

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(30) # 0 special vertical aligment with subscripts
    latex.DrawLatex(0.95, 0.925, f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}")

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.96, "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}")

    if not os.path.isdir(outDir):
        os.system(f'mkdir -p {outDir}')
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/bias{suffix}.{pl}")

#__________________________________________________________
def PseudoSignal(variable, inputDir, outDir, cat, target, z_decays, h_decays, pert=1.05, sel='',
                 ecm=240, lumi=10.8, outName="", plot_file=['png'], logX=False, logY=False, rebin=1, lazy=True, 
                 tot=True, proc_scales={}, density=False):

    if outName == "": outName = "Pseudo-signal"
    sigs, suffix = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays], f'_{sel}_histo'

    leg = ROOT.TLegend(.2, .925-(2/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    # leg.SetNColumns(4)

    h_tot = None
    for i, sig in enumerate(sigs):
        # h_decay = h_decays[i]
        h_sig = getHist(variable, sig, inputDir, 
                        suffix=suffix, rebin=rebin, lazy=lazy)
        h_sig.Rebin(rebin)
        # if h_sig.Integral() > 0:
        #     h_sig.Scale(1./h_sig.Integral())

        # h_sig.SetLineColor(h_decays_colors[h_decay])
        # h_sig.SetLineWidth(2)
        # h_sig.SetLineStyle(1)

        if h_tot==None: h_tot = h_sig.Clone('h_tot')
        else: h_tot.Add(h_sig)

        # leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")

    hist_pseudo = make_pseudosignal(inputDir, variable, target, cat, z_decays, h_decays, 
                                    suffix=suffix, variation=pert, tot=tot, proc_scales=proc_scales)
    hist_pseudo.Rebin(rebin)
    # if h_tot.Integral()>0:
    #     hist_pseudo.Scale(1./h_tot.Integral())
    #     h_tot.Scale(1./h_tot.Integral())
    
    hist_pseudo.SetLineColor(ROOT.kBlack)
    hist_pseudo.SetLineWidth(3)
    hist_pseudo.SetLineStyle(1)

    h_tot.SetLineColor(ROOT.kRed)
    h_tot.SetLineWidth(3)
    h_tot.SetLineStyle(1)

    leg.AddEntry(hist_pseudo, "Pseudo-signal", "L")
    leg.AddEntry(h_tot, "Signal", "L")

    xMin, xMax, yMin, yMax = get_range([h_tot, hist_pseudo], logY=logY)
    xtitle = 'm_{recoil} High [GeV]' if 'high' in sel else 'm_{recoil} Low [GeV]'
    ytitle = 'Normalized to Unity' if density else 'Events'

    cfg = {

        'logx' : logX, 'logy' : logY,
        'xmin' : xMin, 'xmax' : xMax, 'ymin' : yMin, 'ymax' : yMax,
        'xtitle' : xtitle, 'ytitle' : ytitle,
            
        'topRight' : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft' : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(1)
    dummy.Draw("HIST") 

    for hist in [h_tot, hist_pseudo]:
        hist.Draw("SAME HIST")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()
    
    out = f'{outDir}/high' if 'high' in sel else f'{outDir}/low'
    if not os.path.isdir(f'{out}/PseudoSignal'):
        os.system(f'mkdir -p {out}/PseudoSignal')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/PseudoSignal/{outName}_{target}.{pl}")
    canvas.Close()    

#__________________________________________________________
def PseudoRatio(variable, inputDir, outDir, cat, target, z_decays, h_decays, pert=1.05, sel='',
                ecm=240, lumi=10.8, outName="", plot_file=['png'], logX=False, logY=False, rebin=1, lazy=True, 
                tot=True, proc_scales={}, density=False):

    if outName == "": outName = "PseudoRatio"
    sigs, suffix = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays], f'_{sel}_histo'

    leg = ROOT.TLegend(.55, 0.9-3*0.06, .99, .80)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    # leg.SetNColumns(4)

    h_tot = None
    for sig in sigs:
        h_sig = getHist(variable, sig, inputDir, 
                        suffix=suffix, rebin=rebin, lazy=lazy)
        h_sig.Rebin(rebin)
        if h_tot==None: h_tot = h_sig.Clone('h_tot')
        else: h_tot.Add(h_sig)

    hist_pseudo = make_pseudosignal(inputDir, variable, target, cat, z_decays, h_decays, 
                                    suffix=suffix, variation=pert, tot=tot, proc_scales=proc_scales)
    hist_pseudo.Rebin(rebin)
    if density and h_tot.Integral()>0:
        hist_pseudo.Scale(1./h_tot.Integral())
        h_tot.Scale(1./h_tot.Integral())
    
    hist_pseudo.SetLineColor(ROOT.kBlack)
    hist_pseudo.SetLineWidth(3)
    hist_pseudo.SetLineStyle(1)

    h_tot.SetLineColor(ROOT.kRed)
    h_tot.SetLineWidth(3)
    h_tot.SetLineStyle(1)

    h_div = hist_pseudo.Clone('divide')
    h_div.Divide(h_tot)
    leg.AddEntry(hist_pseudo, "Pseudo-signal", "L")
    leg.AddEntry(h_tot, "Signal", "L")

    xMin, xMax, yMin, yMax = get_range([h_tot, hist_pseudo], logY=logY)
    xtitle = 'm_{recoil} High [GeV]' if 'high' in sel else 'm_{recoil} Low [GeV]'
    ytitle = 'Normalized to Unity' if density else 'Events'

    cfg = {

        'logx' : logX, 'logy' : logY,
        'xmin' : xMin, 'xmax' : xMax, 'ymin' : yMin, 'ymax' : yMax,
        'yminR' : 0.95, 'ymaxR' : pert*1.1,
        'xtitle' : xtitle, 'ytitle' : ytitle,
            
        'topRight' : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft' : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

        'ratiofraction': 0.3, 'ytitleR': 'Ratio'

    }

    plotter.cfg = cfg
    canvas, pad1, pad2   = plotter.canvasRatio()
    dummyT, dummyB, line, line1 = plotter.dummyRatio(rline1=pert)

    pad1.SetFillStyle(4000)
    pad2.SetFillStyle(4000)

    pad1.Draw()
    pad2.Draw()

    pad1.cd()
    dummyT.Draw("HIST") 
    for hist in [h_tot, hist_pseudo]:
        hist.Draw("SAME HIST")
    leg.Draw("SAME")
    pad1.SetGrid()
    pad1.Modified()
    pad1.Update()
    
    pad2.cd()
    dummyB.Draw("HIST") 
    h_div.Draw('SAME P')
    line.Draw('SAME')
    line1.Draw('SAME')
    pad2.SetGrid()
    pad2.Modified()
    pad2.Update()

    pad1.cd()
    plotter.auxRatio()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    pad1.RedrawAxis()
    pad2.RedrawAxis()
    canvas.Update()
    
    out = f'{outDir}/high' if 'high' in sel else f'{outDir}/low'
    if not os.path.isdir(f'{out}/PseudoRatio'):
        os.system(f'mkdir -p {out}/PseudoRatio')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/PseudoRatio/{outName}_{target}.{pl}")
    canvas.Close()  
