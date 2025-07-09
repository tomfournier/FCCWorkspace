import sys
import os
import math
import copy
import array
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

from . import plotter

h_decays_labels = {
    "bb": "H#rightarrowb#bar{b}", "cc": "H#rightarrowc#bar{c}", "ss": "H#rightarrows#bar{s}", 
    "gg": "H#rightarrowgg", "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
    "ZZ": "H#rightarrowZZ*", "WW": "H#rightarrowWW*", "Za": "H#rightarrowZ#gamma", 
    "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrowInv"}

h_decays_colors = {
    "bb": ROOT.kBlack, "cc": ROOT.kBlue , "ss": ROOT.kRed, "gg": ROOT.kGreen+1, "mumu": ROOT.kOrange, 
    "tautau": ROOT.kCyan, "ZZ": ROOT.kGray, "WW": ROOT.kGray+2, "Za": ROOT.kGreen+2, "aa": ROOT.kRed+2, 
    "inv": ROOT.kBlue+2}

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
    "gaga"      : ROOT.TColor.GetColor("#9c9ca1"),
    "Rare"      : ROOT.TColor.GetColor("#9c9ca1")
}

sign_name = {
    'cosThetaMiss':         '|cos#theta_{miss}|',
    'mva':                  'MVA', 
    'zll_m':                'm_{ll}', 
    'zll_p':                'p_{ll}',
    'recoil':               'm_{recoil}',
    'acolinearity':         '#Delta#theta_{ll}',
    'acoplanarity':         '#pi-#Delta#phi_{ll}',
    'leading_p':            'p_{l,leading}',
    'subleading_p':         'p_{l,subleading}',
    'leading_theta':        '#theta_{l,leading}',
    'subleading_theta':     '#theta_{l,subleading}',
    'leps_p':               'p_{lepton}',
    'leps_all_p_noSel':     'p_{lepton} no sel',
    'leps_all_theta_noSel': '#theta_{lepton} no sel',
    'leps_iso_noSel':       'I_{rel}',
    'visibleEnergy':        'E_{vis}'
}

sign_label = {
    'cosThetaMiss':         '|cos#theta_{miss}|',
    'mva':                  'MVA Score', 
    'zll_m':                'm_{ll} [GeV]', 
    'zll_p':                'p_{ll} [GeV]',
    'recoil':               'm_{recoil} [GeV]',
    'acolinearity':         '#Delta#theta_{ll}',
    'acoplanarity':         '#pi-#Delta#phi_{ll}',
    'leading_p':            'p_{l,leading} [GeV]',
    'subleading_p':         'p_{l,subleading} [GeV]',
    'leading_theta':        '#theta_{l,leading}',
    'subleading_theta':     '#theta_{l,subleading}',
    'leps_p':               'p_{lepton} [GeV]',
    'leps_all_p_noSel':     'p_{lepton} no sel [GeV]',
    'leps_all_theta_noSel': '#theta_{lepton} no sel [GeV]',
    'leps_iso_noSel':       'Cone isolation',
    'visibleEnergy':        'E_{vis} [GeV]'
}

#__________________________________________________________
def getHist(hName, procs, inputDir, rebin=-1):
    hist = None
    for proc in procs:
        fInName = f"{inputDir}/{proc}.root"
        
        if os.path.exists(fInName):
            fIn = ROOT.TFile(fInName)
        elif os.path.exists(fInName.replace("wzp6", "wz3p6")):
            fIn = ROOT.TFile(fInName.replace("wzp6", "wz3p6"))
        else:
            print(f"ERROR: input file {fInName} not found")
            quit()
        h = fIn.Get(hName)
        h.SetDirectory(0)
        if hist == None:
            hist = h
        else:
            hist.Add(h)
        fIn.Close()
    if rebin!=-1:
        hist.Rebin(rebin)
    return hist

#__________________________________________________________
def CutFlow(inputDir, outDir, procs, procs_cfg, plot_file=['png'], ecm=240, lumi=10.8,
            outName="", hName="cutFlow", cuts=[], labels=[], sig_scale=1.0, yMin=1e6, yMax=1e10):
    
    if outName=="":
        outName = hName

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    hists_yields = []
    significances = []
    h_sig = getHist(hName, procs_cfg[procs[0]], inputDir)

    hists_yields.append(copy.deepcopy(h_sig))
    h_sig.Scale(sig_scale)
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(4)
    h_sig.SetLineStyle(1)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{procs_labels[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for i,bkg in enumerate(procs[1:]):
        h_bkg = getHist(hName, procs_cfg[bkg], inputDir)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, procs_labels[bkg], "F")
        st.Add(h_bkg)
        hists_yields.append(h_bkg)

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetLineWidth(2)

    for i,cut in enumerate(cuts):
        nsig = h_sig.GetBinContent(i+1) / sig_scale ## undo scaling
        nbkg = 0
        for histProc in hists_yields:
            nbkg = nbkg + histProc.GetBinContent(i+1)
        if (nsig+nbkg) == 0:
            print(f"Cut {cut} zero yield sig+bkg")
            s = -1
        else:
            s = nsig / (nsig + nbkg)**0.5
        # print(f'Cut {i}:\tSignificance = {s:.2f}')
        significances.append(s)

    ########### PLOTTING ###########
    cfg = {
        'logy'              : True,
        'logx'              : False,

        'xmin'              : 0,
        'xmax'              : len(cuts),
        'ymin'              : yMin,
        'ymax'              : yMax ,

        'xtitle'            : "",
        'ytitle'            : "Events",

        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",
        }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    canvas.SetTicks()
    dummy = plotter.dummy(len(cuts))
    dummy.GetXaxis().SetLabelSize(0.75*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,label in enumerate(labels): dummy.GetXaxis().SetBinLabel(i+1, label)
    dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")

    st.Draw("SAME HIST")
    h_bkg_tot.Draw("SAME HIST")
    h_sig.Draw("SAME HIST")
    leg.Draw("SAME")

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{outName}.{pl}")

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{outName}.txt", 'w') as f:
        sys.stdout = f

        formatted_row = '{:<10} {:<25} ' + ' '.join(['{:<25}']*len(procs))
        print(formatted_row.format(*(["Cut", "Significance"]+procs)))
        print(formatted_row.format(*(["----------"]+["-----------------------"]*(len(procs)+1))))
        for i,cut in enumerate(cuts):
            row = ["Cut %d"%i, "%.3f"%significances[i]]
            for histProc in hists_yields:
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.4e +/- %.2e" % (yield_, err))

            print(formatted_row.format(*row))
    sys.stdout = out_orig

#__________________________________________________________
def CutFlowDecays(inputDir, outDir, final_state, plot_file=['png'], ecm=240, lumi=10.8, hName="cutFlow", 
                  outName="", cuts=[], cut_labels=[], yMin=0, yMax=150, z_decays=[], h_decays=[], miss=False):

    if outName == "":
        outName = hName

    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
    leg = ROOT.TLegend(.2, .925-(len(sigs)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    hist_tot = None
    eff_final, eff_final_err = [], []
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig, inputDir)
        h_sig.Scale(100./h_sig.GetBinContent(1))

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

        eff_final.append(h_sig.GetBinContent(len(cuts)))
        eff_final_err.append(h_sig.GetBinError(len(cuts)))
        
        if hist_tot == None:
            hist_tot = h_sig.Clone("h_tot")
        else:
            hist_tot.Add(h_sig)

    hist_tot.Scale(1./len(sigs))
    eff_avg = sum(eff_final) / float(len(eff_final))
    eff_avg, eff_avg_err = hist_tot.GetBinContent(len(cuts)), hist_tot.GetBinError(len(cuts))
    eff_min, eff_max = eff_avg-min(eff_final), max(eff_final)-eff_avg

    ########### PLOTTING ###########
    cfg = {
        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : len(cuts),
        'ymin'              : yMin,
        'ymax'              : yMax ,

        'xtitle'            : "",
        'ytitle'            : "Selection efficiency [%]",

        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",
        }

    plotter.cfg = cfg

    canvas = plotter.canvas()
    canvas.SetGrid()
    canvas.SetTicks()
    dummy = plotter.dummy(len(cuts))
    dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,label in enumerate(cut_labels): dummy.GetXaxis().SetBinLabel(i+1, label)
    dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")

    txt = ROOT.TLatex()
    txt.SetTextSize(0.04)
    txt.SetTextColor(1)
    txt.SetTextFont(42)
    txt.SetNDC()
    txt.DrawLatex(0.2, 0.2, f"Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %")
    txt.DrawLatex(0.2, 0.15, f"Min/max: {eff_min:.2f}/{eff_max:.2f}")
    txt.Draw("SAME")

    for hist in hists:
        hist.Draw("SAME HIST")
    leg.Draw("SAME")

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{outName}_decays.{pl}")

    out_orig = sys.stdout
    
    with open(f"{outDir}/cutflow/{outName}_decays.txt", 'w') as f:
        sys.stdout = f

        formatted_row = '{:<10} ' + ' '.join(['{:<18}']*len(hists))
        print(formatted_row.format(*(["Cut"]+h_decays)))
        print(formatted_row.format(*(["----------"]+["----------------"]*len(hists))))
        for i,cut in enumerate(cuts):
            row = ["Cut %d"%i]
            for j,histProc in enumerate(hists):
                yield_, err = hists[j].GetBinContent(i+1), hists[j].GetBinError(i+1)
                row.append("%.2f +/- %.2f" % (yield_, err))
            print(formatted_row.format(*row))
        print("\n")
        print(f"Average: {eff_avg:.3f} +/- {eff_avg_err:.3f}")
        print(f"Min/max: {eff_min:.3f}/{eff_max:.3f}")
    sys.stdout = out_orig
    del canvas

    # make final efficiency plot eff_final, eff_final_err
    if miss:
        xMin, xMax = int(np.min(eff_final))-3, int(np.max(eff_final))+3
    elif final_state == "qq":
        xMin, xMax = int(eff_avg)-10, int(eff_avg)+10
    else:
        xMin, xMax = int(eff_avg)-5, int(eff_avg)+3
    h_pulls = ROOT.TH2F("pulls", "pulls", (xMax-xMin)*10, xMin, xMax, len(sigs)+1, 0, len(sigs)+1)
    g_pulls = ROOT.TGraphErrors(len(sigs)+1)

    ip = 0 # counter for TGraph
    g_pulls.SetPoint(ip, eff_avg, float(ip) + 0.5)
    g_pulls.SetPointError(ip, eff_avg_err, 0.)
    h_pulls.GetYaxis().SetBinLabel(ip+1, "Average")
    ip += 1

    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        g_pulls.SetPoint(ip, eff_final[i], float(ip) + 0.5)
        g_pulls.SetPointError(ip, eff_final_err[i], 0.)
        h_pulls.GetYaxis().SetBinLabel(ip+1, h_decays_labels[h_decay])
        ip += 1

    canvas = ROOT.TCanvas("c", "c", 800, 800)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.1)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.05)
    canvas.SetFillStyle(4000) # transparency?
    canvas.SetGrid(1, 0)
    canvas.SetTickx(1)

    xTitle = "Selection efficiency [%]"

    h_pulls.GetXaxis().SetTitleSize(0.04)
    h_pulls.GetXaxis().SetLabelSize(0.035)
    h_pulls.GetXaxis().SetTitle(xTitle)
    h_pulls.GetXaxis().SetTitleOffset(1)
    h_pulls.GetYaxis().SetLabelSize(0.055)
    h_pulls.GetYaxis().SetTickLength(0)
    h_pulls.GetYaxis().LabelsOption('v')
    h_pulls.SetNdivisions(506, 'XYZ')
    h_pulls.Draw("HIST 0")

    maxx = len(sigs)+1
    line = ROOT.TLine(eff_avg, 0, eff_avg, maxx)
    line.SetLineColor(ROOT.kGray)
    line.SetLineWidth(2)
    line.Draw("SAME")


    shade = ROOT.TGraph()
    shade.SetPoint(0, eff_avg-eff_avg_err, 0)
    shade.SetPoint(1, eff_avg+eff_avg_err, 0)
    shade.SetPoint(2, eff_avg+eff_avg_err, maxx)
    shade.SetPoint(3, eff_avg-eff_avg_err, maxx)
    shade.SetPoint(4, eff_avg-eff_avg_err, 0)
    #shade.SetFillStyle(3013)
    shade.SetFillColor(16)
    shade.SetFillColorAlpha(16, 0.35)
    shade.Draw("SAME F")

    g_pulls.SetMarkerSize(1.2)
    g_pulls.SetMarkerStyle(20)
    g_pulls.SetLineWidth(2)
    g_pulls.Draw('P0 SAME')

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

    txt = ROOT.TLatex()
    txt.SetTextSize(0.04)
    txt.SetTextColor(1)
    txt.SetTextFont(42)
    txt.SetNDC()
    txt.DrawLatex(0.2, 0.2, f"Avg eff: {eff_avg:.2f} #pm {eff_avg_err:.2f} %")
    txt.DrawLatex(0.2, 0.15, f"Min/max: {eff_min:.2f}/{eff_max:.2f}")
    txt.Draw("SAME")

    if not os.path.isdir(f'{outDir}/cutflow'):
            os.system(f'mkdir -p {outDir}/cutflow')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/selection_efficiency.{pl}")

#__________________________________________________________
def PlotDecays(hName, inputDir, outDir, z_decays, h_decays, xMin, xMax, yMin, yMax, xLabel, yLabel, 
               ecm=240, lumi=10.8, outName="", plot_file=['png'], logX=False, logY=True, rebin=1, xLabels=[]):

    if outName == "":
        outName = hName

    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
    leg = ROOT.TLegend(.2, .925-(len(sigs)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig, inputDir)
        h_sig.Rebin(rebin)
        if h_sig.Integral() > 0:
            h_sig.Scale(1./h_sig.Integral())

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

    cfg = {

        'logy'              : logY,
        'logx'              : logX,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : xLabel,
        'ytitle'            : yLabel,
            
        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(1 if len(xLabels) == 0 else len(xLabels))
    if len(xLabels) > 0:
        dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
        dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
        for i,label in enumerate(xLabels): dummy.GetXaxis().SetBinLabel(i+1, label)
        dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST") 

    for i,hist in enumerate(hists):
        hist.Draw("SAME HIST")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()
    
    name = hName.split('_')[-1]
    if   'low'  in name: out = f'{outDir}/higgsDecays/low'
    elif 'high' in name: out = f'{outDir}/higgsDecays/high'
    elif 'mvis' in name: out = f'{outDir}/higgsDecays/mvis'
    elif 'minv' in name: out = f'{outDir}/higgsDecays/minv'
    elif 'vis'  in name: out = f'{outDir}/higgsDecays/vis'
    elif 'inv'  in name: out = f'{outDir}/higgsDecays/inv'
    else :               out = f'{outDir}/higgsDecays/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")
    canvas.Close()

#__________________________________________________________
def SignalRatios(hName, inputDir, outDir, z_decays, h_decays, ecm=240, lumi=10.8, outName="", xMin=0, xMax=100, 
                 plot_file=['png'], yMin=1, yMax=1e5, xLabel="xlabel", yLabel="Events", rebin=-1):

    if outName == "":
        outName = hName

    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays] for y in h_decays]
    leg = ROOT.TLegend(.2, .925-(len(sigs)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    h_tot = None
    for i,sig in enumerate(sigs):
        h_decay = h_decays[i]
        h_sig = getHist(hName, sig, inputDir)
        if rebin!=-1:
            h_sig.Rebin(rebin)
        # h_sig.Scale(1./h_sig.Integral())

        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)
        
        if h_tot == None:
            h_tot = h_sig.Clone("h_tot")
        else:
            h_tot.Add(h_sig)

    h_tot.Scale(1./h_tot.Integral())

    for h in hists:
        h.Scale(1./h.Integral())
        h.Divide(h_tot)

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : xLabel,
        'ytitle'            : yLabel,

        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
    dummy.Draw("HIST") 

    for i,hist in enumerate(hists):
        hist.Draw("SAME HIST")
    leg.Draw("SAME")

    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    if not os.path.isdir(f'{outDir}/ratio'):
            os.system(f'mkdir -p {outDir}/ratio')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/ratio/{outName}.{pl}")
    canvas.Close()

#__________________________________________________________
def makePlot(hName, inputDir, outDir, procs, procs_cfg, xMin, xMax, yMin, yMax, xLabel, yLabel, 
             ecm=240, lumi=10.8, outName="", logX=False, logY=True, rebin=-1, sig_scale=1, xLabels=[], 
             plot_file=['png'], stack=False):

    if outName == "": outName = hName

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    h_sig = getHist(hName, procs_cfg[procs[0]], inputDir, rebin=rebin)
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(3)
    h_sig.SetLineStyle(1)
    h_sig.Scale(sig_scale)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{procs_labels[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for i,bkg in enumerate(procs[1:]):
        h_bkg = getHist(hName, procs_cfg[bkg], inputDir, rebin=rebin)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, procs_labels[bkg], "F")
        st.Add(h_bkg)



    if yMax < 0:
        if logY:
            yMax = math.ceil(max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])*10000)/10.
        else:
            yMax = 1.4*max([h_bkg_tot.GetMaximum(), h_sig.GetMaximum()])

    cfg = {

        'logy'              : logY,
        'logx'              : logX,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : xLabel,
        'ytitle'            : yLabel,
            
        'topRight'          : f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}",
        'topLeft'           : "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}",

    }

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy(1 if len(xLabels) == 0 else len(xLabels))
    if len(xLabels) > 0:
        dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
        dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
        for i,label in enumerate(xLabels): dummy.GetXaxis().SetBinLabel(i+1, label)
        dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")
    st.Draw("HIST SAME")

    if stack:
        h_bkg_tot.Add(h_sig)
        h_bkg_tot.SetLineColor(ROOT.kRed)
    
    else:
        h_bkg_tot.SetLineColor(ROOT.kBlack)
        h_sig.Draw('HIST SAME')
    
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)
    #hTot_err.Draw("E2 SAME")
    h_bkg_tot.Draw("HIST SAME")
    leg.Draw("SAME")
    
    canvas.SetGrid()
    canvas.Modify()
    canvas.Update()

    plotter.aux()
    ROOT.gPad.SetTicks()
    ROOT.gPad.RedrawAxis()

    name = hName.split('_')[-1]
    if   'low'  in name: out = f'{outDir}/higgsDecays/low'
    elif 'high' in name: out = f'{outDir}/higgsDecays/high'
    elif 'mvis' in name: out = f'{outDir}/higgsDecays/mvis'
    elif 'minv' in name: out = f'{outDir}/higgsDecays/minv'
    elif 'vis'  in name: out = f'{outDir}/higgsDecays/vis'
    elif 'inv'  in name: out = f'{outDir}/higgsDecays/inv'
    else :               out = f'{outDir}/higgsDecays/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")
    canvas.Close()

#__________________________________________________________
def significance(hName, inputDir, outDir, procs, procs_cfg, xMin, xMax, outName="", reverse=False, plot_file=['png']):

    if outName == "": outName = hName
    h_sig = getHist(hName, procs_cfg[procs[0]], inputDir)
    sig_tot = h_sig.Integral()

    bkgs_procs = []
    for i,bkg in enumerate(procs[1:]):
        bkgs_procs.extend(procs_cfg[bkg])

    h_bkg = getHist(hName, bkgs_procs, inputDir)
    x, y, l = [], [], []

    for i in range(1, h_sig.GetNbinsX()+1):
        if reverse:
            iStart = 1
            iEnd = i
        else:
            iStart = i
            iEnd = h_sig.GetNbinsX()+1
        center = h_sig.GetBinCenter(i)
        if center > xMax or center < xMin:
            continue
        sig = h_sig.Integral(iStart, iEnd)
        bkg = h_bkg.Integral(iStart, iEnd)
        if (sig+bkg) <= 0 or sig_tot <= 0:
            significance = 0
            sig_loss = 0
        else:
            significance = sig / (sig + bkg)**0.5
            sig_loss = sig / sig_tot
        # print(f"{i} {center:.5f} {significance:.5f} {sig_loss:.5f}")
        x.append(center)
        y.append(significance)
        l.append(sig_loss)

    graph_sign = ROOT.TGraph(len(x), array.array('d', x), array.array('d', y))
    graph_l = ROOT.TGraph(len(x), array.array('d', x), array.array('d', l))
    max_y = max(y)
    max_index = y.index(max_y)
    max_x = x[max_index]
    max_l = l[max_index]

    canvas = ROOT.TCanvas("", "", 1100, 1000)
    canvas.SetTopMargin(0.09)
    canvas.SetRightMargin(0.11)
    canvas.SetLeftMargin(0.11)
    canvas.SetBottomMargin(0.11)

    graph_sign.SetMarkerStyle(20)
    graph_sign.SetMarkerColor(ROOT.kBlue)
    graph_sign.GetXaxis().SetRangeUser(xMin, xMax)
    graph_sign.GetYaxis().SetTitle('Significance')
    graph_sign.GetYaxis().SetTitleFont(43)
    graph_sign.GetYaxis().SetTitleSize(40)
    graph_sign.GetYaxis().SetLabelFont(43)
    graph_sign.GetYaxis().SetLabelSize(35)
    graph_sign.GetXaxis().SetTitleOffset(1.2*graph_sign.GetXaxis().GetTitleOffset())
    graph_sign.GetYaxis().SetLabelOffset(1.2*graph_sign.GetYaxis().GetLabelOffset())

    for key, value in sign_name.items():
        if key in hName:
            cut = value
            label = key
    graph_sign.GetXaxis().SetTitle(sign_label[label])

    graph_sign.Draw("AP")
    canvas.Update()

    # Add a marker for the maximum point
    max_marker = ROOT.TMarker(max_x, max_y, 20)
    max_marker.SetMarkerColor(ROOT.kRed)
    max_marker.SetMarkerSize(1.5)
    max_marker.Draw()

    rightmax = 1.0
    scale =  ROOT.gPad.GetUymax()/rightmax
    rightmin = ROOT.gPad.GetUymin()/ROOT.gPad.GetUymax()
    graph_l.Scale(scale)
    graph_l.SetLineColor(ROOT.kRed)
    graph_l.SetLineWidth(2)
    graph_l.Draw("SAME L")

    axis_r = ROOT.TGaxis(ROOT.gPad.GetUxmax(),ROOT.gPad.GetUymin(), ROOT.gPad.GetUxmax(), 
                         ROOT.gPad.GetUymax(),rightmin,rightmax,510,"+L")
    axis_r.SetLineColor(ROOT.kRed)
    axis_r.SetLabelColor(ROOT.kRed)
    axis_r.SetTitle('Signal efficiency')
    axis_r.SetTitleColor(ROOT.kRed)
    axis_r.SetTitleOffset(1.3*axis_r.GetTitleOffset())
    axis_r.Draw()

    # Add a text box to indicate the maximum value
    text = ROOT.TLatex()
    text.SetTextSize(0.03)
    text.SetTextColor(ROOT.kBlack)
    if reverse:
        text.DrawLatexNDC(0.12, 0.94, f"Max: {cut} < {max_x:.2f}, Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} %")
    else:
        text.DrawLatexNDC(0.12, 0.94, f"Max: {cut} > {max_x:.2f}, Significance = {max_y:.2f}, Signal eff = {max_l*100:.1f} %")

    canvas.Update()

    name = hName.split('_')[-1]
    if   'low'  in name: out = f'{outDir}/significance/low'
    elif 'high' in name: out = f'{outDir}/significance/high'
    elif 'mvis' in name: out = f'{outDir}/significance/mvis'
    elif 'minv' in name: out = f'{outDir}/significance/minv'
    elif 'vis'  in name: out = f'{outDir}/significance/vis'
    elif 'inv'  in name: out = f'{outDir}/significance/inv'
    else :               out = f'{outDir}/significance/nominal'


    suffix = "_reverse" if reverse else ""
    if not os.path.isdir(f'{outDir}/significance'):
            os.system(f'mkdir -p {outDir}/significance')
    
    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}{suffix}.{pl}")
    canvas.Close()
