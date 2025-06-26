import sys
import os
import math
import copy
import array
import numpy as np

import ROOT as root
root.gROOT.SetBatch(True)
root.gStyle.SetOptStat(0)
root.gStyle.SetOptTitle(0)

from . import plotter
import atlasplots as aplt

h_decays_labels = {"bb": "H#rightarrowb#bar{b}", "cc": "H#rightarrowc#bar{c}", "ss": "H#rightarrows#bar{s}", 
                   "gg": "H#rightarrowgg", "mumu": "H#rightarrow#mu^{#plus}#mu^{#minus}", "tautau": "H#rightarrow#tau^{#plus}#tau^{#minus}", 
                   "ZZ": "H#rightarrowZZ*", "WW": "H#rightarrowWW*", "Za": "H#rightarrowZ#gamma", 
                   "aa": "H#rightarrow#gamma#gamma", "inv": "H#rightarrowInv"}

h_decays_colors = {"bb": root.kBlack, "cc": root.kBlue , "ss": root.kRed, "gg": root.kGreen+1, "mumu": root.kOrange, 
                   "tautau": root.kCyan, "ZZ": root.kGray, "WW": root.kGray+2, "Za": root.kGreen+2, "aa": root.kRed+2, 
                   "inv": root.kBlue+2}

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
    "ZH"        : root.TColor.GetColor("#e42536"),
    "ZqqH"      : root.TColor.GetColor("#e42536"),
    "ZmumuH"    : root.TColor.GetColor("#e42536"),
    "ZnunuH"    : root.TColor.GetColor("#e42536"),
    "ZeeH"      : root.TColor.GetColor("#e42536"),
    "WW"        : root.TColor.GetColor("#f89c20"),
    "ZZ"        : root.TColor.GetColor("#5790fc"),
    "Zgamma"    : root.TColor.GetColor("#964a8b"),
    "Zqqgamma"  : root.TColor.GetColor("#964a8b"),
    "Rare"      : root.TColor.GetColor("#9c9ca1")
}

#__________________________________________________________
def getHist(hName, inputDir, target='', rebin=-1):
    
    tar = f'_{target}' if target!='' else ''
    fInName = f"{inputDir}/datacard{tar}.root"
    if os.path.exists(fInName):
        fIn = root.TFile(fInName)
    elif os.path.exists(fInName.replace("wzp6", "wz3p6")):
        fIn = root.TFile(fInName.replace("wzp6", "wz3p6"))
    else:
        print(f"ERROR: input file {fInName} not found")
        quit()
    h = fIn.Get(hName)
    h.SetDirectory(0)
    fIn.Close()
    if rebin!=-1:
        h.Rebin(rebin)
    return h

#__________________________________________________________
def range_hist(hist_original, x_min, x_max):

    # Get the bin numbers corresponding to the range
    bin_min = hist_original.FindBin(x_min)
    bin_max = hist_original.FindBin(x_max)
    

    hist_selected = root.TH1D(hist_original.GetName()+"new", "", bin_max - bin_min, x_min, x_max)

    for bin in range(bin_min, bin_max + 1):
        new_bin = bin - bin_min + 1  # Adjust for new histogram bin indexing
        hist_selected.SetBinContent(new_bin, hist_original.GetBinContent(bin))
        hist_selected.SetBinError(new_bin, hist_original.GetBinError(bin))  # Preserve errors

    # print(bin_min, bin_max, hist_original.Integral(), hist_selected.Integral())
    return hist_selected


#__________________________________________________________
# def Divide(base, sig, h_decay):
#     N = base.GetNbinsX()
#     h1 = root.TH1D('h1', 'Ratio', N, 0.5, N+0.5)
#     for i in range(1, N+1):
#         bin_base, bin_sig = base.GetBinContent(i), sig.GetBinContent(i)
#         err_base, err_sig = base.GetBinError(i),   sig.GetBinError(i)

#         if bin_base!=0:
#             r = bin_sig/bin_base
#             err = r * np.sqrt( (err_base/bin_base)**2 + (err_sig/bin_sig)**2 )
#             h1.SetBinContent(i, r)
#             h1.SetBinError(i, err)

#     h1.SetLineColor(h_decays_colors[h_decay])
#     h1.SetLineWidth(2)
#     h1.SetLineStyle(1)
#     return h1



#__________________________________________________________
def PlotDecays(hName, inputDir, outDir, h_decays, ecm=240, lumi=10.8, outName="", xMin=0, xMax=200, 
               yMin=0, yMax=-1, plot_file=['png'], xLabel="", yLabel="Events", logX=False, logY=True, rebin=-1, xLabels=[]):

    if outName == "": outName = hName

    leg = root.TLegend(.2, .925-(len(h_decays)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists = []
    h_base = getHist(f'{hName}_{h_decays[0]}', inputDir, target=h_decays[0], rebin=rebin)
    if rebin!=-1:
        h_base.Rebin(rebin)
    if xMin!=0 or xMax!=200:
        h_base = range_hist(h_base, xMin, xMax)
    if h_base.Integral() > 0:
        h_base.Scale(1./h_base.Integral())
    
    for h_decay in h_decays:
        h_sig = getHist(f'{hName}_{h_decay}', inputDir, target=h_decay, rebin=rebin)
        if rebin!=-1:
            h_sig.Rebin(rebin)
        if xMin!=0 or xMax!=200:
            h_sig = range_hist(h_sig, xMin, xMax)
        if h_sig.Integral() > 0:
            h_sig.Scale(1./h_sig.Integral())
        
        h_sig.Divide(h_base)

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
    root.gPad.SetTicks()
    root.gPad.RedrawAxis()

    if 'High-mass'  in xLabel: out = f'{outDir}/higgsDecays/high'
    elif 'Low-mass' in xLabel: out = f'{outDir}/higgsDecays/low'
    else :                     out = f'{outDir}/higgsDecays/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        canvas.SaveAs(f"{out}/{outName}.{pl}")



#__________________________________________________________
def makePlot(inputDir, outDir, cat, procs, target, sig='ZH', ecm=240, lumi=10.8, outName="", xMin=None, xMax=None, yMin=None, yMax=None, 
             ymin=0.9, ymax=1.1, xLabel="xlabel", yLabel="Events", logX=False, logY=True, rebin=-1, xLabels=[], plot_file=['png']):

    # aplt.set_atlas_style()

    fig, (ax1, ax2) = aplt.ratio_plot(figsize=(1000, 1000), hspace=0.05)
    ax1.set_pad_margins(left=0.15, right=0.05, top=0.055)
    ax2.set_pad_margins(left=0.15, right=0.05, bottom=0.25)
    leg = ax1.legend(loc=(.75, .99-len(procs)*0.06, .99, .90))
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    h_sig = getHist(f'{cat}_{sig}', inputDir, target=target, rebin=rebin)
    h_sig.SetLineColor(procs_colors[procs[0]])
    h_sig.SetLineWidth(3)
    h_sig.SetLineStyle(1)
    leg.AddEntry(h_sig, procs_labels[procs[0]], "L")

    h_asi = getHist(f'{cat}_data_{target}', inputDir, target=target, rebin=rebin)
    h_asi.SetMarkerStyle(8)
    h_asi.SetMarkerColor(root.kBlack)
    h_asi.SetMarkerSize(1)
    leg.AddEntry(h_asi, "Pseudo-data", "EP")

    st = root.THStack()
    st.SetName("stack")
    h_bkg_tot = None
    for bkg in procs[1:]:
        h_bkg = getHist(f'{cat}_{bkg}', inputDir, target=target, rebin=rebin)

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(procs_colors[bkg])
        h_bkg.SetLineColor(root.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)
        leg.AddEntry(h_bkg, procs_labels[bkg], "F")

        st.Add(h_bkg)

    h_bkg_tot.Add(h_sig)
    h_bkg_tot.SetLineColor(root.kRed)
    h_bkg_tot.SetFillColor(0)
    h_bkg_tot.SetLineWidth(2)

    ax1.plot(st)
    asi = aplt.root_helpers.hist_to_graph(h_asi)
    ax1.plot(h_bkg_tot, "HIST")
    ax1.plot(asi, "EP", markerstyle=8)
    ax1.add_margins(top=0.20)

    ax1.set_xlim(xMin, xMax)
    ax1.set_ylim(yMin, yMax)

    line = root.TLine(ax1.get_xlim()[0], 1, ax1.get_xlim()[1], 1)
    ax2.plot(line)

    ratio = h_asi.Clone('ratio')
    ratio.Divide(h_bkg_tot)
    ratio_graph = aplt.root_helpers.hist_to_graph(ratio)
    ax2.plot(ratio_graph, "P0", markerstyle=8)

    # ax2.add_margins(top=0.1, bottom=0.1)
    ax2.set_xlim(xMin, xMax)
    ax2.set_ylim(ymin, ymax)

    ax2.set_xlabel(xLabel, titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    ax1.set_ylabel(yLabel, titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    ax2.set_ylabel('Data / MC', loc='centre', titlesize=40, titlefont=43, labelfont=43, labelsize=35)
    
    if logX: ax1.set_xscale('log')
    if logY: ax1.set_yscale('log')

    ax1.text(0.95, 0.945, f"#sqrt{{s}} = {ecm} GeV, {lumi} ab^{{#minus1}}", size=0.04, font=42, align=30)
    ax1.text(0.15, 0.95, "#bf{FCC-ee} #scale[0.7]{#it{Simulation}}", size=0.04, font=42, align=10)

    root.gPad.RedrawAxis("g")

    if   'High-mass' in xLabel: out = f'{outDir}/makePlot/high'
    elif 'Low-mass'  in xLabel: out = f'{outDir}/makePlot/low'
    else :                      out = f'{outDir}/makePlot/nominal'

    if not os.path.isdir(out):
            os.system(f'mkdir -p {out}')

    for pl in plot_file:
        fig.savefig(f"{out}/{outName}.{pl}")
