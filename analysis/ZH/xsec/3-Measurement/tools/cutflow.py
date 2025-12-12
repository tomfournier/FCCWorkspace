import os, sys, copy, json, uproot, ROOT
from glob import glob
from tqdm import tqdm

from . import plotter
from .plotting import getMetaInfo, h_decays_colors, h_decays_labels



#__________________________________________________________
def get_processed(file: str) -> int:
    f = ROOT.TFile(file, 'READ')
    return f.Get('eventsProcessed').GetVal()



#__________________________________________________________
def get_cut(files: list[str], cut: str, doeff: bool = False) -> int:
    n, e = 0, cut.replace('cut', 'eff')
    for file in files:
        df = ROOT.ROOT.RDataFrame("events", file)
        if doeff and 'p8_ee_WW_ecm' in file and e in df.GetColumnNames():
            eff = df.Mean(e).GetValue()
        else:
            eff = 1
        n += eff * df.Mean(cut).GetValue()
    return int(n)

# def get_cut(files: list[str] | 'ROOT.vector(string)', cut: str, doeff: bool = False) -> int:
#     """
#     Compute number of events passing `cut` for the provided files.
#     Accepts either a python list of filenames or a ROOT.vector('string').
#     Uses a single RDataFrame over all files to avoid per-file RDataFrame overhead.
#     """
#     # normalize to python list of strings
#     if hasattr(files, "size"):
#         # ROOT.vector('string')
#         files_py = [files[i] for i in range(files.size())]
#     else:
#         files_py = list(files)

#     if len(files_py) == 0:
#         return 0

#     rdf = ROOT.ROOT.RDataFrame("events", files_py)

#     # optional efficiency column handling (existing logic preserved)
#     e = cut.replace('cut', 'eff')
#     eff = 1.0
#     if doeff and any('p8_ee_WW_ecm' in f for f in files_py) and e in rdf.GetColumnNames():
#         eff = float(rdf.Mean(e).GetValue())

#     # if `cut` is an existing column (0/1), compute mean * entries to get count
#     colnames = set(rdf.GetColumnNames())
#     if cut in colnames:
#         total = float(rdf.Count().GetValue())
#         mean = float(rdf.Mean(cut).GetValue())
#         n = eff * mean * total
#     else:
#         # treat `cut` as a filter expression
#         n = float(rdf.Filter(cut).Count().GetValue())

#     return int(n)

# def get_processed(file: str) -> int:
#     # try a fast uproot read of the scalar parameter
#     try:
#         f = uproot.open(file)
#         if 'eventsProcessed' in f:
#             val = f['eventsProcessed'].value
#             return int(val)
#     except Exception:
#         pass
#     # fallback to PyROOT
#     f = ROOT.TFile.Open(file, 'READ')
#     return int(f.Get('eventsProcessed').GetVal())



#__________________________________________________________
def getcut(df, filter: str, cut: str = '', indf: bool = False) -> float:
    if indf:
        n = df.Mean(cut).GetValue()
    else:
        df = df.Filter(filter)
        n = df.Count().GetValue()
    return df, n



#__________________________________________________________
def find_sample_files(input_dir: str, sample_name: str) -> list[str]:
    '''
    Find input files for the specified sample name.
    '''
    result: list[str] = []
    full_input_path = os.path.abspath(os.path.join(input_dir, sample_name))

    # Find all input files ending with .root
    if os.path.isdir(full_input_path):
        all_files = os.listdir(full_input_path)
        # Remove files not ending with `.root`
        all_files = [f for f in all_files if f.endswith('.root')]
        # Remove directories
        all_files = [f for f in all_files
                     if os.path.isfile(os.path.join(full_input_path, f))]
        result = [os.path.join(full_input_path, f) for f in all_files]

    # Handle case when there is just one input file
    if len(result) < 1:
        if os.path.isfile(full_input_path + '.root'):
            result.append(full_input_path + '.root')

    return result



#__________________________________________________________
def is_there_events(proc: str, path: str = '', end='.root') -> bool:
    if os.path.exists(f'{path}/{proc}{end}'):
        filename = f'{path}/{proc}{end}'
        file = uproot.open(filename)
        return 'events' in file
    elif os.path.isdir(f'{path}/{proc}'):
        filenames = glob(f'{path}/{proc}/*')
        isTTree = []
        for i, filename in enumerate(filenames):
            file = uproot.open(filename)
            if 'events' in file:
                isTTree.append(i)
        return len(isTTree)==len(filenames)
    else: 
        print(f'ERROR: Could not find ROOT file for {proc}')
        quit()



#__________________________________________________________
def dump_json(dic, out, outName, hist=False, procs=[]) -> None:

    dictio = copy.deepcopy(dic)
    if hist:
        for proc in procs:
            del dictio[proc]['hist']

    if not os.path.exists(f'{out}'): 
        os.system(f'mkdir -p {out}')
    with open(f'{out}/{outName}.json', 'w') as outfile:
        json.dump(dictio, outfile, indent=4)



#__________________________________________________________
def get_flow(events, procs, procs_cfg, cuts, cat, sel, tot=False,
             json_file=False, loc='', outName='flow', suffix=''):
    
    procs[0] = f'Z{cat}H' if not tot else 'ZH'
    _tot = '_tot' if tot else ''

    flow = {}
    for proc in procs:
        flow[proc] = {}
        flow[proc]['hist'], flow[proc]['cut'], flow[proc]['err']  = [], {}, {}
        hist = ROOT.TH1D(proc+f'_{sel}'+_tot, proc+f'_{sel}'+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[proc]['cut'][cut], flow[proc]['err'][cut]  = 0, 0
            for sample in procs_cfg[proc]:
                if "Hinv" in sample: continue
                flow[proc]['cut'][cut] += events[sample][sel]['cut'][cut]
                flow[proc]['err'][cut] += events[sample][sel]['err'][cut]**2
            
            hist.SetBinContent(i+1, flow[proc]['cut'][cut])
            hist.SetBinError(i+1,   flow[proc]['err'][cut]**0.5)
        flow[proc]['hist'].append(hist)
    
    if json_file:
        dump_json(flow, loc, outName+f'_{sel}'+suffix, hist=True, procs=procs)
    return flow



#__________________________________________________________
def get_flow_decay(events, z_decays, h_decays, cuts, cat, sel, outName='flow_decay',
                   ecm=240, json_file=False, loc='', suffix='', tot=False):
    
    cats = z_decays if tot else [cat]
    sigs = [[f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in cats] for y in h_decays]
    _tot = '_tot' if tot else ''

    flow = {}
    for h, sig in zip(h_decays, sigs):
        flow[h] = {}
        flow[h]['hist'], flow[h]['cut'], flow[h]['err'] = [], {}, {}
        hist = ROOT.TH1D('H'+h+f'_{sel}'+_tot, 'H'+h+f'_{sel}'+_tot, len(cuts[sel]), 0, len(cuts[sel]))
        for i, cut in enumerate(cuts[sel]):
            flow[h]['cut'][cut], flow[h]['err'][cut] = 0, 0
            for s in sig:
                flow[h]['cut'][cut] += events[s][sel]['cut'][cut]
                flow[h]['err'][cut] += events[s][sel]['err'][cut]**2
            hist.SetBinContent(i+1, flow[h]['cut'][cut])
            hist.SetBinError(i+1,   flow[h]['err'][cut]**0.5)
        flow[h]['hist'].append(hist)

    if json_file:
        dump_json(flow, loc, outName+f'_{sel}'+suffix, hist=True, procs=h_decays)
    return flow



#__________________________________________________________
def get_flows(procs, procs_cfg, cuts, events, cat, sel, z_decays, h_decays, 
              ecm=240, json_file=False, tot=False, loc_json=''):
    
    suffix = '_tot' if tot else ''
    flow = get_flow(events, procs, procs_cfg, cuts, cat, sel, tot=tot,
                    json_file=json_file, loc=loc_json, suffix=suffix)
    
    flow_decay = get_flow_decay(events, z_decays, h_decays, cuts, cat, sel, 
                                ecm=ecm, json_file=json_file, loc=loc_json, suffix=suffix, tot=tot)
    return flow, flow_decay



#__________________________________________________________
def CutFlow(flow, outDir, cat, sel, procs, colors, legend, cuts, labels, ecm=240, lumi=10.8,
            outName="cutFlow", plot_file=['png'], sig_scale=1.0, yMin=1e4, yMax=1e10):
    
    cat = 'e' if cat=='ee' else '#mu'

    leg = ROOT.TLegend(.55, 0.99-(len(procs))*0.06, .99, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.2)

    hists_yields, significances = [], []
    h_sig = flow[procs[0]]['hist'][0]

    hists_yields.append(copy.deepcopy(h_sig))
    h_sig.Scale(sig_scale)
    h_sig.SetLineColor(colors[procs[0]])
    h_sig.SetLineWidth(4)
    h_sig.SetLineStyle(1)
    if sig_scale != 1:
        leg.AddEntry(h_sig, f"{legend[procs[0]]} (#times {int(sig_scale)})", "L")
    else:
        leg.AddEntry(h_sig, legend[procs[0]], "L")

    st, h_bkg_tot = ROOT.THStack(), None
    st.SetName("stack")
    for bkg in procs[1:]:
        h_bkg = flow[bkg]['hist'][0]

        if h_bkg_tot == None: h_bkg_tot = h_bkg.Clone("h_bkg_tot")
        else: h_bkg_tot.Add(h_bkg)
        
        h_bkg.SetFillColor(colors[bkg])
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetLineWidth(1)
        h_bkg.SetLineStyle(1)

        leg.AddEntry(h_bkg, legend[bkg], "F")
        st.Add(h_bkg)
        hists_yields.append(h_bkg)

    h_bkg_tot.SetLineColor(ROOT.kBlack)
    h_bkg_tot.SetLineWidth(2)

    for i,cut in enumerate(cuts[sel]):
        nsig, nbkg = h_sig.GetBinContent(i+1) / sig_scale, 0 ## undo scaling
        for histProc in hists_yields:
            nbkg += histProc.GetBinContent(i+1)
        if (nsig+nbkg) == 0:
            print(f"Cut {cut} zero yield sig+bkg")
            s = -1
        else: s = nsig / (nsig + nbkg)**0.5
        significances.append(s)

    ########### PLOTTING ###########
    cfg = {
        'logy'              : True,
        'logx'              : False,

        'xmin'              : 0,
        'xmax'              : len(cuts[sel]),
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
    dummy = plotter.dummy(len(cuts[sel]))
    dummy.GetXaxis().SetLabelSize(0.75*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,cut in enumerate(labels[sel]): dummy.GetXaxis().SetBinLabel(i+1, labels[sel][cut].replace('#ell', cat))
    dummy.GetXaxis().LabelsOption("u")
    dummy.Draw("HIST")

    st.Draw("SAME HIST")
    h_bkg_tot.Draw("SAME HIST")
    h_sig.Draw("SAME HIST")
    leg.Draw("SAME")

    if not os.path.isdir(f'{outDir}/cutflow/{sel}'):
        os.system(f'mkdir -p {outDir}/cutflow/{sel}')

    plotter.aux()
    canvas.RedrawAxis()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{sel}/{outName}.{pl}")

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{sel}/{outName}.txt", 'w') as f:
        sys.stdout = f
        formatted_row = '{:<10} {:<25} ' + ' '.join(['{:<25}']*len(procs))
        print(formatted_row.format(*(["Cut", "Significance"]+procs)))
        print(formatted_row.format(*(["----------"]+["-----------------------"]*(len(procs)+1))))
        for i in range(len(cuts[sel])):
            row = ["Cut %d"%i, "%.3f"%significances[i]]
            for histProc in hists_yields:
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.2e +/- %.2e" % (yield_, err))

            print(formatted_row.format(*row))
    sys.stdout = out_orig



#__________________________________________________________
def CutFlowDecays(flow, outDir, cat, sel, h_decays, cuts, labels, suffix='', ecm=240, lumi=10.8, 
                  outName="cutFlow_decays", plot_file=['png'], yMin=0, yMax=150):

    cat = 'e' if cat=='ee' else '#mu'

    leg = ROOT.TLegend(.2, .925-(len(h_decays)/4+1)*0.07, .95, .925)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.03)
    leg.SetMargin(0.25)
    leg.SetNColumns(4)

    hists, hist_yield, hist_tot = [], [], None
    eff_final, eff_final_err = [], []
    for h_decay in h_decays:
        h_sig = flow[h_decay]['hist'][0]
        hist_yield.append(h_sig.Clone(f'yield_{h_decay}'))
        h_sig.Scale(100./h_sig.GetBinContent(1))
        h_sig.SetLineColor(h_decays_colors[h_decay])
        h_sig.SetLineWidth(2)
        h_sig.SetLineStyle(1)

        leg.AddEntry(h_sig, h_decays_labels[h_decay], "L")
        hists.append(h_sig)

        eff_final.append(h_sig.GetBinContent(len(cuts[sel])))
        eff_final_err.append(h_sig.GetBinError(len(cuts[sel])))
        
        if hist_tot == None:
            hist_tot = h_sig.Clone("h_tot")
        else:
            hist_tot.Add(h_sig)

    hist_tot.Scale(1./len(h_decays))
    eff_avg = sum(eff_final) / float(len(eff_final))
    eff_avg, eff_avg_err = hist_tot.GetBinContent(len(cuts[sel])), hist_tot.GetBinError(len(cuts[sel]))
    eff_min, eff_max = eff_avg-min(eff_final), max(eff_final)-eff_avg

    ########### PLOTTING ###########
    cfg = {
        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : len(cuts[sel]),
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
    dummy = plotter.dummy(len(cuts[sel]))
    dummy.GetXaxis().SetLabelSize(0.8*dummy.GetXaxis().GetLabelSize())
    dummy.GetXaxis().SetLabelOffset(1.3*dummy.GetXaxis().GetLabelOffset())
    for i,cut in enumerate(labels[sel]): dummy.GetXaxis().SetBinLabel(i+1, labels[sel][cut].replace('#ell', cat))
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

    if not os.path.isdir(f'{outDir}/cutflow/{sel}'):
        os.system(f'mkdir -p {outDir}/cutflow/{sel}')

    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{sel}/{outName}.{pl}")

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{sel}/{outName}.txt", 'w') as f:
        sys.stdout = f
        formatted_row = '{:<10} ' + ' '.join(['{:<18}']*len(hists))
        print(formatted_row.format(*(["Cut"]+h_decays)))
        print(formatted_row.format(*(["----------"]+["----------------"]*len(hists))))
        for i in range(len(cuts[sel])):
            row = ["Cut %d"%i]
            for histProc in hists:
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.2f +/- %.2f" % (yield_, err))
            print(formatted_row.format(*row))
        print("\n")
        print(f"Average: {eff_avg:.3f} +/- {eff_avg_err:.3f}")
        print(f"Min/max: {eff_min:.3f}/{eff_max:.3f}")
    sys.stdout = out_orig

    out_orig = sys.stdout
    with open(f"{outDir}/cutflow/{sel}/yields_decays{suffix}.txt", 'w') as f:
        sys.stdout = f
        formatted_row = '{:<10} ' + ' '.join(['{:<25}']*len(hist_yield))
        print(formatted_row.format(*(["Cut"]+h_decays)))
        print(formatted_row.format(*(["----------"]+["-----------------------"]*(len(hist_yield)))))
        for i in range(len(cuts[sel])):
            row = ["Cut %d"%i]
            for histProc in hist_yield:
                yield_, err = histProc.GetBinContent(i+1), histProc.GetBinError(i+1)
                row.append("%.2e +/- %.2e" % (yield_, err))

            print(formatted_row.format(*row))
    sys.stdout = out_orig
    del canvas

    # make final efficiency plot eff_final, eff_final_err
    xMin, xMax = int(min(eff_final))-5, int(max(eff_final))+3

    h_pulls = ROOT.TH2F("pulls", "pulls", (xMax-xMin)*10, xMin, xMax, len(h_decays)+1, 0, len(h_decays)+1)
    g_pulls = ROOT.TGraphErrors(len(h_decays)+1)

    g_pulls.SetPoint(0, eff_avg, 0.5)
    g_pulls.SetPointError(0, eff_avg_err, 0.)
    h_pulls.GetYaxis().SetBinLabel(1, "Average")

    for i,h_decay in enumerate(h_decays):
        g_pulls.SetPoint(i+1, eff_final[i], float(i+1) + 0.5)
        g_pulls.SetPointError(i+1, eff_final_err[i], 0.)
        h_pulls.GetYaxis().SetBinLabel(i+2, h_decays_labels[h_decay])

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

    maxx = len(h_decays)+1
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

    if not os.path.isdir(f'{outDir}/cutflow/{sel}'):
        os.system(f'mkdir -p {outDir}/cutflow/{sel}')
    for pl in plot_file:
        canvas.SaveAs(f"{outDir}/cutflow/{sel}/selection_efficiency{suffix}.{pl}")



#__________________________________________________________
def get_cutflow(inputDir, outDir, cat, sels, procs, procs_cfg, colors, legend, 
                cuts, cuts_label, z_decays, h_decays, H_decays, plot_file=['png'], 
                ecm=240, lumi=10.8, sig_scale=1.0, defineList={}, 
                scaled=True, tot=False, json_file=False, loc_json='', onlyhad=True):

    ROOT.ROOT.EnableImplicitMT()
    ncpus = ROOT.GetThreadPoolSize()
    print(f'----->[Info] Threading on {ncpus} CPUs')

    events, file_list = {}, {}
    new_procs = copy.deepcopy(procs)
    if not tot: new_procs[0] = f'Z{cat}H'
    new_procs[0] = new_procs[0].lower() # To include Hinv

    print('----->[Info] Getting processed events')
    for k in tqdm(range(len(procs))):
        proc = new_procs[k]
        for sample in procs_cfg[proc]:
            events[sample] = {}
            events[sample]['eventsProcessed'], events[sample]['cross-section'] = 0, 0

            file_list[sample] = ROOT.vector('string')()
            flist = find_sample_files(inputDir, sample)
            events[sample]['cross-section'] += getMetaInfo(sample, remove=True)
            for filepath in flist:
                events[sample]['eventsProcessed'] += get_processed(filepath)
                file_list[sample].push_back(filepath)
    
    print('\n----->[Info] Getting cuts from dataframe')
    for proc in new_procs:
        print(f'\n----->[Info] From {proc}')
        for sample in procs_cfg[proc]:
            # if not onlyhad and ('p8_ee_WW_ee' in sample or 'p8_ee_WW_mumu' in sample):
            #     continue
            print(f'----->[Info] sample {sample}')
            if is_there_events(sample, inputDir):
                df = ROOT.ROOT.RDataFrame('events', file_list[sample])
                # if onlyhad: df = df.Filter("!ww_leptonic")
                if len(defineList)>0:
                    print('----->[Info] Defining additional columns')
                    for define in defineList: df = df.Define(define, defineList[define])
                for sel in sels:
                    if not sel in cuts: continue
                    print(f'----->[Info] For selection: {sel}')
                    events[sample][sel] = {}
                    events[sample][sel]['cut'], events[sample][sel]['err'] = {}, {}
                    events[sample][sel]['filter'] = {}
                    df1 = None
                    for cut, filter in cuts[sel].items():
                        print(f'----->[Info] For cut {cut}')
                        events[sample][sel]['filter'][cut] = filter
                        indf = cut in df.GetColumnNames()
                        if (len(file_list[sample])>1) and indf:
                            events[sample][sel]['cut'][cut] = get_cut(file_list[sample], cut, doeff=False) #onlyhad)
                        else:
                            if df1==None:
                                df1, events[sample][sel]['cut'][cut] = getcut(df, filter, cut=cut, indf=indf)
                            else:
                                df1, events[sample][sel]['cut'][cut] = getcut(df1, filter, cut=cut, indf=indf)
                        events[sample][sel]['err'][cut] = events[sample][sel]['cut'][cut]**(0.5)
                        if scaled:
                            scale = lumi * 1e6 * events[sample]['cross-section'] / events[sample]['eventsProcessed']
                            events[sample][sel]['cut'][cut] *= scale
                            events[sample][sel]['err'][cut] *= scale
            else:
                for sel in sels:
                    if not sel in cuts: continue
                    events[sample][sel] = {}
                    events[sample][sel]['cut'], events[sample][sel]['err'] = {}, {}
                    events[sample][sel]['filter'] = {}
                    for cut, filter in cuts[sel].items():
                        if cut=='cut0':
                            scale = lumi * 1e6 * events[sample]['cross-section'] / events[sample]['eventsProcessed']
                            events[sample][sel]['cut'][cut] = scale * events[sample]['eventsProcessed']
                            events[sample][sel]['err'][cut] = scale * events[sample]['eventsProcessed']**(0.5)
                        else:
                            events[sample][sel]['cut'][cut] = 0
                            events[sample][sel]['err'][cut] = 0

    dump_json(events, loc_json, 'events')
    
    for sel in sels:
        if not sel in cuts: continue
        if not tot:
            flow, flow_decay = get_flows(procs, procs_cfg, cuts, events, cat, sel, z_decays, H_decays, 
                                        ecm=ecm, json_file=json_file, loc_json=loc_json+f'/{sel}')

            CutFlow(flow, outDir, cat, sel, procs, colors, legend, cuts, cuts_label, ecm=ecm, lumi=lumi, 
                    outName='cutFlow', plot_file=plot_file, sig_scale=sig_scale)
            CutFlowDecays(flow_decay, outDir, cat, sel, h_decays, H_decays, cuts, cuts_label, 
                          ecm=ecm, lumi=lumi, plot_file=plot_file)
        else:
            procs_cat, procs_tot = copy.deepcopy(procs), copy.deepcopy(procs)
            procs_cat[0] = f'Z{cat}H'

            flow, flow_decay = get_flows(procs_cat, procs_cfg, cuts, events, cat, sel, z_decays, H_decays, 
                                         ecm=ecm, json_file=json_file, loc_json=loc_json+f'/{sel}')
            CutFlow(flow, outDir, cat, sel, procs_cat, colors, legend, cuts, cuts_label, 
                    ecm=ecm, lumi=lumi, outName='cutFlow', plot_file=plot_file, sig_scale=sig_scale)
            CutFlowDecays(flow_decay, outDir, cat, sel, H_decays, cuts, cuts_label, 
                          ecm=ecm, lumi=lumi, plot_file=plot_file)
            
            flow_tot, flow_decay_tot = get_flows(procs_tot, procs_cfg, cuts, events, cat, sel, z_decays, H_decays, 
                                                 ecm=ecm, json_file=json_file, loc_json=loc_json+f'/{sel}', tot=True)
            CutFlow(flow_tot, outDir, cat, sel, procs_tot, colors, legend, cuts, cuts_label, ecm=ecm, lumi=lumi, 
                    outName='cutFlow_tot', plot_file=plot_file, sig_scale=sig_scale)
            CutFlowDecays(flow_decay_tot, outDir, cat, sel, H_decays, cuts, cuts_label, ecm=ecm, lumi=lumi, 
                          outName='cutFlow_decays_tot', plot_file=plot_file, suffix='_tot', yMin=-30, yMax=160)
