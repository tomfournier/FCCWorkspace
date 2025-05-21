import os
import numpy as np
import ROOT
import argparse
import importlib

userConfig = importlib.import_module('userConfig')

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, help="Target pseudodata", default="bb")
parser.add_argument("--run", help="Run combine", action='store_true')
parser.add_argument("--pert", type=float, help="Target pseudodata size", default=1.0)
parser.add_argument("--bbb", help="Enable bin-by-bin statistics (BB)", action='store_true')
parser.add_argument("--freezeBackgrounds", help="Freeze backgrounds", action='store_true')
parser.add_argument("--floatBackgrounds", help="Float backgrounds", action='store_true')
parser.add_argument("--plot_dc", help="Plot datacard", action='store_true')

args = parser.parse_args()

def getMetaInfo(proc):
    fIn = ROOT.TFile(f"{inputDir}/{proc}.root")
    xsec = fIn.Get("crossSection").GetVal()
    
    # if "HZZ" in proc: # HZZ contains invisible, remove xsec
    #     xsec_inv = getMetaInfo(proc.replace("mumuH", "ZH").replace("HZZ", "Hinv"))
    #     print("REMOVE INV FROM ZZ XSEC", proc, xsec, xsec-xsec_inv)
    #     xsec = xsec - xsec_inv
    return xsec

def removeNegativeBins(hist):
    totNeg, tot = 0., 0.
    if "TH1" in hist.ClassName():
        pass
    elif "TH2" in hist.ClassName():
        pass

    elif "TH3" in hist.ClassName():
        nbinsX = hist.GetNbinsX()
        nbinsY = hist.GetNbinsY()
        nbinsZ = hist.GetNbinsZ()
        for x in range(1, nbinsX + 1):
            for y in range(1, nbinsY + 1):
                for z in range(1, nbinsZ + 1):
                    content = hist.GetBinContent(x, y, z)
                    error = hist.GetBinError(x, y, z)  # Retrieve bin error
                    tot += content
                    if content < 0:
                        totNeg += content
                        hist.SetBinContent(x, y, z, 0)
                        hist.SetBinError(x, y, z, 0)
    if totNeg != 0:
        print(f"WARNING: TOTAL {tot}, NEGATIVE {totNeg}, FRACTION {totNeg/tot}")
    return hist

def getSingleHist(hName, proc):
    if "Hinv" in proc:
        proc = proc.replace("wzp6", "wz3p6")
    fIn = ROOT.TFile(f"{inputDir}/{proc}.root")
    h = fIn.Get(hName)
    h.SetDirectory(0)
    fIn.Close()
    h = removeNegativeBins(h)
    return h

def getHists(hName, procName):
    hist = None
    procs = procs_cfg[procName]
    for proc in procs:
        h = getSingleHist(hName, proc)
        if hist == None:
            hist = h
        else:
            hist.Add(h)
    if procName in proc_scales:
        hist.Scale(proc_scales[procName])
        print(f"SCALE {procName} with factor {proc_scales[procName]}")
    return hist

def unroll(hist, rebin=1):
    if "TH1" in hist.ClassName():
        hist = hist.Rebin(rebin)
        hist = removeNegativeBins(hist)
        return hist

    elif "TH2" in hist.ClassName():
        # Get the number of bins in X and Y
        n_bins_x = hist.GetNbinsX()
        n_bins_y = hist.GetNbinsY()

        # Create a 1D histogram to hold the unrolled data
        n_bins_1d = n_bins_x * n_bins_y
        h1 = ROOT.TH1D("h1", "1D Unrolled Histogram", n_bins_1d, 0.5, n_bins_1d + 0.5)

        # Loop over all bins in the 2D histogram
        for bin_x in range(1, n_bins_x + 1):  # Bin indexing starts at 1
            for bin_y in range(1, n_bins_y + 1):
                # Calculate the global bin number for the 1D histogram
                bin_1d = (bin_y - 1) * n_bins_x + bin_x

                # Get the content and error from the 2D histogram
                content = hist.GetBinContent(bin_x, bin_y)
                error = hist.GetBinError(bin_x, bin_y)

                # Set the content and error in the 1D histogram
                h1.SetBinContent(bin_1d, content)
                h1.SetBinError(bin_1d, error)
        return h1


    elif "TH3" in hist.ClassName():
        # Get binning information
        nbinsX = hist.GetNbinsX()
        nbinsY = hist.GetNbinsY()
        nbinsZ = hist.GetNbinsZ()
        nbins1D = nbinsX * nbinsY * nbinsZ

        # Create a 1D histogram with the correct number of bins
        h1 = ROOT.TH1D("h1_unrolled", "Unrolled 3D Histogram", nbins1D, 0, nbins1D)

        # Fill the 1D histogram by unrolling the 3D histogram
        bin1D = 1  # ROOT bins are 1-based
        for x in range(1, nbinsX + 1):
            for y in range(1, nbinsY + 1):
                for z in range(1, nbinsZ + 1):
                    content = hist.GetBinContent(x, y, z)
                    error = hist.GetBinError(x, y, z)  # Retrieve bin error
                    
                    if content < 0:
                        print("WARNING NEGATIVE CONTENT", content, hist.GetName())
                        content = 0
                        error = 0

                    h1.SetBinContent(bin1D, content)
                    h1.SetBinError(bin1D, error)  # Set the bin error
                    bin1D += 1  # Increment the 1D bin index
        return h1

    else:
        return hist

def make_pseudodata(procs, target="bb", variation=1.0):

    print(f'----->[Info] Making pseudo data for {target} channel')
    print(f'----->[Info] Perturbation of the cross-section: {(variation-1)*100:.2f} %')

    xsec_tot = 0 # total cross-section
    xsec_target = 0  # nominal cross-section of the target process
    xsec_rest = 0 # cross-section of the rest

    sigProcs = procs_cfg[procs[0]] # get all signal processes
    for h_decay in h_decays:
        xsec = 0.
        for z_decay in z_decays:
            proc = f'wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}'
            if not proc in sigProcs:
                continue
            xsec += getMetaInfo(proc)
        xsec_tot += xsec
        if h_decay != target:
            xsec_rest += xsec
        else:
            xsec_target += xsec

    xsec_new = variation*xsec_tot
    xsec_delta = xsec_new - xsec_tot # difference in cross-section
    print(f'----->[Info] Cross-section of {target} channel: {xsec_target*1e6:.1f} fb')
    scale_target = (xsec_target + xsec_delta)/xsec_target
    scale_rest = (xsec_rest - xsec_delta)/xsec_rest

    print(f"----->[Info] Total cross-section of the signal: {xsec_tot*1e6:.1f} fb")
    print(f"----->[Info] New cross-section: {xsec_new*1e6:.1f} fb")
    print(f"----->[Info] Difference between new and previous cross-section: {xsec_delta*1e6:.1f} fb")
    print(f"----->[Info] Scale of the {target} channel: {scale_target:.3f}")
    print(f"----->[Info] Scale of the rest: {scale_rest:.3f}")

    hist_pseudo = None # start with all backgrounds
    for proc in procs[1:]:
        h = getHists(hName, proc)
        if hist_pseudo == None:
            hist_pseudo = h
        else:
            hist_pseudo.Add(h)

    xsec_tot_new = 0
    for h_decay in h_decays:
        xsec = 0.
        hist = None
        for z_decay in z_decays:
            proc = f'wzp6_ee_{z_decay}H_H{h_decay}_ecm{ecm}'
            if not proc in sigProcs:
                continue
            xsec += getMetaInfo(proc)
            h = getSingleHist(hName, proc)
            if hist == None:
                hist = h
            else:
                hist.Add(h)
        if h_decay == target:
            hist.Scale(scale_target)
            xsec_tot_new += xsec*scale_target
        else:
            hist.Scale(scale_rest)
            xsec_tot_new += xsec*scale_rest
        hist_pseudo.Add(hist)
    
    print(f'----->[CROSS-CHECK] This quantity {xsec_tot*1e6:.4f} must be equal to this one {xsec_tot_new*1e6:.4f}')
    print(f'\tDifference: {np.abs(xsec_tot-xsec_tot_new)*1e6:.4f}')
    return hist_pseudo

if __name__ == "__main__":

    ecm = userConfig.ecm
    outDir = userConfig.loc.BIAS_HIST
    sigma = 1
    bkg_unc = 1.01

    z_decays = ['mumu'] # ["qq", "bb", "cc", "ss", "ee", "mumu", "nunu" , "tautau"]
    h_decays = ["bb", "cc", "gg", "ss", "mumu", "tautau", "ZZ", "WW", "Za", "aa"] # , "inv"]

    # bbb = " --binByBinStat" if args.bbb else ""

    procs_cfg = {
        "ZH"        : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in z_decays for y in h_decays],
        "ZmumuH"    : [f'wzp6_ee_mumuH_H{y}_ecm{ecm}' for y in h_decays],
        "ZeeH"      : [f'wzp6_ee_{x}H_H{y}_ecm{ecm}' for x in ["ee"] for y in h_decays],
        "WW"        : [f'p8_ee_WW_ecm{ecm}'],
        "ZZ"        : [f'p8_ee_ZZ_ecm{ecm}'],
        "Zgamma"    : [f'wzp6_ee_tautau_ecm{ecm}', f'wzp6_ee_mumu_ecm{ecm}',
                       f'wzp6_ee_ee_Mee_30_150_ecm{ecm}'],
        "Rare"      : [f'wzp6_egamma_eZ_Zmumu_ecm{ecm}', f'wzp6_gammae_eZ_Zmumu_ecm{ecm}', 
                       f'wzp6_gaga_mumu_60_ecm{ecm}', f'wzp6_egamma_eZ_Zee_ecm{ecm}', 
                       f'wzp6_gammae_eZ_Zee_ecm{ecm}', f'wzp6_gaga_ee_60_ecm{ecm}', 
                       f'wzp6_gaga_tautau_60_ecm{ecm}', f'wzp6_ee_nuenueZ_ecm{ecm}'],
    }

    procs_scales_250 = {
        "ZH": 1.048,
        "WW": 0.971,
        "ZZ": 0.939,
        "Zgamma": 0.919,
    }
    
    procs_scales_polL = {
        "ZH": 1.554,
        "WW": 2.166,
        "ZZ": 1.330,
        "Zgamma": 1.263,
    }
    
    procs_scales_polR = {
        "ZH": 1.047,
        "WW": 0.219,
        "ZZ": 1.011,
        "Zgamma": 1.018,
    }
    
    proc_scales = procs_scales_250 ## change fit to ASIMOV -t -1 !!!
    proc_scales = {}

    final_state = userConfig.final_state
    p = -1 if args.floatBackgrounds else 1
    hists = []

    if final_state == "ee" or final_state == "mumu":

        ## NOMINAL
        inputDir = userConfig.loc.MODEL
        hName, rebin = f'{final_state}_recoil_m_mva', 1

        procs = ["ZH", "WW", "ZZ", "Zgamma", "Rare"] # first must be signal
        procs_idx = [0, p*1, p*2, p*3, p*4]

        for proc in procs:
            h = getHists(hName, proc)
            h = unroll(h, rebin=rebin)
            h.SetName(f"{final_state}_{proc}")
            hists.append(h)

        hist_pseudo = make_pseudodata(procs, target=args.target, variation=args.pert)
        hist_pseudo = unroll(hist_pseudo, rebin=rebin)
        hist_pseudo.SetName(f"{final_state}_data_{args.target}")
        hists.append(hist_pseudo)

    if not os.path.isdir(f'{outDir}/datacard'):
        os.system(f'mkdir -p {outDir}/datacard')

    print('----->[Info] Saving pseudo histograms')
    fOut = ROOT.TFile(f"{outDir}/datacard/datacard_{args.target}.root", "RECREATE")
    for hist in hists:
        hist.Write()
    fOut.Close()
    print(f'----->[Info] Histograms saved in {outDir}/datacard/datacard_{args.target}.root')

    print('----->[Info] Making datacard')

    procs_str = "".join([f"{proc:{' '}{'<'}{12}}" for proc in procs])
    cats_procs_str = "".join([f"{final_state:{' '}{'<'}{12}}" for _ in range(len(procs))])
    cats_procs_idx_str = "".join([f"{str(proc_idx):{' '}{'<'}{12}}" for proc_idx in procs_idx])
    rates_procs = "".join([f"{'-1':{' '}{'<'}{12}}"]*len(procs))

    dc = ""
    dc += "imax *\n"
    dc += "jmax *\n"
    dc += "kmax *\n"
    dc += "##########################################################################\n"
    dc += f"shapes *        * datacard_{args.target}.root $CHANNEL_$PROCESS\n"
    dc += f"shapes data_obs * datacard_{args.target}.root $CHANNEL_data_{args.target}\n"
    dc += "##########################################################################\n"
    dc += f"bin                   {final_state}\n"
    dc += "observation            -1\n"
    dc += "##########################################################################\n"
    dc += f"bin                   {cats_procs_str}\n"
    dc += f"process               {procs_str}\n"
    dc += f"process               {cats_procs_idx_str}\n"
    dc += f"rate                  {rates_procs}\n"
    dc += "##########################################################################\n"

    if not args.freezeBackgrounds and not args.floatBackgrounds:
        for i, proc in enumerate(procs):
            if i == 0: continue # no signal

            dc_tmp = f"{f'norm_{proc}':{' '}{'<'}{15}} {'lnN':{' '}{'<'}{5}} "
            for proc1 in procs:
                if proc==proc1:
                    val = str(bkg_unc)
                else:
                    val = '-'
                dc_tmp += f"{val:{' '}{'<'}{12}}"
            dc += f'{dc_tmp}\n'
    else:
        dc_tmp = f"{f'norm_{proc}':{' '}{'<'}{15}} {'lnN':{' '}{'<'}{10}} "
        for proc1 in procs:
                if proc==procs[0]:
                    val = str(1.000000005)
                else:
                    val = '-'
                dc_tmp += f"{val:{' '}{'<'}{12}}"
        dc += dc_tmp

    print('----->[Info] Saving datacard')
    f = open(f"{outDir}/datacard/datacard_{args.target}.txt", 'w')
    f.write(dc)
    f.close()
    print(f'----->[Info] Saved datacard in {outDir}/datacard/datacard_{args.target}.txt')

    if args.plot_dc:
        print(dc)

    if args.run:
        cmd = f"python3 4-Combine/fit.py --outputdir {outDir} --target {args.target}"
        os.system(cmd)