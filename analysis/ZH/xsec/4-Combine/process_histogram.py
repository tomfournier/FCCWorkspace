import os, time, json, ROOT, argparse, importlib

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--cat',  help='Final state (ee, mumu), qq is not available yet', 
                    choices=['ee', 'mumu'], type=str, default='')
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', 
                    choices=[240, 365], type=int, default=240)
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, event, z_decays, h_decays

categories, ecm = [arg.cat] if arg.cat!='' else ['ee', 'mumu'], arg.ecm
hName = ['zll_recoil_m']
selections = [
    'Baseline'
]

#__________________________________________________________
def getMetaInfo(proc, info='crossSection', 
                procFile="FCCee_procDict_winter2023_IDEA.json"):
    if not ('eos' in procFile):
        procFile = os.path.join(os.getenv('FCCDICTSDIR').split(':')[0], '') + procFile 
    with open(procFile, 'r') as f:
        procDict=json.load(f)
    xsec = procDict[proc][info]
    return xsec

#__________________________________________________________
def get_hist(hName, proc, suffix=''):
    print(f'----->[Info] Getting histogram from \n\t {inputdir}/{proc}{suffix}.root')
    f = ROOT.TFile(f"{inputdir}/{proc}{suffix}.root")
    h = f.Get(hName)
    h.SetDirectory(0)

    xsec = f.Get("crossSection").GetVal()
    if 'HZZ' in proc: 
        xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
        h.Scale((xsec-xsec_inv)/xsec)
    if 'p8_ee_WW_ecm' in proc:
        xsec_ee   = getMetaInfo(proc.replace('WW_ecm', 'WW_ee_ecm'))
        xsec_mumu = getMetaInfo(proc.replace('WW_ecm', 'WW_mumu_ecm'))
        h.Scale((xsec-xsec_ee-xsec_mumu)/xsec)
    f.Close()
    return h

#__________________________________________________________
def concat(h_list, hName):
    print(f'----->[Info] Concatenating {hName}')
    # Get the total number of bins in all histograms
    bins = sum([h.GetNbinsX() for h in h_list])
    h_concat = ROOT.TH1D("h1", "1D Unrolled Histogram", bins, 0.5, bins + 0.5)

    tot = 0
    for hist in h_list: # Loop over all histograms
        for bin in range(1, hist.GetNbinsX() + 1): # Bin indexing starts at 1
            # Get the content and error from the histogram
            content = hist.GetBinContent(bin)
            error   = hist.GetBinError(bin)

            # Set the content and error in the new histogram
            h_concat.SetBinContent(bin+tot, content)
            h_concat.SetBinError(bin+tot,   error)
        tot += hist.GetNbinsX()
    h_concat.SetName(hName)
            
    return h_concat



for cat in categories:
    print(f'----->[Info] Processing histograms for {cat}')

    inputdir  = get_loc(loc.HIST_PREPROCESSED, cat, ecm, '')
    inDir     = get_loc(loc.EVENTS,            cat, ecm, '')+'/'

    samples_bkg = [
        f"p8_ee_WW_ecm{ecm}", f"p8_ee_WW_ee_ecm{ecm}", f"p8_ee_WW_mumu_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
        f"wzp6_ee_ee_Mee_30_150_ecm{ecm}", f"wzp6_ee_mumu_ecm{ecm}", f"wzp6_ee_tautau_ecm{ecm}",
        f"wzp6_gaga_ee_60_ecm{ecm}",       f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
        f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',  f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
        f'wzp6_egamma_eZ_Zee_ecm{ecm}',    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
        f"wzp6_ee_nuenueZ_ecm{ecm}"
    ]
    samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
    samples_sig.extend([f"wzp6_ee_eeH_ecm{ecm}", f"wzp6_ee_mumuH_ecm{ecm}", f'wzp6_ee_ZH_Hinv_ecm{ecm}'])
    procs = event(samples_sig + samples_bkg, inDir)

    for sel in selections:
        print(f'\n----->[Info] Processing histograms for {sel}\n')
        outputdir = get_loc(loc.HIST_PROCESSED,    cat, ecm, sel)
        if not os.path.isdir(f'{outputdir}'):
            os.system(f'mkdir -p {outputdir}')

        for proc in procs:
            print(f'----->[Info] Processing {proc}')
            hists = []
            for histo in hName:
                h_high = get_hist(histo, proc, suffix=f'_{sel}_high_histo')
                h_low  = get_hist(histo, proc, suffix=f'_{sel}_low_histo')
                h      = concat([h_low, h_high], histo)
                print(f'----->[Info] Done concatenating {histo}')
                hists.append(h)
            print(f'----->[Info] {proc} histograms for {cat} channel processed')

            f = ROOT.TFile(f"{outputdir}/{proc}.root", "RECREATE")
            print(f'----->[Info] Saving histograms')
            for hist in hists:
                hist.Write()
            f.Close()
            print(f'----->[Info] Saved histograms in {outputdir}/{proc}.root\n')

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
