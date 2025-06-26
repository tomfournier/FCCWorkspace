import os
import ROOT
import importlib, time, argparse

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--ecm', help='Center of mass energy (240, 365)', choices=[240, 365], type=int, default=240)
parser.add_argument('--recoil120', help='Cut with 120 GeV < recoil mass < 140 GeV instead of 100 GeV < recoil mass < 150 GeV', action='store_true')
parser.add_argument('--miss', help='Add the cos(theta_miss) < 0.98 cut', action='store_true')
parser.add_argument('--bdt', help='Add cos(theta_miss) cut in the training variables of the BDT', action='store_true')
parser.add_argument('--leading', help='Add the p_leading and p_subleading cuts', action='store_true')
arg = parser.parse_args()

userConfig = importlib.import_module('userConfig')
from userConfig import loc, get_loc, select, z_decays, h_decays

ecm, sel = arg.ecm, select(arg.recoil120, arg.miss, arg.bdt, arg.leading)

samples_bkg = [
    f"p8_ee_WW_ecm{ecm}", f"p8_ee_ZZ_ecm{ecm}",
    f"wzp6_ee_ee_Mee_30_150_ecm{ecm}", f"wzp6_ee_mumu_ecm{ecm}", f"wzp6_ee_tautau_ecm{ecm}",
    f"wzp6_gaga_ee_60_ecm{ecm}",       f"wzp6_gaga_mumu_60_ecm{ecm}", f"wzp6_gaga_tautau_60_ecm{ecm}", 
    f'wzp6_egamma_eZ_Zmumu_ecm{ecm}',  f'wzp6_gammae_eZ_Zmumu_ecm{ecm}',
    f'wzp6_egamma_eZ_Zee_ecm{ecm}',    f'wzp6_gammae_eZ_Zee_ecm{ecm}',
    f"wzp6_ee_nuenueZ_ecm{ecm}"
]
samples_sig = [f"wzp6_ee_{x}H_H{y}_ecm{ecm}" for x in z_decays for y in h_decays]
for i in ['ee', 'mumu']:
    samples_sig.append(f"wzp6_ee_{i}H_ecm{ecm}")
samples_sig.append(f'wzp6_ee_ZH_Hinv_ecm{ecm}')

samples = samples_sig + samples_bkg
procs = samples

hName = ['recoil_m_mva', 'zll_recoil_m_mva_high', 'zll_recoil_m_mva_low', 'zll_recoil']



#__________________________________________________________
def getMetaInfo(proc):
    fIn = ROOT.TFile(f"{inputdir}/{proc}.root")
    xsec = fIn.Get("crossSection").GetVal()
    return xsec

#__________________________________________________________
def get_hist(hName, proc):
    print(f'----->[Info] Getting histogram from \n\t {inputdir}/{proc}.root')
    f = ROOT.TFile(f"{inputdir}/{proc}.root")
    h = f.Get(hName)
    h.SetDirectory(0)

    xsec = f.Get("crossSection").GetVal()
    if 'HZZ' in proc: 
        xsec_inv = getMetaInfo(proc.replace('HZZ', 'Hinv'))
        h.Scale((xsec-xsec_inv)/xsec)
    f.Close()
    return h

#__________________________________________________________
def unroll(hist, hName):
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
    h1.SetName(hName)
            
    return h1

inputdir  = get_loc(loc.HIST_PREPROCESSED, '', ecm, sel)
outputdir = get_loc(loc.HIST_PROCESSED, '', ecm, sel)

if not os.path.isdir(f'{outputdir}'):
    os.system(f'mkdir -p {outputdir}')

print('----->[Info] Processing histograms for Combine')
for proc in procs:
    hists = []
    print(f'----->[Info] Processing {proc}')
    for cat in ['ee', 'mumu']:
        for histo in hName:
            h = get_hist(f'{cat}_{histo}', proc)
            if histo=='recoil_m_mva': h = unroll(h, f'{cat}_{histo}')
            hists.append(h)
        print(f'----->[Info] {proc} histograms for {cat} channel processed')

    f = ROOT.TFile(f"{outputdir}/{proc}.root", "RECREATE")
    print(f'----->[Info] Saving histograms')
    for hist in hists:
        hist.Write()
    f.Close()
    print(f'----->[Info] Saved histograms in {outputdir}/{proc}.root')

print('\n\n------------------------------------\n')
print(f'Time taken to run the code: {time.time()-t1:.1f} s')
print('\n------------------------------------\n\n')
