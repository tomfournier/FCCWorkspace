# Logbook: Internship part

## 26/03/2025

- Start of the Internship
- Aclimating with the environment and doing bibliographic work
- Have to make a bibliographic report by the $22^{\textrm{th}}$ of April

## 21/04/2025

- Ang sent a link to have the analysis code at this [link](https://codimd.web.cern.ch/v-2loZ2BSmSurcYI1v-Nkg)

## 22/04/2025

- Report sent, can start reading the code of the analysis
- Repository successfully cloned but there are troubles in building it
- FCCAnalyses part successfully built but Combined-Limit part have still some problem

## 24/04/2025

- Problem: HiggsTools is not found when doing preselection
  - Solution: When building FCCAnalyses, be sure to be on AngDevOld branch to have HiggsTools

## 25/04/2025

- Problem: tools.utils module is not found
  - Found it on FCCeePhysicsPerformance ([link to the github](https://github.com/Ang-Li-93/FCCeePhysicsPerformance/tree/ZH_recoil)) on the ZH_recoil in ```FCCeePhysicsPerformance/case-studies/higgs/``` folder

## 28/04/2025

- Problem: `userConfig.py` file is not found when running `fccanalysis run` command
  - Solution: have to put it in the same folder as the one we are when doing the command
  - Put `userConfig.py` in FCCWorkplace folder
- At last the selection is running, plots are available there (add link to plots)

## 29/04/2025

- Created a new repository ([link to the github](https://github.com/tomfournier/FCCWorkspace)) to start clean
- Plots seem strange (add a link to plots), have to search for the reason

## 30/04/2025

- Repository cleaned and BDT working
- automatizing the code to not have to change everything each time we change flavor ($\mu\mu$ or $ee$)
- Redoing the preselection without additional signal sample the disrupt the histograms
  - commit `cleaning the repository` and `cleaning the repository v2`

## 01/05/2025

- Making plots visible in the github to consult them online
  - commit `plot v1`
- Making cover letter and update resume for the candidature to the doctoral school for the thesis
- Making slides for Monday meeting with the Higgs team

## 02/05/2025

- Embellishing BDT plots
- Finishing cleaning the repository
- Creating the logbook to keep tracks of activities
- Redoing the preselection with the good `processList` and planning to run the $ee$ channel (Will take some times as the new samples are bigger than the previous one)
  - Preselection takes too much time, will do the preselection in the week-end and do the final analysis on Monday after the team meeting
  - commit ``

# To do:
- Adding plot links to logbook
- Redo the preselection for $\mu\mu$ and $ee$
- Verify yield coherence with article
- Finish slides for Monday meeting
- Finish candidature for doctoral school
- Recontact NPAC prof for reference letter if no answer on Monday
- Forgot others