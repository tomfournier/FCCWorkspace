#include "FCCAnalyses/defines.h"
#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom.h>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include "functions.h"

#ifndef FCCPhysicsUtils_H
#define FCCPhysicsUtils_H

namespace FCCAnalyses {

// Check if a MC particle is an electron or positron
inline bool isElectron(int pdg) {
    return std::abs(pdg) == 11;
}

// Check if a MC particle is an muon or antimuon
inline bool isMuon(int pdg) {
    return std::abs(pdg) == 13;
}

// Check if a MC particle is an tau or antitau
inline bool isTau(int pdg) {
    return std::abs(pdg) == 15;
}


// Check if a particle is a lepton (e, mu, tau if include_tau = true) 
inline bool isLepton (int pdg, bool inlcude_tau = false) {
    int abs_pdg = std::abs(pdg);
    if(inlcude_tau){
        return (abs_pdg == 11 || abs_pdg == 13 || abs_pdg == 15);
    }
    return (abs_pdg == 11 || abs_pdg == 13);
}



// Check if a MC particle is a quark (u, d, s, c, b, t)
inline bool isQuark(int pdg, bool include_t = false) {
    int abs_pdg = std::abs(pdg);
    if (include_t) {
        return (abs_pdg >= 1 && abs_pdg <= 6);
    }
    return (abs_pdg >= 1 && abs_pdg <= 5);
}



// Check if a MC particle is a photon
inline bool isPhoton(int pdg) {
    return (pdg == 22);
}

// Check if a MC particle is a gluon
inline bool isGluon(int pdg) {
    return (pdg == 21);
}

// Check if a MC particle is a W
inline bool isW(int pdg) {
    return std::abs(pdg) == 24;
}

// Check if a MC particle is a Z boson
inline bool isZ(int pdg) {
    return std::abs(pdg) == 23;
}

// Check if a MC particle is a Higgs boson
inline bool isHiggs(int pdg) {
    return std::abs(pdg) == 25;
}


inline bool isBoson (int pdg, bool include_gluon = false) {
    if (include_gluon) {
        return isPhoton(pdg) || isW(pdg) || isZ(pdg) || isHiggs(pdg) || isGluon(pdg);
    }
    return isPhoton(pdg) || isW(pdg) || isZ(pdg) || isHiggs(pdg);
}


// Check if the particle directly come from the initial state
inline bool isFromInitial(int pdg) {
    return pdg == -999;
}


inline edm4hep::MCParticleData getParent(edm4hep::MCParticleData particle, Vec_mc mc, Vec_i parents) {
    edm4hep::MCParticleData result;

    int pb = parents[particle.parents_begin];
    int pe = parents[particle.parents_end];

    if (pe - pb > 1) return result;
    result = mc[pb];
    
    return result;
}


// Get parent particle PDG
inline int getParentPDG(int idx, Vec_mc mc, Vec_i parents){
    
    if (idx < 0 || idx > parents.size()) {
        return -999;
    }
    
    const auto& p = mc[idx];
    int pb = parents[p.parents_begin];
    int pe = parents[p.parents_end];

    if (pe - pb > 1) return -999;
    return mc[pb].PDG;
}

inline int getParentPDG(edm4hep::MCParticleData particle, Vec_mc mc, Vec_i parents) {
    int pb = particle.parents_begin;
    int pe = particle.parents_end;

    if (pe - pb > 1) return -999;
    int parent_idx = parents[pb];
    return mc[parent_idx].PDG;
}


// Return the index of the parent particle
inline Vec_i getParentID(Vec_i mcind, Vec_mc mc, Vec_i parents){
  Vec_i result;

  for (size_t i = 0; i < mcind.size(); ++i) {

    int ind = mcind.at(i);
    auto p = mc.at(ind);

    if (ind<0){
      result.push_back(-999);
      continue;
    }
    if (p.parents_end - p.parents_begin > 1) { result.push_back(-999); }
    else { result.push_back(parents.at(p.parents_begin)); }
  
    }
  return result;
}


// returns a vector with the indices (in the Particle block) of the daughters of the particle i
inline Vec_i getDaughtersID(int i, Vec_mc mc, Vec_i daughters) {

  Vec_i res;
  if ( i < 0 || i >= mc.size() ) return res;

  int db = mc.at(i).daughters_begin ;
  int de = mc.at(i).daughters_end;
  if  ( db == de ) return res;   // particle is stable
  for (int id = db; id < de; id++) {
     res.push_back( daughters[id] ) ;
  }
  return res;
}

inline bool isStable(edm4hep::MCParticleData mc, Vec_i daughters) {
    int db = mc.daughters_begin;
    int de = mc.daughters_end;
    return db == de;
}

// returns a vector with the PDG ID of the daughters of the particle i
inline Vec_i getDaughtersPDG(int i, Vec_mc mc, Vec_i daughters) {

    Vec_i res;
    if ( i < 0 || i >= mc.size() ) return res;

    int db = mc.at(i).daughters_begin ;
    int de = mc.at(i).daughters_end;
    if  ( db == de ) return res;   // particle is stable
    for (int id = db; id < de; id++) {
        int daughter_idx = daughters[id];
        res.push_back( mc[daughter_idx].PDG ) ;
  }
  return res;
}


inline int getLeptonOrigin(
    const edm4hep::MCParticleData &p,
    const Vec_mc &mc,
    const Vec_i &parents,
    const bool include_tau) {


    int pdg = p.PDG;
    if ( !isLepton(pdg, include_tau) ) return -1;

    int pdg_parent = getParentPDG(p, mc, parents);

    // Directly come from the initial state
    if (isFromInitial(pdg_parent)) return 0;

    // Come from a boson
    if (isBoson(pdg_parent, false)) return pdg_parent;

    // Come from the tau or muon leptonic decay
    if ((isElectron(pdg) || isMuon(pdg)) && isTau(pdg_parent)) return pdg_parent;
    if (isElectron(pdg) && isMuon(pdg_parent)) return pdg_parent;

    // Same particle -> iterate
    if ( pdg_parent == pdg  ) {
        int index = parents.at(p.parents_begin);
        return getLeptonOrigin(mc.at(index), mc, parents, include_tau);
    }

    // Come from a hadron decay
    return pdg_parent;
}


inline int getQuarkOrigin(
    const edm4hep::MCParticleData &p,
    const Vec_mc &mc,
    const Vec_i &parents,
    const bool include_t) {


    int pdg = p.PDG;
    if ( !isQuark(pdg, include_t) ) return -1;

    int pdg_parent = getParentPDG(p, mc, parents);

    // Directly come from the initial state
    if (isFromInitial(pdg_parent)) return 0;

    // Come from a boson
    if (isBoson(pdg_parent, true)) return pdg_parent;

    // Same particle -> iterate
    if ( pdg_parent == pdg  ) {
        int index = parents.at(p.parents_begin);
        return getQuarkOrigin(mc.at(index), mc, parents, include_t);
    }

    // Couldn't find the quark origin
    return -2;
}

inline Vec_rp smearPhotonEnergyResolution(Vec_rp in, Vec_i in_idx, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc) {

    // avoid events that have the extra soft photon and screws up the MC/RECO collections
    if(reco.size() != recind.size()) return in;


    // IDEA default
    auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.03^2 + 0.002^2)", 0, 1000); // 0.03=constant term (A), 0.005=stochastic term (B) 0.002=noise term (C)
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.01^2 + 0.002^2)", 0, 1000); // stochastic term S=1%
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.02^2 + 0.002^2)", 0, 1000); // stochastic term S=2%
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.05^2 + 0.002^2)", 0, 1000); // stochastic term S=5%
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.10^2 + 0.002^2)", 0, 1000); // stochastic term S=10%
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.25^2 + 0.002^2)", 0, 1000); // stochastic term S=25%
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.50^2 + 0.002^2)", 0, 1000); // stochastic term S=50%

    // Dual readout
    //auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.01^2 + x*0.11^2 + 0.05^2)", 0, 1000);

    float scale = 1.0; // additional scaling

    Vec_rp result;
    result.reserve(in.size());
    for(int i = 0; i < in.size(); ++i) {
        auto & p = in[i];
        edm4hep::ReconstructedParticleData p_new = p;
        TLorentzVector reco_p4;
        reco_p4.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);

        int mc_index = mcind[recind[in_idx[i]]]; //DOES NOT WORK???
        if(mc_index >= 0 && mc_index < (int)mc.size()) {
            TLorentzVector mc_p4;
            mc_p4.SetXYZM(mc.at(mc_index).momentum.x, mc.at(mc_index).momentum.y, mc.at(mc_index).momentum.z, mc.at(mc_index).mass);
            float new_energy = reco_p4.E();
            if(abs(reco_p4.E()-mc_p4.Energy())/mc_p4.Energy() > 5) { // avoid extreme variations
                std::cout << "MC-RECO MISMATCH: MC_E=" << mc_p4.Energy() << " RECO_E=" << reco_p4.Energy() << std::endl;
            }
            else {
                float res = ecal_res_formula->Eval(mc_p4.Energy())*scale;
                new_energy = gRandom->Gaus(mc_p4.Energy(), res);
            }

            p_new.energy = new_energy;
            // recompute momentum magnitude
            float smeared_p = std::sqrt(p_new.energy * p_new.energy - reco_p4.M() * reco_p4.M());

            // recompute mom x, y, z using original reco particle direction
            p_new.momentum.x = smeared_p * std::sin(reco_p4.Theta()) * std::cos(reco_p4.Phi());
            p_new.momentum.y = smeared_p * std::sin(reco_p4.Theta()) * std::sin(reco_p4.Phi());
            p_new.momentum.z = smeared_p * std::cos(reco_p4.Theta());
        }
        result.push_back(p_new);
    }
    return result;
}

}

#endif