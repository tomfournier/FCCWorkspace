#include "FCCAnalyses/defines.h"
#include <TF1.h>
#include <TLorentzVector.h>
#include <TRandom.h>
#include <csignal>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include "TVector3.h"
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

    if (particle.parents_begin < 0 || particle.parents_begin >= (int)parents.size()) return result;
    if (particle.parents_end < 0 || particle.parents_end >= (int)parents.size()) return result;
    
    int pb = parents[particle.parents_begin];
    int pe = parents[particle.parents_end];

    if (pe - pb > 1) return result;
    if (pb < 0 || pb >= (int)mc.size()) return result;
    result = mc[pb];
    
    return result;
}

inline Vec_mc getParent(Vec_mc particles, Vec_mc mc, Vec_i parents){
    Vec_mc result;

    for (auto particle : particles) {
        if (particle.parents_begin < 0 || particle.parents_begin >= (int)parents.size()) continue;
        int pb = parents[particle.parents_begin];
        if (pb < 0 || pb >= (int)mc.size()) continue;
        result.push_back(mc.at(pb));
    }
    return result;
}


// Get parent particle PDG
inline int getParentPDG(int idx, Vec_mc mc, Vec_i parents){
    
    if (idx < 0 || idx >= (int)mc.size()) {
        return -9999;
    }
    
    const auto& p = mc[idx];
    if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) return -9999;
    if (p.parents_end < 0 || p.parents_end >= (int)parents.size()) return -9999;
    
    int pb = parents[p.parents_begin];
    int pe = parents[p.parents_end];

    if (pe - pb > 1) return -999;
    if (pb < 0 || pb >= (int)mc.size()) return -9999;
    return mc[pb].PDG;
}

inline int getParentPDG(edm4hep::MCParticleData particle, Vec_mc mc, Vec_i parents) {
    int pb = particle.parents_begin;
    int pe = particle.parents_end;

    if (pe - pb > 1) return -999;
    if (pb < 0 || pb >= (int)parents.size()) return -9999;
    int parent_idx = parents[pb];
    if (parent_idx < 0 || parent_idx >= (int)mc.size()) return -9999;
    return mc[parent_idx].PDG;
}


// Return the index of the parent particle
// inline Vec_i getParentID(Vec_i mcind, Vec_mc mc, Vec_i parents){
//   Vec_i result;

//   for (size_t i = 0; i < mcind.size(); ++i) {

//     int ind = mcind.at(i);
//     if (ind < 0 || ind >= (int)mc.size()){
//         result.push_back(-9999);
//         continue;
//     }

//     auto p = mc.at(ind);
    
//     if (p.parents_end - p.parents_begin > 1) { result.push_back(-999); }
//     else {
//         if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) {
//             result.push_back(-9999);
//         } else {
//             result.push_back(parents.at(p.parents_begin));
//         }
//     }
  
//     }
//   return result;
// }


inline Vec_i getParentID(Vec_i ind, Vec_i mcind, Vec_mc mc, Vec_i parents){
    Vec_i result;

    for (int i : ind) {

        if (i < 0 || i >= (int)mcind.size()) {
            result.push_back(-9999);
            continue;
        }
        int idx = mcind.at(i);

        if (idx < 0 || idx >= (int)mc.size()) {
            result.push_back(-9999);
            continue;
        }
        auto p = mc.at(idx);

        if (p.parents_end - p.parents_begin > 1) result.push_back(-999);
        else {
            if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) {
                result.push_back(-9999);
            } else {
                result.push_back(parents.at(p.parents_begin));
            }
        }
    }
    return result;
}


inline Vec_i getParentID(Vec_i ind, Vec_mc mc, Vec_i parents){
    Vec_i result;

    for (int i : ind) {

        if (i < 0 || i >= (int)mc.size()) {
            result.push_back(-9999);
            continue;
        }
        auto p = mc.at(i);

        if (p.parents_end - p.parents_begin > 1) result.push_back(-999);
        else {
            if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) {
                result.push_back(-9999);
            } else {
                result.push_back(parents.at(p.parents_begin));
            }
        }
    }
    return result;
}

// Get parent IDs from a vector of MC particles
inline Vec_i getParentID(Vec_mc particles, Vec_mc mc, Vec_i parents){
    Vec_i result;

    for (auto p : particles) {

        if (p.parents_end - p.parents_begin > 1) result.push_back(-999);
        else {
            if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) {
                result.push_back(-9999);
            } else {
                result.push_back(parents.at(p.parents_begin));
            }
        }
    }
    return result;
}


inline int getParentID(int i, Vec_mc mc, Vec_i parents){

    auto p = mc.at(i);

    // invalid indice
    if (i < 0 || i >= (int)mc.size()) return -9999;

    // more than one parent
    if (p.parents_end - p.parents_begin > 1) return -999;
    
    // check if parent index is within bounds
    if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) return -9999;
    return parents.at(p.parents_begin); 
}

inline int getParentID(edm4hep::MCParticleData p, Vec_mc mc, Vec_i parents){
    if (p.parents_end - p.parents_begin > 1) return -999;  // more than one parent
    // check if parent index is within bounds
    if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) return -9999;
    return parents.at(p.parents_begin); 
}


// returns a vector with the indices (in the Particle block) of the daughters of the particle i
inline Vec_i getDaughtersID(int i, Vec_mc mc, Vec_i daughters) {

  Vec_i res;
  if ( i < 0 || i >= (int)mc.size() ) return res;

  int db = mc.at(i).daughters_begin ;
  int de = mc.at(i).daughters_end;
  if  ( db == de ) return res;   // particle is stable
  for (int id = db; id < de; id++) {
     if (id < 0 || id >= (int)daughters.size()) continue;
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
    if ( i < 0 || i >= (int)mc.size() ) return res;

    int db = mc.at(i).daughters_begin ;
    int de = mc.at(i).daughters_end;
    if  ( db == de ) return res;   // particle is stable
    for (int id = db; id < de; id++) {
        if (id < 0 || id >= (int)daughters.size()) continue;
        int daughter_idx = daughters[id];
        if (daughter_idx < 0 || daughter_idx >= (int)mc.size()) continue;
        res.push_back( mc[daughter_idx].PDG ) ;
  }
  return res;
}


inline int getLeptonOrigin(
    edm4hep::MCParticleData p,
    Vec_mc &mc,
    Vec_i &parents,
    bool include_tau = false) {


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
        if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) return -9999;
        int index = parents.at(p.parents_begin);
        if (index < 0 || index >= (int)mc.size()) return -9999;
        return getLeptonOrigin(mc.at(index), mc, parents, include_tau);
    }

    // Come from a hadron decay
    return pdg_parent;
}


inline Vec_i getLeptonOrigin(
    Vec_mc leptons,
    Vec_mc &mc,
    Vec_i &parents,
    bool include_tau = false) {

    Vec_i result;

    for (auto & p : leptons){

        int pdg = p.PDG;
        if ( !isLepton(pdg, include_tau) ) { result.push_back(-1); continue; }

        int pdg_parent = getParentPDG(p, mc, parents);

        // Directly come from the initial state
        if (isFromInitial(pdg_parent)) { result.push_back(0); continue; }

        // Come from a boson
        if (isBoson(pdg_parent, false)) { result.push_back(pdg_parent); continue; }

        // Come from the tau or muon leptonic decay
        if ((isElectron(pdg) || isMuon(pdg)) && isTau(pdg_parent)) { result.push_back(pdg_parent); continue; }
        if (isElectron(pdg) && isMuon(pdg_parent)) { result.push_back(pdg_parent); continue; }

        // Same particle -> iterate
        if ( pdg_parent == pdg  ) {
            if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) { result.push_back(-9999); continue; };
            int index = parents.at(p.parents_begin);
            if (index < 0 || index >= (int)mc.size()) { result.push_back(-9999); continue; }
            result.push_back(getLeptonOrigin(mc.at(index), mc, parents, include_tau));
            continue;
        }
        // Come from a hadron decay
        result.push_back(pdg_parent);
    }
    return result;
}


inline int getPhotonOrigin(
    edm4hep::MCParticleData p,
    Vec_mc mc,
    Vec_i parents,
    bool include_tau = false) {

    int pdg = p.PDG;
    if (!isPhoton(pdg)) return -1;

    int pdg_parent = getParentPDG(p, mc, parents);

    // come from Higgs decay
    if (isHiggs(pdg_parent)) return pdg_parent;

    // Come from FSR (optinally include tau in FSR)
    if (isMuon(pdg_parent) || (isTau(pdg_parent) && include_tau)) return 0;

    if (isElectron(pdg_parent)) {
        
        // verify that photon comes from FSR
        int parent_idx = getParentID(p, mc, parents);
        if (parent_idx < 0 || parent_idx >= (int)mc.size()) return -1;  // error case
        auto parent = mc.at(parent_idx);

        // from ISR
        if (parent.parents_begin == 0) return 909;
        // from FSR
        else return 0;
    }

    // coming radiation or decay of other particles
    return pdg_parent;
}


inline Vec_i getPhotonOrigin(
    Vec_mc photons,
    Vec_mc mc,
    Vec_i parents,
    bool include_tau = false) {

    Vec_i result;

    for (auto & p : photons) {

        int pdg = p.PDG;
        if (!isPhoton(pdg)) { result.push_back(-1); continue; }

        int pdg_parent = getParentPDG(p, mc, parents);

        // come from Higgs decay
        if (isHiggs(pdg_parent)) { result.push_back(pdg_parent); continue;}

        // Come from FSR (optinally include tau in FSR)
        if (isMuon(pdg_parent) || (isTau(pdg_parent) && include_tau)) {
           result.push_back(0); continue;
        }

        if (isElectron(pdg_parent)) {
            
            // verify that photon comes from FSR
            int parent_idx = getParentID(p, mc, parents);
            if (parent_idx < 0 || parent_idx >= (int)mc.size()) {
                result.push_back(-9999);  // error case
                continue;
            }
            auto parent = mc.at(parent_idx);

            // from ISR
            if (parent.parents_begin == 0) {result.push_back(909); continue; }
            // from FSR
            else { result.push_back(0); continue; }
        }
        // coming radiation or decay of other particles
        result.push_back(pdg_parent);
    }
    return result;
}


// check if photon comes from ISR (to use with the result from getPhotonOrigin)
inline bool fromISR(int origin){
    return origin == 909;
}

// check if photon comes from FSR (to use with the result from getPhotonOrigin)
inline bool fromFSR(int origin){
    return origin == 0;
}


inline Vec_i fromISR(Vec_i origins){
    Vec_i result;
    for (int origin : origins) {
        result.push_back(origin == 909 ? 1 : 0);
    }
    return result;
}

inline Vec_i fromFSR(Vec_i origins){
    Vec_i result;
    for (int origin : origins) {
        result.push_back(origin == 0 ? 1 : 0);
    }
    return result;
}


inline int getQuarkOrigin(
    edm4hep::MCParticleData p,
    Vec_mc &mc,
    Vec_i &parents,
    bool include_t) {


    int pdg = p.PDG;
    if ( !isQuark(pdg, include_t) ) return -1;

    int pdg_parent = getParentPDG(p, mc, parents);

    // Directly come from the initial state
    if (isFromInitial(pdg_parent)) return 0;

    // Come from a boson
    if (isBoson(pdg_parent, true)) return pdg_parent;

    // Same particle -> iterate
    if ( pdg_parent == pdg  ) {
        if (p.parents_begin < 0 || p.parents_begin >= (int)parents.size()) return -9999;
        int index = parents.at(p.parents_begin);
        if (index < 0 || index >= (int)mc.size()) return -9999;
        return getQuarkOrigin(mc.at(index), mc, parents, include_t);
    }

    // Couldn't find the quark origin
    return -2;
}


inline Vec_mc fromRP2MC(Vec_i ind, Vec_i mcind, Vec_mc mc){
    Vec_mc result;

    for (int i : ind) {
        // Check if index is within mcind array bounds
        if (i < 0 || i >= (int)mcind.size()) continue;
        
        // Get the MC index and check if it's valid
        int mc_idx = mcind.at(i);
        if (mc_idx < 0 || mc_idx >= (int)mc.size()) continue;
        
        // Now safe to access MC particle
        auto p = mc.at(mc_idx);
        result.push_back(p);
    }
    return result;
}


inline Vec_mc getMC(Vec_i ind, Vec_mc mc){
    Vec_mc result;

    for (int i : ind) {
        
        // Get the MC index and check if it's valid
        if (i < 0 || i >= (int)mc.size()) continue;
        
        // Now safe to access MC particle
        auto p = mc.at(i);
        result.push_back(p);
    }
    return result;
}


inline Vec_rp smearPhotonEnergyResolution(Vec_rp in, Vec_i in_idx, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc, float scale = 1.0) {

    // avoid events that have the extra soft photon and screws up the MC/RECO collections
    if(reco.size() != recind.size()) return in;


    // IDEA default: 0.005=constant term (A), 0.03=stochastic term (B) 0.002=noise term (C)
    auto ecal_res_formula = new TF1("ecal_res", "TMath::Sqrt(x^2*0.005^2 + x*0.03^2 + 0.002^2)", 0, 1000);

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