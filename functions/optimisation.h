#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include <stdexcept>
#include "ROOT/RVec.hxx"
#include "TString.h"
#include "functions.h"
#include "utils.h"

#ifndef FCCPhysicsOptimisation_H
#define FCCPhysicsOptimisation_H

namespace FCCAnalyses {


/***********************
*** HELPER FUNCTIONS ***
************************/

// Build a ReconstructedParticleData from an MC particle
// Converts MC 4-momentum to rp structure with specified charge
inline rp buildRecoParticle(const edm4hep::MCParticleData& mc) {
    rp particle;
    TLorentzVector tlv;
    tlv.SetXYZM(mc.momentum.x, mc.momentum.y, mc.momentum.z, mc.mass);
    particle.momentum.x = tlv.Px();
    particle.momentum.y = tlv.Py();
    particle.momentum.z = tlv.Pz();
    particle.energy = tlv.E();
    particle.mass = tlv.M();
    particle.charge = mc.charge;
    return particle;
}

inline bool hasHiggsDaughter(edm4hep::MCParticleData particle, Vec_mc mc, Vec_i daughters) {
    int db = particle.daughters_begin;
    int de = particle.daughters_end;

    if (db == de) return false;
    for (int i; i < de; i++) {
        auto daughter = mc[daughters[i]];
        if (isHiggs(daughter.PDG)) return true;
    }
    return false;
}

// Struct to hold Lorentz vectors and PDG IDs of quarks/gluons
struct QuarkLVectors {
    Vec_tlv vectors;
    Vec_i pdgIds;
    Vec_i indices;  // MC indices for reference
};

// Struct to hold MC truth matching info for a jet
struct JetMCInfo {
    int   idx = -1;       // Index of the matched MC quark
    int   pdg = 0;        // PDG ID of matched quark
    float dr  = 9e99;     // ΔR used for matching
};

// Make Lorentz vectors from MC quarks (and optionally gluons)
// Returns struct with vectors, PDG IDs, and MC indices
inline QuarkLVectors makeQuarkLorentzVectors(Vec_mc mc, bool findGluons = false) {
    QuarkLVectors result;
    for(size_t i = 0; i < mc.size(); ++i) {
        int pdgid = abs(mc[i].PDG);
        if(!isQuark(pdgid) && !findGluons) continue;  // only quarks
        if(!isQuark(pdgid) && !isGluon(pdgid) && findGluons) continue;  // only quarks and gluons
        
        TLorentzVector tlv;
        tlv.SetXYZM(mc[i].momentum.x, mc[i].momentum.y, mc[i].momentum.z, mc[i].mass);
        result.vectors.push_back(tlv);
        result.pdgIds.push_back(mc[i].PDG);
        result.indices.push_back(i);
    }
    return result;
}



/***********************
*** LEPTONIC CHANNEL ***
************************/

// Struct to hold Z pair information for chi2_recoil_frac optimization
// Contains Z system, leptons, and pre-computed mass and recoil values
struct ZPairInfo {
    rp z_system;                // Z 4-momentum
    rp lepton1, lepton2;        // Lepton 4-momenta
    
    float z_mass;               // Z system mass
    float recoil;               // Recoil mass

    int mc_idx1 = -1, mc_idx2 = -1;    // Indices of MC leptons 
    int leg_idx1, leg_idx2;            // Indices of leptons in input collection
};

struct TrueZInfo {
    rp z_system;
    rp lepton1, lepton2;
    TLorentzVector l1, l2;
    Vec_i idx;
};

// Build the Z resonance by computing all valid lepton pairs and returning
// mass and recoil values for each pair. This allows optimization of chi2_recoil_frac
// without needing to recompute pairs for each chi2_recoil_frac value.
struct leptonicZBuilder{
    float ecm;
    bool m_use_MC;
    leptonicZBuilder(float arg_ecm, bool arg_use_MC);
    ROOT::RVec<ZPairInfo> operator()(Vec_rp legs, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc);
};

inline leptonicZBuilder::leptonicZBuilder(float arg_ecm, bool arg_use_MC) {ecm = arg_ecm; m_use_MC = arg_use_MC;}
inline ROOT::RVec<ZPairInfo> leptonicZBuilder::operator()(Vec_rp legs, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc) {
    ROOT::RVec<ZPairInfo> all_pairs;
    int n = legs.size();

    if(n < 2) return all_pairs;  // Need at least 2 leptons for a pair

    // Generate all lepton pair permutations
    Vec_b w(n);
    std::fill(w.end() - 2, w.end(), true); // Select all 2-combinations
    do {
        // Get indices of selected pair
        int idx1 = -1, idx2 = -1;
        for(int i = 0; i < n; ++i) {
            if(w[i]) {
                if(idx1 < 0) idx1 = i;
                else { idx2 = i; break; }
            }
        }

        // Get leptons for this pair
        rp leg1 = legs[idx1];
        rp leg2 = legs[idx2];

        // Skip if charge is not zero (must be neutral pair)
        if(leg1.charge + leg2.charge != 0) continue;

        // Build Z 4-momentum
        TLorentzVector lv1, lv2, l1_mc, l2_mc;

        // get MC indices for evaluation
        int track_idx1 = leg1.tracks_begin;
        int track_idx2 = leg2.tracks_begin;
        int mc_idx1 = ReconstructedParticle2MC::getTrack2MC_index(track_idx1, recind, mcind, reco);
        int mc_idx2 = ReconstructedParticle2MC::getTrack2MC_index(track_idx2, recind, mcind, reco);

        
        // Use reconstructed kinematics
        lv1.SetXYZM(leg1.momentum.x, leg1.momentum.y, leg1.momentum.z, leg1.mass);
        lv2.SetXYZM(leg2.momentum.x, leg2.momentum.y, leg2.momentum.z, leg2.mass);
        
        TLorentzVector z_lv = lv1 + lv2;
        float z_mass = z_lv.M();
        
        // Calculate recoil mass
        TLorentzVector recoil(0, 0, 0, ecm);
        recoil -= z_lv;
        float recoil_mass = recoil.M();
        
        // Store pair info
        ZPairInfo pair_info;
        pair_info.z_system.momentum.x = z_lv.Px();
        pair_info.z_system.momentum.y = z_lv.Py();
        pair_info.z_system.momentum.z = z_lv.Pz();
        pair_info.z_system.mass   = z_mass;
        pair_info.z_system.charge = 0;
        pair_info.z_system.energy = z_lv.E();
        
        if (lv1.P() > lv2.P()) {
            pair_info.lepton1 = leg1;
            pair_info.lepton2 = leg2;

            pair_info.leg_idx1 = idx1;
            pair_info.leg_idx2 = idx2;

            if (mc_idx1 >= 0 || mc_idx1 < mc.size()) pair_info.mc_idx1 = mc_idx1;
            if (mc_idx2 >= 0 || mc_idx2 < mc.size()) pair_info.mc_idx2 = mc_idx2;

        } else {
            pair_info.lepton1 = leg2;
            pair_info.lepton2 = leg1;

            pair_info.leg_idx1 = idx2;
            pair_info.leg_idx2 = idx1;

            if (mc_idx1 >= 0 || mc_idx1 < mc.size()) pair_info.mc_idx2 = mc_idx1;
            if (mc_idx2 >= 0 || mc_idx2 < mc.size()) pair_info.mc_idx1 = mc_idx2;
        }
        
        pair_info.z_mass = z_mass;
        pair_info.recoil = recoil_mass;

        all_pairs.emplace_back(pair_info);

    } while(std::next_permutation(w.begin(), w.end()));

    return all_pairs;
}


// Find the true Z boson from ZH production
// Returns a vector with 3 ReconstructedParticleData-like structures:
// [0] = Z system (momentum and mass from MC)
// [1] = first lepton from Z, [2] = second lepton from Z
// Returns dummy structures if no valid Z->ll is found
inline TrueZInfo getTrueZ(TString cat, Vec_mc mc, Vec_i parents, Vec_i daughters) {

    TrueZInfo result;
    Vec_mc cand;
    Vec_i idx;
    
    // Loop through MC particles to find Z bosons
    for (size_t i = 0; i < mc.size(); ++i) {
        const auto& particle = mc[i];
        
        if (cat == "ee") { if (!isElectron(particle.PDG)) continue; }
        else if (cat == "mumu") { if (!isMuon(particle.PDG)) continue; }
        else { throw std::invalid_argument("cat only accept 'mumu' or 'ee' values"); }

        // Only consider stable particle
        // if getLeptonOrigin = 0, the lepton comes directly comes from the initial state
        if (isStable(particle, daughters) && getLeptonOrigin(particle, mc, parents, false) == 0) {
            cand.push_back(getParent(particle, mc, parents));
            idx.push_back(i);
        }
    }

    if (cand.size() == 2 && ( cand.at(0).charge + cand.at(1).charge == 0) ) {
        edm4hep::MCParticleData cand1 = cand.at(0);
        edm4hep::MCParticleData cand2 = cand.at(1);

        rp l1 = buildRecoParticle(cand1);
        rp l2 = buildRecoParticle(cand2);

        TLorentzVector l1_lv, l2_lv, z_lv;
        l1_lv.SetXYZM(l1.momentum.x, l1.momentum.y, l1.momentum.z, l1.mass);
        l2_lv.SetXYZM(l2.momentum.x, l2.momentum.y, l2.momentum.z, l2.mass);
        z_lv = l1_lv + l2_lv;

        rp z_rp;
        z_rp.momentum.x = z_lv.Px();
        z_rp.momentum.y = z_lv.Py();
        z_rp.momentum.z = z_lv.Pz();
        z_rp.energy     = z_lv.Energy();
        z_rp.mass       = z_lv.M();
        z_rp.charge     = 0;

        result.z_system = z_rp;
        int idx1 = idx[0], idx2 = idx[1];
        if (l1_lv.P() > l2_lv.P()) {
            result.lepton1 = l1;
            result.lepton2 = l2;
            result.l1 = l1_lv;
            result.l2 = l2_lv;
            result.idx = {idx1, idx2};
        } else {
            result.lepton1 = l2;
            result.lepton2 = l1;
            result.l1 = l2_lv;
            result.l2 = l1_lv;
            result.idx = {idx2, idx1};
        }
        return result;
    }
    
    // If no valid Z->ll found, return dummy particles
    auto dummy = edm4hep::ReconstructedParticleData();
    dummy.momentum.x = 0;
    dummy.momentum.y = 0;
    dummy.momentum.z = 0;
    dummy.mass = -999;
    dummy.energy = -999;
    result.z_system = dummy;
    result.lepton1 = dummy;
    result.lepton2 = dummy;

    return result;
}



/***********************
*** HADRONIC CHANNEL ***
************************/

inline Vec_i jetTruthFlavour(Vec_rp jets_rp, Vec_rp reco, Vec_mc mc, Vec_i mcind, bool findGluons = true) {
    // jet truth finder: match the gen-level partons (eventually with gluons) with the jet constituents
    // matching by mimimizing the dr between the parton and the jet

    QuarkLVectors quarkData = makeQuarkLorentzVectors(mc, findGluons);
    Vec_tlv jets = makeLorentzVectors(jets_rp);  // Lorentz-vector of all jets

    Vec_tlv genQuarks = quarkData.vectors;
    Vec_i QuarksPDG = quarkData.pdgIds;

    Vec_i usedIdx, result;
    for (size_t iJet = 0; iJet < jets.size(); iJet++) {
        Vec_d dr;
        for (size_t iGen = 0; iGen < genQuarks.size(); iGen++) {
            if (std::find(usedIdx.begin(), usedIdx.end(), iGen) != usedIdx.end()) {
                dr.push_back(9e99); // set infinite dr, skip
                continue;
            }
            dr.push_back(jets[iJet].DeltaR(genQuarks[iGen]));
        }
        int minDrIdx = std::min_element(dr.begin(), dr.end()) - dr.begin();
        usedIdx.push_back(minDrIdx);
        if (dr[minDrIdx] == 9e99) { result.push_back(-999); }
        else { result.push_back(QuarksPDG[minDrIdx]); }
    }
    return result;
}


// Match a single jet to MC quarks using ΔR-based matching (like jetTruthFinder)
inline ROOT::VecOps::RVec<JetMCInfo> jets2MC(Vec_rp jets_rp, Vec_rp reco, Vec_mc mc, bool findGluons = true) {
    
    ROOT::VecOps::RVec<JetMCInfo> result;

    if (jets_rp.empty()) return result;  // No jets reconstructed
    
    // Build Lorentz vectors of all MC quarks
    QuarkLVectors quarkData = makeQuarkLorentzVectors(mc, findGluons);
    Vec_tlv genQuarks = quarkData.vectors;   // MC Lorentz vectors
    Vec_i Quarks_idx  = quarkData.indices;   // MC indices
    Vec_i QuarksPDG   = quarkData.pdgIds;    // PDG IDs
    
    if(genQuarks.empty()) return result;  // No quarks found
    
    // Build Lorentz vectors of all reconstructed jets
    Vec_tlv jets = makeLorentzVectors(jets_rp);
    
    // Calculate ΔR sum for each MC quark
    float min_dr_sum = 1e9;
    int best_quark_idx = -1;

    Vec_i usedIdx;
    for (size_t iJet = 0; iJet < jets.size(); iJet++) {
        Vec_d dr;
        JetMCInfo jetMC;
        for (size_t iGen = 0; iGen < genQuarks.size(); iGen++) {
            if (std::find(usedIdx.begin(), usedIdx.end(), iGen) != usedIdx.end()) {
                dr.push_back(9e99);
                continue;
            }
            dr.push_back(jets[iJet].DeltaR(genQuarks[iGen]));
        }
        int minDrIdx = std::min_element(dr.begin(), dr.end()) - dr.begin();
        usedIdx.push_back(minDrIdx);
        if (dr[minDrIdx] == 9e99) { result.push_back(jetMC); }
        else {
            jetMC.idx = minDrIdx;
            jetMC.pdg = QuarksPDG[minDrIdx];
            jetMC.dr  = dr[minDrIdx];

            result.push_back(jetMC);
        }
    }
    
    return result;
}

// make all_pairs for jets
// make best_pair for jets
// make trueZ for jets
// make best clustering algo



/*****************************
*** EXTRACTOR FUNCTIONS ***
*****************************/

// Extract z_mass from all ZPairInfo entries
inline Vec_f getAllPairsZMass(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_f result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.z_mass);
    }
    return result;
}

// Extract recoil mass from all ZPairInfo entries
inline Vec_f getAllPairsRecoil(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_f result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.recoil);
    }
    return result;
}

// Extract leg_idx1 from all ZPairInfo entries
inline Vec_i getAllPairsLegIdx1(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_i result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.leg_idx1);
    }
    return result;
}

// Extract leg_idx2 from all ZPairInfo entries
inline Vec_i getAllPairsLegIdx2(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_i result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.leg_idx2);
    }
    return result;
}

// Extract mc_idx1 from all ZPairInfo entries
inline Vec_i getAllPairsMCIdx1(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_i result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.mc_idx1);
    }
    return result;
}

// Extract mc_idx2 from all ZPairInfo entries
inline Vec_i getAllPairsMCIdx2(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_i result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.mc_idx2);
    }
    return result;
}

// Extract z_system from TrueZInfo
inline Vec_rp getTrueZRP(TrueZInfo trueZ) {
    Vec_rp result;
    result.push_back(trueZ.z_system);
    return result;
}

// Extract lepton1 from TrueZInfo
inline Vec_rp getTrueZLepton1(TrueZInfo trueZ) {
    Vec_rp result;
    result.push_back(trueZ.lepton1);
    return result;
}

// Extract lepton2 from TrueZInfo
inline Vec_rp getTrueZLepton2(TrueZInfo trueZ) {
    Vec_rp result;
    result.push_back(trueZ.lepton2);
    return result;
}

// Extract z_system from all ZPairInfo entries
inline Vec_rp getZPairsZSystem(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_rp result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.z_system);
    }
    return result;
}

// Extract lepton1 from all ZPairInfo entries
inline Vec_rp getZPairsLepton1(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_rp result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.lepton1);
    }
    return result;
}

// Extract lepton2 from all ZPairInfo entries
inline Vec_rp getZPairsLepton2(ROOT::RVec<ZPairInfo> all_pairs) {
    Vec_rp result;
    for (const auto& pair : all_pairs) {
        result.push_back(pair.lepton2);
    }
    return result;
}

// Extract mc_idx1 from TrueZInfo
inline int getTrueZMCIdx1(TrueZInfo trueZ) {
    return trueZ.idx.size() > 0 ? trueZ.idx[0] : -1;
}

// Extract mc_idx2 from TrueZInfo
inline int getTrueZMCIdx2(TrueZInfo trueZ) {
    return trueZ.idx.size() > 1 ? trueZ.idx[1] : -1;
}

}

#endif