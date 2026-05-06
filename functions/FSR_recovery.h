#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <algorithm>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include <utility>
#include "ROOT/RVec.hxx"
#include "TVector3.h"
#include "functions.h"
#include "utils.h"

#ifndef FCCPhysicsOptimisation_H
#define FCCPhysicsOptimisation_H

namespace FCCAnalyses {


/***********************
*** HELPER FUNCTIONS ***
************************/

// Struct to hold FSR association statistics
struct FSRStats {
    int total_associated_photons = 0;  // Total number of photons associated with leptons
    int actual_fsr_photons = 0;        // Photons that are genuine FSR (from same parent)
    int good_associations = 0;         // Number of lepton-photon pairs with same parent
};


inline Vec_i getParentPhotonIdx(edm4hep::MCParticleData parent, Vec_mc mc, Vec_i daughters) {
    Vec_i result;

    int db = parent.daughters_begin;
    int de = parent.daughters_end;

    // Check if the parent has valid daughter range
    if (db < 0 || de < 0 || db >= de || db >= (int)daughters.size()) return result;

    for (int i = db; i < de; i++) {
        if (i < 0 || i >= (int)daughters.size()) continue;
        if (daughters[i] < 0 || daughters[i] >= (int)mc.size()) continue;
        auto daughter = mc.at(daughters[i]);
        if (isPhoton(daughter.PDG)) { result.push_back(daughters[i]); }
    }
    return result;
}


inline Vec_i hasRadiated(edm4hep::MCParticleData particle, Vec_mc mc, Vec_i parents, Vec_i daughters) {

    auto parent = getParent(particle, mc, parents);
    Vec_i radiated_ph;
    
    int pb = parent.daughters_begin;
    int pe = parent.daughters_end;
    
    // Check if the parent is valid (daughters_begin and daughters_end should be reasonable)
    if (pb < 0 || pe < 0 || pb >= pe || pb >= (int)daughters.size()) return radiated_ph;
    
    for (int i = pb; i < pe; i++) {
        if (i < 0 || i >= (int)daughters.size()) continue;
        if (daughters[i] < 0 || daughters[i] >= (int)mc.size()) continue;
        auto daughter = mc.at(daughters[i]);
        if (isPhoton(daughter.PDG)) {
            radiated_ph.push_back(daughters[i]);
        }
    }
    return radiated_ph;
}



/*******************
*** FSR RECOVERY ***
********************/

inline Vec_rp recoverFSR(Vec_rp &leps, Vec_i photons, Vec_rp rps, float threshold = 0.99) {

    Vec_i usedIdx;

    for (auto &particle : leps) {

        TLorentzVector p_tlv, tmp_tlv;
        p_tlv.SetPxPyPzE(particle.momentum.x, particle.momentum.y, particle.momentum.z, particle.energy);
        tmp_tlv.SetPxPyPzE(particle.momentum.x, particle.momentum.y, particle.momentum.z, particle.energy);

        for (int ph_idx = 0; ph_idx < photons.size(); ph_idx++) {

            // Pass already used photons
            if (std::find(usedIdx.begin(), usedIdx.end(), ph_idx) != usedIdx.end()) continue;

            // Check if photon index is within bounds
            if (photons[ph_idx] < 0 || photons[ph_idx] >= (int)rps.size()) continue;

            rp photon = rps.at(photons[ph_idx]);
            TLorentzVector ph_tlv;
            ph_tlv.SetPxPyPzE(photon.momentum.x, photon.momentum.y, photon.momentum.z, photon.energy);

            TVector3 v1 = tmp_tlv.Vect();
            TVector3 v2 = ph_tlv.Vect();

            float cosTheta = v1.Dot(v2) / ( v1.Mag()* v2.Mag());

            if (cosTheta >= threshold) {
                tmp_tlv += ph_tlv;
                usedIdx.push_back(ph_idx);
            }
        }

        if (p_tlv != tmp_tlv) {
            particle.momentum.x = tmp_tlv.Px();
            particle.momentum.y = tmp_tlv.Py();
            particle.momentum.z = tmp_tlv.Pz();
            particle.energy = tmp_tlv.Energy();
        }
    }
    return leps;
}


inline Vec_f lepGaPair(Vec_f leps, Vec_f photons, bool lep_value = true) {
    Vec_f result;

    for (auto &lep : leps) {
        for (auto &photon : photons) {
            if (lep_value) result.push_back(lep);
            else result.push_back(photon);
        }
    }
    return result;
}


inline Vec_f getCosTheta(Vec_tlv leps, Vec_tlv photons) {
    Vec_f result;

    for (auto &lep : leps) {
        TVector3 v1 = lep.Vect();
        for (auto &photon : photons) {
            TVector3 v2 = photon.Vect();
            result.push_back(v1.Dot(v2) / ( v1.Mag() * v2.Mag() ) );
        }
    }
    return result;
}


inline Vec_f getAcolinearity(Vec_tlv leps, Vec_tlv photons){
    Vec_f result;

    for (auto lep : leps) {
        TVector3 v1 = lep.Vect();
        for (auto photon : photons) {
            TVector3 v2 = photon.Vect();
            float aco = std::acos( v1.Dot(v2) / (v1.Mag()*v2.Mag()) *(-1) );
            result.push_back(aco);
        }
    }
    return result;
}


inline Vec_f getAcopolarity(Vec_tlv leps, Vec_tlv photons){
    Vec_f result;

    for (auto lep : leps) {
        for (auto photon : photons) {
            float aco = std::abs( lep.Theta() - photon.Theta() );
            result.push_back(aco);
        }
    }
    return result;
}


inline Vec_f getAcoplanarity(Vec_tlv leps, Vec_tlv photons){
    Vec_f result;

    for (auto lep : leps) {
        for (auto photon : photons) {
            float aco = std::abs( lep.Phi() - photon.Phi() );
            if (aco > M_PI) aco = 2 * M_PI - aco;
            aco = M_PI - aco;
            result.push_back(aco);
        }
    }
    return result;
}

inline Vec_f getDeltaR(Vec_tlv leps, Vec_tlv photons){
    Vec_f result;

    for (auto lep : leps) {
        for (auto photon : photons) {
            result.push_back(lep.DeltaR(photon));
        }
    }
    return result;
}


inline Vec_i fromSameParent(Vec_i lep_parent_id, Vec_i ph_parent_id){
    Vec_i result;

    for (int lep_id : lep_parent_id) {
        for (int ph_id : ph_parent_id) {
            result.push_back(ph_id == lep_id ? 1 : 0);
        }
    }
    return result;
}

inline Vec_i nRadiated(Vec_i lep_parent_id, Vec_i ph_parent_id){
    Vec_i result;

    for (int lep_id : lep_parent_id) {
        int n = 0;
        for (int ph_id : ph_parent_id) {
            if (ph_id == lep_id) n += 1;
        }
        result.push_back(n);
    }
    return result;
}


inline Vec_i getMCPhotons(Vec_mc mc, Vec_i daughters){
    Vec_i result;

    for (int i = 0; i < mc.size(); i++) {
        auto p = mc.at(i);
        if (!isPhoton(p.PDG)) continue;
        if (isStable(p, daughters)) result.push_back(i);
    }
    return result;
}


/*****************************
*** FSR RECOVERY VALIDATION ***
*****************************/

inline std::vector<std::pair<int, Vec_i>> recoverFSR_idx(Vec_rp leps, Vec_i photons, Vec_rp rps, float threshold = 0.99) {
    std::vector<std::pair<int, Vec_i>> pairs;
    Vec_i usedIdx;

    TLorentzVector p_tlv, ph_tlv;
    TVector3 v1, v2;

    float cosTheta;
    for (auto particle : leps) {
        
        p_tlv.SetPxPyPzE(particle.momentum.x, particle.momentum.y, particle.momentum.z, particle.energy);
        v1 = p_tlv.Vect();

        Vec_i ph_fsr;

        for (int ph_idx = 0; ph_idx < photons.size(); ph_idx++) {
            
            // pass already used photons
            if (std::find(usedIdx.begin(), usedIdx.end(), ph_idx) != usedIdx.end()) continue;

            // Check if photon index is within bounds
            if (photons[ph_idx] < 0 || photons[ph_idx] >= (int)rps.size()) continue;

            // get the photon 3-momentum
            rp photon = rps.at(photons[ph_idx]);
            ph_tlv.SetPxPyPzE(photon.momentum.x, photon.momentum.y, photon.momentum.z, photon.energy);
            v2 = ph_tlv.Vect();

            cosTheta = v1.Dot(v2) / ( v1.Mag() * v2.Mag() );

            if (cosTheta >= threshold) {
                // merge the lepton and the photon
                p_tlv += ph_tlv;
                usedIdx.push_back(ph_idx);

                // get the indice of the merged photons for MC verification
                ph_fsr.push_back(photons[ph_idx]);
            }
        }
        pairs.push_back({particle.tracks_begin, ph_fsr});
    }

    return pairs;
}

// Validate FSR associations by checking if leptons and photons come from the same parent
// Returns FSRStats struct with:
//   - total_associated_photons: total number of photons paired with leptons via collinearity
//   - actual_fsr_photons: total photons that were actually radiated by the lepton
//   - good_associations: number of associated photons that match the true radiated photons
inline ROOT::RVec<Vec_i> validateFSRAssociations(
    std::vector<std::pair<int, Vec_i>> pairs,
    Vec_rp rps,
    Vec_mc mc,
    Vec_i parents,
    Vec_i daughters,
    Vec_i recind,
    Vec_i mcind) {

    ROOT::RVec<Vec_i> result;

    for (const auto& pair : pairs) {

        Vec_i stats;
        stats.reserve(3);

        int track_idx = pair.first;
        int mc_lepton_idx = ReconstructedParticle2MC::getTrack2MC_index(track_idx, recind, mcind, rps);
        
        // Invalid lepton MC index
        if (mc_lepton_idx < 0 || mc_lepton_idx >= (int)mc.size()) continue;

        auto lepton = mc.at(mc_lepton_idx);
        Vec_i radiated_ph = hasRadiated(lepton, mc, parents, daughters);

        // Process each photon associated with this lepton
        const Vec_i& photon_indices = pair.second;
        stats.push_back(photon_indices.size());
        stats.push_back(radiated_ph.size());

        // Convert reco photon indices to MC indices for comparison
        Vec_i mc_associated_photons;
        for (int ph_idx : photon_indices) {
            if (ph_idx < 0 || ph_idx >= (int)mcind.size()) continue;
            int mc_ph_idx = mcind.at(ph_idx);
            if (mc_ph_idx >= 0 && mc_ph_idx < (int)mc.size()) {
                mc_associated_photons.push_back(mc_ph_idx);
            }
        }

        // Compare: count how many associated photons match the true radiated photons
        int good_associations = 0;
        for (int assoc_ph : mc_associated_photons) {
            if (std::find(radiated_ph.begin(), radiated_ph.end(), assoc_ph) != radiated_ph.end()) {
                good_associations += 1;
            }
        }
        stats.push_back(good_associations);
        result.push_back(stats);
    }

    return result;
}

}

#endif