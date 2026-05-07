#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <algorithm>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include <stdexcept>
#include "ROOT/RVec.hxx"
#include "TVector3.h"
#include "functions.h"
#include "utils.h"

#ifndef FCCPhysicsFSRRecovery_H
#define FCCPhysicsFSRRecovery_H

namespace FCCAnalyses {

/*******************
*** FSR RECOVERY ***
********************/

inline Vec_rp recoverFSR(Vec_rp &leps, Vec_i photons, Vec_rp rps, Vec_f iso, int method = 0, float threshold = 0.2, float iso_thr = 1e10) {

    Vec_i usedIdx;

    for (size_t i = 0; i < leps.size(); i++) {

        // only consider isolated leptons
        if (iso.at(i) > iso_thr) continue;
        auto particle = leps.at(i);

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

            // use dR as merging criteria
            if (method == 0) {
                float dr = tmp_tlv.DeltaR(ph_tlv);

                if (dr <= threshold) {
                    tmp_tlv += ph_tlv;
                    usedIdx.push_back(ph_idx);
                }
            // use acolinearity as merging criteria
            } else if (method == 1) {

                TVector3 v1 = tmp_tlv.Vect(), v2 = ph_tlv.Vect();
                float acol = std::acos(v1.Dot(v2) / (v2.Mag()*v2.Mag()) * (-1));

                if (acol >= threshold) {
                    tmp_tlv += ph_tlv;
                    usedIdx.push_back(ph_idx);
                }
            // use acoplanarity as merging criteria
            } else if (method == 2) {
                
                float acop = abs(tmp_tlv.Phi() - ph_tlv.Phi());
                if (acop > M_PI) acop = 2 * M_PI - acop;
                acop = M_PI - acop;

                if (acop <= threshold) {
                    tmp_tlv += ph_tlv;
                    usedIdx.push_back(ph_idx);
                }
            // use acopolarity as merging criteria
            } else if (method == 3) {
                
                float acop = abs(tmp_tlv.Theta() - ph_tlv.Theta());

                if (acop <= threshold) {
                    tmp_tlv += ph_tlv;
                    usedIdx.push_back(ph_idx);
                }
            // use cosTheta between two vectors as merging criteria
            } else if (method == 4) {

                TVector3 v1 = tmp_tlv.Vect(), v2 = ph_tlv.Vect();
                float cosTheta = v1.Dot(v2) / (v1.Mag() * v2.Mag());

                if (cosTheta >= threshold) {
                    tmp_tlv += ph_tlv;
                    usedIdx.push_back(ph_idx);
                }
            } else {
                throw std::invalid_argument("Invalid FSR recovery method. Method must be 0-4.");
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

}

#endif