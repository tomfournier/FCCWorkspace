#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <cstdlib>
#include <edm4hep/MCParticleData.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include "TVector3.h"
#include "functions.h"
#include "utils.h"

#ifndef FCCPhysicsFSRRecovery_H
#define FCCPhysicsFSRRecovery_H

namespace FCCAnalyses {

/*******************
*** FSR RECOVERY ***
********************/

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