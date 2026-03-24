#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <cstdlib>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
// #include <iostream>

#ifndef FCCPhysicsFunctions_H
#define FCCPhysicsFunctions_H

namespace FCCAnalyses {

/***********************
*** CUSTOM FUNCTIONS ***
***********************/

// acopolarity between two reco particles
inline float acopolarity(Vec_rp in) {
    if(in.size() < 2) return -999;

    TLorentzVector p1;
    p1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);

    TLorentzVector p2;
    p2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);

    float acol = abs(p1.Theta() - p2.Theta());
    return acol;
}

// acopolarity between two reco particles
inline float acopolarity(Vec_tlv in) {
    if(in.size() < 2) return -999;
    auto p1 = in[0];
    auto p2 = in[1];

    float acol = abs(p1.Theta() - p2.Theta());
    return acol;
}


// acolinearity between two reco particles
inline float acolinearity(Vec_rp in) {
    if(in.size() < 2) return -999;

    TLorentzVector p1;
    p1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);

    TLorentzVector p2;
    p2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);

    TVector3 v1 = p1.Vect();
    TVector3 v2 = p2.Vect();
    return std::acos(v1.Dot(v2)/(v1.Mag()*v2.Mag())*(-1.));
}
// acolinearity between two reco particles
inline float acolinearity(Vec_tlv in) {
    if(in.size() < 2) return -999;
    auto p1 = in[0];
    auto p2 = in[1];

    TVector3 v1 = p1.Vect();
    TVector3 v2 = p2.Vect();
    return std::acos(v1.Dot(v2)/(v1.Mag()*v2.Mag())*(-1.));
}


// // acoplanarity between two reco particles
inline float acoplanarity(Vec_rp in) {
    if(in.size() < 2) return -999;

    TLorentzVector p1;
    p1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);

    TLorentzVector p2;
    p2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);

    float acop = abs(p1.Phi() - p2.Phi());
    if(acop > M_PI) acop = 2 * M_PI - acop;
    acop = M_PI - acop;

    return acop;
}
// acoplanarity between two reco particles
inline float acoplanarity(Vec_tlv in) {
    if(in.size() < 2) return -999;
    auto p1 = in[0];
    auto p2 = in[1];

    float acop = abs(p1.Phi() - p2.Phi());
    if(acop > M_PI) acop = 2 * M_PI - acop;
    acop = M_PI - acop;

    return acop;
}


// calculate the cosine(theta) of the missing energy vector, based on reco particles
inline float cosThetaMissingEnergy(Vec_rp in) {
    float px = 0, py = 0, pz = 0;
    for(auto &p : in) {
        px += -p.momentum.x;
        py += -p.momentum.y;
        pz += -p.momentum.z;
    }

    return std::abs(pz) / std::sqrt(px*px + py*py + pz*pz);
}


// deltaR between two reco particles, based on eta
inline float deltaR(Vec_rp in) {
    if(in.size() != 2) return -1;
    
    TLorentzVector tlv1;
    tlv1.SetPxPyPzE(in.at(0).momentum.x, in.at(0).momentum.y, in.at(0).momentum.z, in.at(0).energy);

    TLorentzVector tlv2;
    tlv2.SetPxPyPzE(in.at(1).momentum.x, in.at(1).momentum.y, in.at(1).momentum.z, in.at(1).energy);

    return tlv1.DeltaR(tlv2); 
}


// computes longitudinal and transversal energy balance of all particles
inline Vec_f energy_imbalance(Vec_rp in) {
    float e_tot = 0;
    float e_trans = 0;
    float e_long = 0;
    for(auto &p : in) {
        float mag = std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y + p.momentum.z*p.momentum.z);
        float cost = p.momentum.z / mag;
        float sint =  std::sqrt(p.momentum.x*p.momentum.x + p.momentum.y*p.momentum.y) / mag;
        if(p.momentum.y < 0) sint *= -1.0;
        e_tot += p.energy;
        e_long += cost*p.energy;
        e_trans += sint*p.energy;
    }
    Vec_f result;
    result.push_back(e_tot);
    result.push_back(std::abs(e_trans));
    result.push_back(std::abs(e_long));
    return result;
}


// calculate cosine(theta), based on reco particle
inline Vec_f get_costheta(Vec_rp in) {
    Vec_f result;
    for (auto & p: in) {
        TLorentzVector tlv;
        tlv.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);
        result.push_back(cos(tlv.Theta()));
    }
    return result;
}

inline Vec_f get_costheta(Vec_f theta, bool absolute = false){
    Vec_f ret; 
    for(auto & t: theta) {
        if (absolute) {
            ret.push_back(abs(cos(t)));
        }
        else {
            ret.push_back(cos(t));
        }
    }
    return ret;
}


// calculate the cosine(theta) of the missing energy vector
inline float get_cosTheta_miss(Vec_rp met){

    float costheta = 0.;
    if(met.size() > 0) {
        TLorentzVector lv_met;
        lv_met.SetPxPyPzE(met[0].momentum.x, met[0].momentum.y, met[0].momentum.z, met[0].energy);
        costheta = fabs(std::cos(lv_met.Theta()));
    }
    return costheta;
}


inline int get_max_idx(Vec_f prop){
    int idx = -1;
    float max = -1.;
    for (int i = 0; i<prop.size(); i++) {
        if(prop[i]>max){
            max = prop[i];
            idx = i;
        }
    }
    return idx;
}


// compute Higgstrahlungness
inline float Higgsstrahlungness(float mll, float mrecoil) {
  float mZ = 91.2;
  float mH = 125.;
  float chi2_recoil_frac = 0.4;
  float chiZ = std::pow(mll - mZ, 2); // mass
  float chiH = std::pow(mrecoil - mH, 2); // recoil
  float chi2 = (1.0-chi2_recoil_frac)*chiZ + chi2_recoil_frac*chiH;
  return chi2;
}


// return true is event has W+W- decaying into electron or muon
// take into account W -> tau -> electron or muon
inline bool is_ww_leptonic(Vec_mc mc, Vec_i ind) {
   int l1 = 0;
   int l2 = 0;
   for(size_t i = 0; i < mc.size(); ++i) {
        auto & p = mc[i];
        if(std::abs(p.PDG) == 24) {
            int ds = p.daughters_begin;
            int de = p.daughters_end;
            for(int k=ds; k<de; k++) {
                int pdg = abs(mc[ind[k]].PDG);
                if(pdg == 24) continue;
                if(pdg == 11 or pdg == 13) {
                    if(l1 == 0) l1 = pdg;
                    else l2 = pdg;
                }
            }
        }
        else if(std::abs(p.PDG) == 15) { // tau decays
            int ds = p.daughters_begin;
            int de = p.daughters_end;
            for(int k=ds; k<de; k++) {
                int pdg = abs(mc[ind[k]].PDG);
                if(pdg == 15) continue;
                if(pdg == 11 or pdg == 13) {
                    if(l1 == 0) l1 = pdg;
                    else l2 = pdg;
                }
            }
        }
   }
   if(l1 == l2 && (l1==13 || l1 == 11)) {
       return true;
   }
   return false;
}


// return true is event is H -> ZZ* -> 4nu
inline bool is_hzz_invisible(Vec_mc mc, Vec_i ind) {
   int d1 = 0;
   int d2 = 0;
   for(size_t i = 0; i < mc.size(); ++i) {
        auto & p = mc[i];
        if(std::abs(p.PDG) != 23) continue;

        int ds = p.daughters_begin;
        int de = p.daughters_end;
        int idx_ds = ind[ds];
        int idx_de = ind[de-1];
        int pdg_d1 = abs(mc[idx_ds].PDG);
        int pdg_d2 = abs(mc[idx_de].PDG);

        if(std::abs(pdg_d1) == 23 or std::abs(pdg_d2) == 23) continue;
        if(d1 == 0) d1 += pdg_d1 + pdg_d2;
        else d2 += pdg_d1 + pdg_d2;
   }
   if((d1==24 || d1==28 || d1==32) && (d2==24 || d2==28 || d2==32)) {
       return true;
   }
   return false;
}


// make Lorentz vectors for a given RECO particle collection
inline Vec_tlv makeLorentzVectors(Vec_rp in) {
    Vec_tlv result;
    for(auto & p: in) {
        TLorentzVector tlv;
        tlv.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);
        result.push_back(tlv);
    }
    return result;
}


// make Lorentzvectors from pseudojets
inline Vec_tlv makeLorentzVectors(Vec_f jets_px, Vec_f jets_py, Vec_f jets_pz, Vec_f jets_e) {
    Vec_tlv result;
    for(int i=0; i<jets_px.size(); i++) {
        TLorentzVector tlv;
        tlv.SetPxPyPzE(jets_px[i], jets_py[i], jets_pz[i], jets_e[i]);
        result.push_back(tlv);
    }
    return result;
}


// returns missing energy vector, based on reco particles
inline Vec_rp missingEnergy(float ecm, Vec_rp in, float p_cutoff = 0.0) {
    float px = 0, py = 0, pz = 0, e = 0;
    for(auto &p : in) {
        if (std::sqrt(p.momentum.x * p.momentum.x + p.momentum.y*p.momentum.y) < p_cutoff) continue;
        px += -p.momentum.x;
        py += -p.momentum.y;
        pz += -p.momentum.z;
        e += p.energy;
    }

    Vec_rp ret;
    rp res;
    res.momentum.x = px;
    res.momentum.y = py;
    res.momentum.z = pz;
    res.energy = ecm-e;

    ret.emplace_back(res);
    return ret;
}


// calculate the missing mass, given a ECM value
inline float missingMass(float ecm, Vec_rp in, float p_cutoff = 0.0) {
    float px = 0, py = 0, pz = 0, e = 0;
    for(auto &p : in) {
        if (std::sqrt(p.momentum.x * p.momentum.x + p.momentum.y*p.momentum.y) < p_cutoff) continue;
        px += p.momentum.x;
        py += p.momentum.y;
        pz += p.momentum.z;
        e  += p.energy;
    }
    if(ecm < e) return -99.;

    float ptot2 = std::pow(px, 2) + std::pow(py, 2) + std::pow(pz, 2);
    float de2 = std::pow(ecm - e, 2);
    if (de2 < ptot2) return -999.;
    float Mmiss = std::sqrt(de2 - ptot2);
    return Mmiss;
}


// calculate the visible energy of the event
inline float visibleEnergy(Vec_rp in, float p_cutoff = 0.0) {
    float e = 0;
    for(auto &p : in) {
        if (std::sqrt(p.momentum.x * p.momentum.x + p.momentum.y*p.momentum.y) < p_cutoff) continue;
        e += p.energy;
    }
    return e;
}


// calculate the visible mass of the event
inline float visibleMass(Vec_rp in, float p_cutoff = 0.0) {
    float px = 0, py = 0, pz = 0, e = 0;
    for(auto &p : in) {
        if (std::sqrt(p.momentum.x * p.momentum.x + p.momentum.y*p.momentum.y) < p_cutoff) continue;
        px += p.momentum.x;
        py += p.momentum.y;
        pz += p.momentum.z;
        e += p.energy;
    }

    float ptot2 = std::pow(px, 2) + std::pow(py, 2) + std::pow(pz, 2);
    float de2 = std::pow(e, 2);
    if (de2 < ptot2) return -999.;
    float Mvis = std::sqrt(de2 - ptot2);
    return Mvis;
}


// compute the cone isolation for reco particles
struct coneIsolation {

    coneIsolation(float arg_dr_min, float arg_dr_max);
    double deltaR(double eta1, double phi1, double eta2, double phi2) { 
        return TMath::Sqrt(TMath::Power(eta1-eta2, 2) + (TMath::Power(phi1-phi2, 2))); 
    };

    float dr_min = 0;
    float dr_max = 0.4;
    Vec_f operator() (Vec_rp in, Vec_rp rps) ;
};

inline coneIsolation::coneIsolation(float arg_dr_min, float arg_dr_max) : dr_min(arg_dr_min), dr_max( arg_dr_max ) { };
inline Vec_f coneIsolation::coneIsolation::operator() (Vec_rp in, Vec_rp rps) {

    Vec_f result;
    result.reserve(in.size());

    // compute the isolation (see https://github.com/delphes/delphes/blob/master/modules/Isolation.cc#L154) 
    for(size_t i = 0; i < in.size(); ++i) {
        ROOT::Math::PxPyPzEVector tlv_target;
        tlv_target.SetPxPyPzE(in.at(i).momentum.x, in.at(i).momentum.y, in.at(i).momentum.z, in.at(i).energy);
        double sumNeutral = 0.0;
        double sumCharged = 0.0;

        for(size_t i = 0; i < rps.size(); ++i) {
            ROOT::Math::PxPyPzEVector tlv;
            tlv.SetPxPyPzE(rps.at(i).momentum.x, rps.at(i).momentum.y, rps.at(i).momentum.z, rps.at(i).energy);
            double dr = coneIsolation::deltaR(tlv_target.Eta(), tlv_target.Phi(), tlv.Eta(), tlv.Phi());
            if(dr < dr_min || dr > dr_max) continue;
            if(rps.at(i).charge == 0) sumNeutral += tlv.P();
            else sumCharged += tlv.P();
        }

        double sum = sumCharged + sumNeutral;
        double ratio= sum / tlv_target.P();
        result.emplace_back(ratio);
    }
    return result;
}



/*******************
*** CUSTOM CLASS ***
*******************/

// calculate the number of foward leptons
struct polarAngleCategorization {
polarAngleCategorization(float arg_thetaMin, float arg_thetaMax);
float thetaMin = 0;
float thetaMax = 5;
int operator() (Vec_rp in);
};

// calculate the number of foward leptons
inline FCCAnalyses::polarAngleCategorization::polarAngleCategorization(float arg_thetaMin, float arg_thetaMax) : thetaMin(arg_thetaMin), thetaMax(arg_thetaMax) {};
inline int polarAngleCategorization::operator() (Vec_rp in) {
    
    int nFwd = 0; // number of forward leptons
    for (size_t i = 0; i < in.size(); ++i) {
        
        auto & p = in[i];
        TLorentzVector lv;
        lv.SetXYZM(p.momentum.x, p.momentum.y, p.momentum.z, p.mass);
        if(lv.Theta() < thetaMin || lv.Theta() > thetaMax) nFwd += 1;
    }
    return nFwd;
}


// build the Z resonance based on the available leptons. Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
// technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, index and 2 the leptons of the pair
struct resonanceBuilder_mass_recoil {
    float m_resonance_mass;
    float m_recoil_mass;
    float chi2_recoil_frac;
    float ecm;
    bool m_use_MC_Kinematics;
    resonanceBuilder_mass_recoil(float arg_resonance_mass, float arg_recoil_mass, float arg_chi2_recoil_frac, float arg_ecm, bool arg_use_MC_Kinematics);
    Vec_rp operator()(Vec_rp legs, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc, Vec_i parents, Vec_i daugthers) ;
};

inline resonanceBuilder_mass_recoil::resonanceBuilder_mass_recoil(float arg_resonance_mass, float arg_recoil_mass, float arg_chi2_recoil_frac, float arg_ecm, bool arg_use_MC_Kinematics) 
           {m_resonance_mass = arg_resonance_mass, m_recoil_mass = arg_recoil_mass, chi2_recoil_frac = arg_chi2_recoil_frac, ecm = arg_ecm, m_use_MC_Kinematics = arg_use_MC_Kinematics;}

inline Vec_rp resonanceBuilder_mass_recoil::resonanceBuilder_mass_recoil::operator()(Vec_rp legs, Vec_i recind, Vec_i mcind, Vec_rp reco, Vec_mc mc, Vec_i parents, Vec_i daugthers) {
    Vec_rp result;
    result.reserve(3);
    std::vector<std::vector<int>> pairs; // for each permutation, add the indices of the particles
    int n = legs.size();

    if(n > 1) { 
        ROOT::VecOps::RVec<bool> w(n);
        std::fill(w.end() - 2, w.end(), true); // helper variable for permutations
        do {
            std::vector<int> pair;
            rp reso;
            reso.charge = 0;
            TLorentzVector reso_lv; 
            for(int i = 0; i < n; ++i) {
                if(w[i]) {
                    pair.push_back(i);
                    reso.charge += legs[i].charge;
                    TLorentzVector leg_lv;

                    if(m_use_MC_Kinematics) { // MC kinematics
                        int track_index = legs[i].tracks_begin;   // index in the Track array
                        int mc_index = ReconstructedParticle2MC::getTrack2MC_index(track_index, recind, mcind, reco);
                        if (mc_index >= 0 && mc_index < mc.size()) {
                            leg_lv.SetXYZM(mc.at(mc_index).momentum.x, mc.at(mc_index).momentum.y, mc.at(mc_index).momentum.z, mc.at(mc_index).mass);
                        }
                    }
                    else { // reco kinematics
                         leg_lv.SetXYZM(legs[i].momentum.x, legs[i].momentum.y, legs[i].momentum.z, legs[i].mass);
                    }
                    reso_lv += leg_lv;
                }
            }

            if(reso.charge != 0) continue; // neglect non-zero charge pairs
            reso.momentum.x = reso_lv.Px();
            reso.momentum.y = reso_lv.Py();
            reso.momentum.z = reso_lv.Pz();
            reso.mass = reso_lv.M();
            result.emplace_back(reso);
            pairs.push_back(pair);

        } while(std::next_permutation(w.begin(), w.end()));
    }
    else {
        auto dummy = edm4hep::ReconstructedParticleData();
        dummy.momentum.x = 0;
        dummy.momentum.y = 0;
        dummy.momentum.z = 0;
        dummy.mass = 0;
        result.emplace_back(dummy);
        result.emplace_back(dummy);
        result.emplace_back(dummy);
        return result;
    }

    if(result.size() > 1) {

        Vec_rp bestReso;
        int idx_min = -1;
        float d_min = 9e9;
        for (int i = 0; i < result.size(); ++i) {
            
            // calculate recoil
            auto recoil_p4 = TLorentzVector(0, 0, 0, ecm);
            TLorentzVector tv1;
            tv1.SetXYZM(result.at(i).momentum.x, result.at(i).momentum.y, result.at(i).momentum.z, result.at(i).mass);
            recoil_p4 -= tv1;

            auto recoil_fcc = edm4hep::ReconstructedParticleData();
            recoil_fcc.momentum.x = recoil_p4.Px();
            recoil_fcc.momentum.y = recoil_p4.Py();
            recoil_fcc.momentum.z = recoil_p4.Pz();
            recoil_fcc.mass = recoil_p4.M();

            TLorentzVector tg;
            tg.SetXYZM(result.at(i).momentum.x, result.at(i).momentum.y, result.at(i).momentum.z, result.at(i).mass);

            // float boost = tg.P();
            float mass = std::pow(result.at(i).mass - m_resonance_mass, 2); // mass
            float rec = std::pow(recoil_fcc.mass - m_recoil_mass, 2); // recoil
            float d = (1.0-chi2_recoil_frac)*mass + chi2_recoil_frac*rec;

            if(d < d_min) {
                d_min = d;
                idx_min = i;
            }

        }
        if(idx_min > -1) { 
            bestReso.push_back(result.at(idx_min));
            auto & l1 = legs[pairs[idx_min][0]];
            auto & l2 = legs[pairs[idx_min][1]];
            bestReso.emplace_back(l1);
            bestReso.emplace_back(l2);
        }
        else {
            auto dummy = edm4hep::ReconstructedParticleData();
            dummy.momentum.x = 0;
            dummy.momentum.y = 0;
            dummy.momentum.z = 0;
            dummy.mass = 0;
            result.emplace_back(dummy);
            result.emplace_back(dummy);
            result.emplace_back(dummy);
            return result;
        }
        return bestReso;
    }
    else {
        auto & l1 = legs[0];
        auto & l2 = legs[1];
        result.emplace_back(l1);
        result.emplace_back(l2);
        return result;
    }
}

// filter reconstructed particles based on a property with a defined upper limit (isocut)
struct sel_isol {
sel_isol(float arg_isocut);
float m_isocut = 9999.;
Vec_rp  operator() (Vec_rp particles, Vec_f var );
};

inline sel_isol::sel_isol( float arg_isocut ) : m_isocut (arg_isocut) {};
inline Vec_rp FCCAnalyses::sel_isol::operator() (  Vec_rp particles, Vec_f var ) { 
  
  Vec_rp result;
  for (size_t i=0; i < particles.size(); ++i) {
    auto & p = particles[i];
    if ( var[i] < m_isocut) result.emplace_back(p);
  }
  return result;
}


// filter reconstructed particles (in) based a property (prop) within a defined range (m_min, m_max)
struct sel_range {
    sel_range(float arg_min, float arg_max, bool arg_abs = false);
    float m_min = 0.;
    float m_max = 1.;
    bool m_abs = false;
    Vec_rp operator() (Vec_rp in, Vec_f prop);
};

inline sel_range::sel_range(float arg_min, float arg_max, bool arg_abs) : m_min(arg_min), m_max(arg_max), m_abs(arg_abs) {};
inline Vec_rp sel_range::operator() (Vec_rp in, Vec_f prop) {
    Vec_rp result;
    for (size_t i = 0; i < in.size(); ++i) {
        auto & p = in[i];
        float val = (m_abs) ? abs(prop[i]) : prop[i];
        if(val > m_min && val < m_max) result.push_back(p);
    }
    return result;
}


// filter reconstructed particles (in) based a property (prop) within a defined range (m_min, m_max)
struct sel_range_idx {
    sel_range_idx(float arg_min, float arg_max, bool arg_abs = false, Vec_i arg_idx = {});
    float m_min = 0.;
    float m_max = 1.;
    bool m_abs = false;
    Vec_i m_idx = {};
    Vec_rp operator() (Vec_rp in, Vec_f prop);
};

inline sel_range_idx::sel_range_idx(float arg_min, float arg_max, bool arg_abs, Vec_i arg_idx) : m_min(arg_min), m_max(arg_max), m_abs(arg_abs), m_idx(arg_idx) {};
inline Vec_rp sel_range_idx::operator() (Vec_rp in, Vec_f prop) {
    Vec_rp result;
    result.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        auto & p = in[i];
        if(std::find(m_idx.begin(), m_idx.end(), i)!=m_idx.end()){
            float val = (m_abs) ? abs(prop[i]) : prop[i];
            if(val > m_min && val < m_max) result.emplace_back(p);
        }
        else {
            result.emplace_back(p);
        }
    }
    return result;
}


/*************************************************************
*** FUNCTION TO FIND TRUE Z->ll PAIRING FROM MC PARTICLES ***
*************************************************************/

// Find the true Z boson that decays to leptons and does NOT come from H->ZZ* decay
// Returns a vector with 3 ReconstructedParticleData-like structures:
// [0] = Z system (momentum and mass from MC)
// [1] = first lepton from Z
// [2] = second lepton from Z
// Returns dummy structures if no valid Z->ll is found
inline Vec_rp getTrueZ_from_H_decay(Vec_mc mc, Vec_i parents, Vec_i daughters) {
    Vec_rp result;
    result.reserve(3);
    
    // Helper lambda to check if a particle is a lepton (e, mu, tau)
    auto isLepton = [](int pdg) {
        int abs_pdg = std::abs(pdg);
        return (abs_pdg == 11 || abs_pdg == 13 || abs_pdg == 15); // e, mu, tau
    };
    
    // Helper lambda to get parent PDG
    auto getParentPDG = [&mc, &parents](int particle_idx) -> int {
        if (particle_idx >= mc.size()) return 0;
        const auto& p = mc[particle_idx];
        if (p.parents_begin >= 0 && p.parents_begin < parents.size()) {
            int parent_idx = parents[p.parents_begin];
            if (parent_idx >= 0 && parent_idx < mc.size()) {
                return mc[parent_idx].PDG;
            }
        }
        return 0;
    };
    
    // Loop through MC particles to find Z bosons
    for (size_t i = 0; i < mc.size(); ++i) {
        const auto& particle = mc[i];
        
        // Check if this is a Z boson (PDG = 23)
        if (std::abs(particle.PDG) != 23) continue;
        
        // Check that parent is NOT a Higgs (PDG = 25)
        int parent_pdg = getParentPDG(i);
        if (std::abs(parent_pdg) == 25) continue; // Skip Z from H->ZZ*
        
        // Get daughters of the Z
        int ds = particle.daughters_begin;
        int de = particle.daughters_end;
        
        if (ds < 0 || de <= ds) continue; // No valid daughters
        if (de - ds != 2) continue; // Z should have exactly 2 daughters (or could be tau->l chain)
        
        // Get daughter indices
        int d1_idx = daughters[ds];
        int d2_idx = daughters[de - 1];
        
        if (d1_idx < 0 || d1_idx >= mc.size() || d2_idx < 0 || d2_idx >= mc.size()) continue;
        
        const auto& d1 = mc[d1_idx];
        const auto& d2 = mc[d2_idx];
        
        // Check if daughters are leptons (or tau that decays to lepton)
        // For simplicity, check direct lepton daughters
        if (isLepton(d1.PDG) && isLepton(d2.PDG)) {
            // Check that they have opposite charge (for proper Z decay)
            // Neutrinos have charge 0, so we need opposite flavor leptons
            int abs_pdg1 = std::abs(d1.PDG);
            int abs_pdg2 = std::abs(d2.PDG);
            
            // Valid Z->ll: both leptons of same flavor
            if (abs_pdg1 == abs_pdg2) {
                // Build the result
                // Create Z particle (index 0)
                rp z_particle;
                TLorentzVector z_tlv;
                z_tlv.SetXYZM(particle.momentum.x, particle.momentum.y, particle.momentum.z, particle.mass);
                z_particle.momentum.x = z_tlv.Px();
                z_particle.momentum.y = z_tlv.Py();
                z_particle.momentum.z = z_tlv.Pz();
                z_particle.energy = z_tlv.E();
                z_particle.mass = z_tlv.M();
                z_particle.charge = 0;
                result.push_back(z_particle);
                
                // Create lepton 1 (index 1)
                rp lepton1;
                TLorentzVector l1_tlv;
                l1_tlv.SetXYZM(d1.momentum.x, d1.momentum.y, d1.momentum.z, d1.mass);
                lepton1.momentum.x = l1_tlv.Px();
                lepton1.momentum.y = l1_tlv.Py();
                lepton1.momentum.z = l1_tlv.Pz();
                lepton1.energy = l1_tlv.E();
                lepton1.mass = l1_tlv.M();
                lepton1.charge = (d1.PDG > 0) ? -1 : 1; // e- has charge -1, e+ has +1
                result.push_back(lepton1);
                
                // Create lepton 2 (index 2)
                rp lepton2;
                TLorentzVector l2_tlv;
                l2_tlv.SetXYZM(d2.momentum.x, d2.momentum.y, d2.momentum.z, d2.mass);
                lepton2.momentum.x = l2_tlv.Px();
                lepton2.momentum.y = l2_tlv.Py();
                lepton2.momentum.z = l2_tlv.Pz();
                lepton2.energy = l2_tlv.E();
                lepton2.mass = l2_tlv.M();
                lepton2.charge = (d2.PDG > 0) ? -1 : 1;
                result.push_back(lepton2);
                
                return result; // Return first valid Z->ll that's not from Higgs
            }
        }
    }
    
    // If no valid Z->ll found, return 3 dummy particles
    auto dummy = edm4hep::ReconstructedParticleData();
    dummy.momentum.x = 0;
    dummy.momentum.y = 0;
    dummy.momentum.z = 0;
    dummy.mass = -999;
    dummy.energy = -999;
    result.push_back(dummy);
    result.push_back(dummy);
    result.push_back(dummy);
    return result;
}


// Helper function to match reconstructed leptons to MC truth leptons
// Returns indices of reco leptons that best match the true Z leptons
// Input: reconstructed leptons, MC true Z info, reco-to-MC index mapping
inline Vec_i match_reco_to_MC_Z(Vec_rp reco_leptons, Vec_rp true_z_leptons, Vec_i recind, Vec_i mcind) {
    Vec_i result;
    
    // If true Z info is dummy (mass < 0), return empty
    if (true_z_leptons.size() < 3 || true_z_leptons[0].mass < 0) {
        return result;
    }
    
    // Try to find which reco leptons match the true Z leptons
    std::vector<int> matched_indices;
    
    for (size_t i = 0; i < recind.size(); ++i) {
        int reco_idx = recind[i];
        int mc_idx = mcind[i];
        
        if (reco_idx < 0 || reco_idx >= reco_leptons.size()) continue;
        if (mc_idx < 0) continue; // Not matched to MC
        
        // Check if this reco particle matches one of the true Z leptons
        // by comparing the MC index with the true Z daughters
        // (We encode MC indices into the true_z_leptons via energy field as a workaround)
        // For now, just collect all matched reco indices
        matched_indices.push_back(reco_idx);
    }
    
    // Return matched indices
    for (int idx : matched_indices) {
        result.push_back(idx);
    }
    
    return result;
}


// Evaluate how well a reconstructed Z pair matches the true Z from MC
// Returns 1 if the pair matches, 0 otherwise
// This is useful for optimizing chi2_recoil_frac
// Uses track association via getTrack2MC_index for robust MC matching
inline int evaluateZ_pairing(Vec_rp reco_z_pair, Vec_rp true_z_leptons, Vec_rp reco_leptons, Vec_i recind, Vec_i mcind, Vec_mc mc, Vec_i parents, Vec_i daughters) {
    // reco_z_pair: [0] = Z system, [1] = lepton1, [2] = lepton2 from resonanceBuilder_mass_recoil
    // true_z_leptons: [0] = Z system, [1] = lepton1, [2] = lepton2 from getTrueZ_from_H_decay
    
    if (reco_z_pair.size() < 3 || true_z_leptons.size() < 3) return 0;
    if (true_z_leptons[0].mass < 0 || reco_z_pair[0].mass < 0) return 0; // Dummy particles
    
    // Get MC indices of the true Z leptons
    // We need to find which MC particles they correspond to
    std::vector<int> true_mc_indices;
    for (size_t i = 0; i < mc.size(); ++i) {
        const auto& particle = mc[i];
        
        // Check if this MC particle matches one of the true Z leptons by 4-momentum
        for (int j = 1; j < 3; ++j) { // Check both true leptons [1] and [2]
            const auto& true_lep = true_z_leptons[j];
            
            // Compare 4-momentum (momentum + mass should match exactly for MC)
            const float tol = 1e-6;
            float dp_x = std::abs(particle.momentum.x - true_lep.momentum.x);
            float dp_y = std::abs(particle.momentum.y - true_lep.momentum.y);
            float dp_z = std::abs(particle.momentum.z - true_lep.momentum.z);
            float dm = std::abs(particle.mass - true_lep.mass);
            
            if (dp_x < tol && dp_y < tol && dp_z < tol && dm < tol) {
                true_mc_indices.push_back(i);
                break; // Found match for this true lepton
            }
        }
    }
    
    if (true_mc_indices.size() < 2) return 0; // Couldn't identify true MC leptons
    
    // Get MC indices of the reconstructed Z leptons using track association
    std::vector<int> reco_mc_indices;
    
    for (int k = 1; k < 3; ++k) { // Check both reco leptons [1] and [2]
        const auto& reco_lep = reco_z_pair[k];
        
        // Use track association to find MC particle
        if (reco_lep.tracks_begin >= 0) {
            int mc_idx = ReconstructedParticle2MC::getTrack2MC_index(reco_lep.tracks_begin, recind, mcind, reco_leptons);
            if (mc_idx >= 0) {
                reco_mc_indices.push_back(mc_idx);
            }
        }
    }
    
    if (reco_mc_indices.size() < 2) return 0; // Couldn't match reco leptons to MC
    
    // Check if reconstructed leptons match true Z leptons
    // Both leptons from reco_z_pair should be in the true_mc_indices list
    int matches = 0;
    for (int reco_mc : reco_mc_indices) {
        for (int true_mc : true_mc_indices) {
            if (reco_mc == true_mc) {
                matches++;
                break;
            }
        }
    }
    
    // Return 1 only if both reconstructed leptons match true Z leptons
    return (matches == 2) ? 1 : 0;
}

}

#endif