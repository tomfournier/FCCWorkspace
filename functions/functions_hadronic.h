#include "FCCAnalyses/defines.h"
#include <TLorentzVector.h>
#include <edm4hep/ReconstructedParticleData.h>
#include <cmath>
#include <iostream>

namespace FCCAnalyses {

inline constexpr float frac_mz_240 = 1.0f;
inline constexpr float frac_pz_240 = 1.0f;
inline constexpr float frac_rec_240 = 1.0f;
inline constexpr float pz_240 = 52.0f;

inline constexpr float frac_mz_365 = 5.0f;
inline constexpr float frac_pz_365 = 1.0f;
inline constexpr float frac_rec_365 = 1.0f;
inline constexpr float pz_365 = 143.0f;

inline constexpr float mw = 80.4;


inline Vec_tlv pair_WW_N4(Vec_rp in) {
    // assume 4 input jets
    Vec_tlv ret;

    TLorentzVector j1, j2, j3, j4, W1, W2;
    j1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);
    j2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);
    j3.SetXYZM(in[2].momentum.x, in[2].momentum.y, in[2].momentum.z, in[2].mass);
    j4.SetXYZM(in[3].momentum.x, in[3].momentum.y, in[3].momentum.z, in[3].mass);

    float chi2_1 = std::pow((j1+j2).M()-mw, 2) + std::pow((j3+j4).M()-mw, 2);
    float chi2_2 = std::pow((j1+j3).M()-mw, 2) + std::pow((j2+j4).M()-mw, 2);
    float chi2_3 = std::pow((j1+j4).M()-mw, 2) + std::pow((j2+j3).M()-mw, 2);

    if(chi2_1<chi2_2 && chi2_1<chi2_3) {
        W1 = j1+j2;
        W2 = j3+j4;
    }
    else if(chi2_2<chi2_1 && chi2_2<chi2_3) {
        W1 = j1+j3;
        W2 = j2+j4;
    }
    else if(chi2_3<chi2_1 && chi2_3<chi2_2) {
        W1 = j1+j4;
        W2 = j2+j3;
    }

    ret.push_back(W1);
    ret.push_back(W2);
    return ret;
}


inline Vec_f pair_W_p(Vec_rp in) {
    // assume 4 input jets
    Vec_f ret;

    TLorentzVector j1, j2, j3, j4;
    j1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);
    j2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);
    j3.SetXYZM(in[2].momentum.x, in[2].momentum.y, in[2].momentum.z, in[2].mass);
    j4.SetXYZM(in[3].momentum.x, in[3].momentum.y, in[3].momentum.z, in[3].mass);

    float chi2_1 = std::pow((j1+j2).M()-mw, 2) + std::pow((j3+j4).M()-mw, 2);
    float chi2_2 = std::pow((j1+j3).M()-mw, 2) + std::pow((j2+j4).M()-mw, 2);
    float chi2_3 = std::pow((j1+j4).M()-mw, 2) + std::pow((j2+j3).M()-mw, 2);
    
    float w1 = -999.;
    float w2 = -999.;
    if(chi2_1<chi2_2 && chi2_1<chi2_3) {
        w1 = (j1+j2).P();
        w2 = (j3+j4).P();
    }
    else if(chi2_2<chi2_1 && chi2_2<chi2_3) {
        w1 = (j1+j3).P();
        w2 = (j2+j4).P();
    }
    else if(chi2_3<chi2_1 && chi2_3<chi2_2) {
        w1 = (j1+j4).P();
        w2 = (j2+j3).P();
    }
    ret.push_back(w1);
    ret.push_back(w2);
    return ret;
}


inline float pair_W_dphi(Vec_rp in) {
    TLorentzVector j1, j2, j3, j4;
    j1.SetXYZM(in[0].momentum.x, in[0].momentum.y, in[0].momentum.z, in[0].mass);
    j2.SetXYZM(in[1].momentum.x, in[1].momentum.y, in[1].momentum.z, in[1].mass);
    j3.SetXYZM(in[2].momentum.x, in[2].momentum.y, in[2].momentum.z, in[2].mass);
    j4.SetXYZM(in[3].momentum.x, in[3].momentum.y, in[3].momentum.z, in[3].mass);

    float chi2_1 = std::pow((j1+j2).M()-mw, 2) + std::pow((j3+j4).M()-mw, 2);
    float chi2_2 = std::pow((j1+j3).M()-mw, 2) + std::pow((j2+j4).M()-mw, 2);
    float chi2_3 = std::pow((j1+j4).M()-mw, 2) + std::pow((j2+j3).M()-mw, 2);
    
    float ret = -999;
    if(chi2_1<chi2_2 && chi2_1<chi2_3)      { ret = (j1+j2).DeltaPhi((j3+j4)); }
    else if(chi2_2<chi2_1 && chi2_2<chi2_3) { ret = (j1+j3).DeltaPhi((j2+j4)); }
    else if(chi2_3<chi2_1 && chi2_3<chi2_2) { ret = (j1+j4).DeltaPhi((j2+j3)); }
    return std::abs(ret);
}


inline Vec_rp jets2rp(Vec_f px, Vec_f py, Vec_f pz, Vec_f e, Vec_f m) {
    Vec_rp ret;
    for(int i = 0; i < px.size(); i++) {
        edm4hep::ReconstructedParticleData p;
        p.momentum.x = px[i];
        p.momentum.y = py[i];
        p.momentum.z = pz[i];
        p.mass   = m[i];
        p.energy = e[i];
        p.charge = 0;
        ret.push_back(p);
    }
    return ret;
}


inline Vec_rp select_jets(Vec_rp in, std::vector<std::vector<int>> constituents, int njets_sel, Vec_rp reco) {
    // njets_sel = the current njets clustering algo
    Vec_rp ret;
    for(int i = 0; i < in.size(); i++) {
        float p = std::sqrt(in[i].momentum.x*in[i].momentum.x + in[i].momentum.y*in[i].momentum.y + in[i].momentum.z*in[i].momentum.z);
        if(p < 5) continue; // at least 5 GeV momemtum
        ret.push_back(in[i]);
    }
    return ret;
}


inline Vec_rp expand_jets(Vec_rp in, int njets) {
    if(in.size() < njets) in.resize(njets); // add empty jets
    return in;
}


inline int best_clustering_idx(Vec_f mz, Vec_f pz, Vec_f mrec, Vec_i njets, Vec_i njets_target, int ecm=240) {

    float frac_mz  = 1.0;
    float frac_pz  = 1.0;
    float frac_rec = 1.0;
    float pz_ = 52;
    if(ecm == 240) {
        frac_mz  = frac_mz_240;
        frac_pz  = frac_pz_240;
        frac_rec = frac_rec_240;
        pz_ = pz_240;
    }
    if(ecm == 365) {
        frac_mz  = frac_mz_365;
        frac_pz  = frac_pz_365;
        frac_rec = frac_rec_365;
        pz_ = pz_365;
    }

    float mz_ = 91.2;
    float mh_ = 125.0;

    Vec_f chi2;
    for(int i = 0; i < mz.size(); i++) {

        float c = frac_mz*std::pow(mz[i]-mz_, 2) + frac_rec*std::pow(mrec[i]-mh_, 2) + frac_pz*std::pow(pz[i]-pz_, 2);
        if(i==0 && njets[0] < 2) c = 9e99;
        else if(i > 0 && njets[i] != njets_target[i]) c = 9e99;
        chi2.push_back(c);
    }

    // extra constraints: the number of good candidate jets must be at least 2 to form a good z-candidate
    // sometimes exclusive clustering gives wrong njets
    int min_dx = std::distance(std::begin(chi2), std::min_element(std::begin(chi2), std::end(chi2)));
    if(min_dx == 0 and njets[0] >= 2) return 0; // requirement for inclusive clustering
    else if(njets[min_dx] == njets_target[min_dx] && min_dx != 0) return min_dx; // requirement for exlusive clustering
    else {
        return -1; // could not cluster
    }
}


// build the Z resonance based on the available leptons. Returns the best lepton pair compatible with the Z mass and recoil at 125 GeV
// technically, it returns a ReconstructedParticleData object with index 0 the di-lepton system, index and 2 the leptons of the pair
struct resonanceBuilder_mass_recoil_hadronic {
    float m_resonance_mass;
    float m_recoil_mass;
    float chi2_recoil_frac;
    float ecm;
    resonanceBuilder_mass_recoil_hadronic(float arg_resonance_mass, float arg_recoil_mass, float arg_chi2_recoil_frac, float arg_ecm);
    Vec_rp operator()(Vec_rp legs);
};

inline resonanceBuilder_mass_recoil_hadronic::resonanceBuilder_mass_recoil_hadronic(float arg_resonance_mass, float arg_recoil_mass, float arg_chi2_recoil_frac, float arg_ecm) 
    {m_resonance_mass = arg_resonance_mass, m_recoil_mass = arg_recoil_mass, chi2_recoil_frac = arg_chi2_recoil_frac, ecm = arg_ecm;}

inline Vec_rp resonanceBuilder_mass_recoil_hadronic::operator()(Vec_rp legs) {
    float frac_mz  = 1.0;
    float frac_pz  = 1.0;
    float frac_rec = 1.0;
    float pz_ = 52;
    if(ecm == 240) {
        frac_mz  = frac_mz_240;
        frac_pz  = frac_pz_240;
        frac_rec = frac_rec_240;
        pz_ = pz_240;
    }
    if(ecm == 365) {
        frac_mz  = frac_mz_365;
        frac_pz  = frac_pz_365;
        frac_rec = frac_rec_365;
        pz_ = pz_365;
    }

    Vec_rp result;
    result.reserve(3);
    std::vector<std::vector<int>> pairs; // for each permutation, add the indices of the jets
    int n = legs.size();
    if(n > 1) {
        ROOT::VecOps::RVec<bool> v(n);
        std::fill(v.end() - 2, v.end(), true); // helper variable for permutations
        do {
            std::vector<int> pair;
            rp reso;
            reso.charge = 0;
            TLorentzVector reso_lv;
            Vec_f jet_momenta;
            for(int i = 0; i < n; ++i) {
                if(v[i]) {
                    pair.push_back(i);
                    TLorentzVector leg_lv;
                    leg_lv.SetXYZM(legs[i].momentum.x, legs[i].momentum.y, legs[i].momentum.z, legs[i].mass);
                    reso_lv += leg_lv;
                    jet_momenta.push_back(leg_lv.P());
                }
            }

            reso.momentum.x = reso_lv.Px();
            reso.momentum.y = reso_lv.Py();
            reso.momentum.z = reso_lv.Pz();
            reso.mass = reso_lv.M();
            
            result.emplace_back(reso);
            pairs.push_back(pair);

        } while(std::next_permutation(v.begin(), v.end()));
    }
    else { return result; }

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

            float momentum = tg.P();
            float mass = frac_mz  * std::pow(result.at(i).mass - m_resonance_mass, 2); // mass
            float rec  = frac_rec * std::pow(recoil_fcc.mass - m_recoil_mass, 2); // recoil
            float p    = frac_pz  * std::pow(momentum - pz_, 2);
            float d = mass + rec + p;

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
            std::cout << "ERROR: resonanceBuilder_mass_recoil, no mininum found." << std::endl;
            exit(1);
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

}