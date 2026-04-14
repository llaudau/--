#pragma once
#include "matrix.hpp"
#include <random>
#include <cmath>

namespace qcd {

// Gram-Schmidt reunitarization for SU(3)
inline void reunitarize(SU3& U) {
    // Normalize row 0
    double n0 = 0.0;
    for(int c=0;c<3;c++) n0 += std::norm(U(0,c));
    n0 = std::sqrt(n0);
    for(int c=0;c<3;c++) U(0,c) *= (1.0/n0);

    // Row 1: orthogonalize against row 0
    complex proj;
    for(int c=0;c<3;c++) proj += std::conj(U(0,c))*U(1,c);
    for(int c=0;c<3;c++) U(1,c) -= proj*U(0,c);
    double n1 = 0.0;
    for(int c=0;c<3;c++) n1 += std::norm(U(1,c));
    n1 = std::sqrt(n1);
    for(int c=0;c<3;c++) U(1,c) *= (1.0/n1);

    // Row 2 = (row0 x row1)*
    U(2,0) = std::conj(U(0,1)*U(1,2) - U(0,2)*U(1,1));
    U(2,1) = std::conj(U(0,2)*U(1,0) - U(0,0)*U(1,2));
    U(2,2) = std::conj(U(0,0)*U(1,1) - U(0,1)*U(1,0));
}

// Generate SU(3) matrix near identity: U ~ exp(i*eps*H), H traceless Hermitian
inline SU3 random_su3_near_identity(std::mt19937_64& rng, double eps) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    double a01=gauss(rng), b01=gauss(rng);
    double a02=gauss(rng), b02=gauss(rng);
    double a12=gauss(rng), b12=gauss(rng);
    double d0=gauss(rng), d1=gauss(rng);
    SU3 H;
    H(0,1)=complex( a01, b01); H(1,0)=complex( a01,-b01);
    H(0,2)=complex( a02, b02); H(2,0)=complex( a02,-b02);
    H(1,2)=complex( a12, b12); H(2,1)=complex( a12,-b12);
    H(0,0)=complex(d0,0.0);
    H(1,1)=complex(d1,0.0);
    H(2,2)=complex(-d0-d1,0.0);
    SU3 iH = H * complex(0.0, eps);
    SU3 U = SU3::identity() + iH + (iH * iH) * 0.5;
    reunitarize(U);
    return U;
}

// Generate random su(3) algebra element (Hermitian traceless).
// Convention: K = Tr[P²], link update U = exp(i*eps*P)*U.
// Boltzmann weight: exp(-K) = exp(-Tr[P²]).
// In algebra coords: K = (1/2)*sum(pi_a²), so pi_a ~ N(0,1).
// Off-diagonal matrix elements: Re,Im ~ N(0, 1/2).
// Diagonal: from T3, T8 generators with pi_3,pi_8 ~ N(0,1).
inline SU3 gaussian_su3_algebra(std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    static const double s = 0.5;  // off-diagonal sigma = 1/2
    static const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    double a01=gauss(rng)*s, b01=gauss(rng)*s;
    double a02=gauss(rng)*s, b02=gauss(rng)*s;
    double a12=gauss(rng)*s, b12=gauss(rng)*s;
    // Diagonal from Gell-Mann T3 and T8
    double d0 = gauss(rng) * s;
    double d1 = gauss(rng) * s * inv_sqrt3;
    SU3 H;
    H(0,1)=complex( a01, b01); H(1,0)=complex( a01,-b01);
    H(0,2)=complex( a02, b02); H(2,0)=complex( a02,-b02);
    H(1,2)=complex( a12, b12); H(2,1)=complex( a12,-b12);
    H(0,0)=complex(d0+d1,0.0);
    H(1,1)=complex(-d0+d1,0.0);
    H(2,2)=complex(-2*d1,0.0);
    return H;
}

} // namespace qcd
