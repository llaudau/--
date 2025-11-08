#include "lattice.h"
#include "constant.h"


// random SU2 and random SU3 matrix with an epsi parameter
SU2Matrix R(double epsi){

    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<> d(-0.5, 0.5); 
    std::uniform_int_distribution<> e(0,1);

    // random vector r
    int r0=2*e(gen)-1 ; 
    Vector3d r;
    r<<d(gen),d(gen), d(gen);
    r=r/r.norm();

    // random SU2 matrix
    SU2Matrix R0=SU2Matrix::Zero();
    R0+=r0*std::sqrt(1-epsi*epsi)*SU2Matrix::Identity();
    R0+=ComplexD(0,1)*epsi*r[0]*Sigma1;
    R0+=ComplexD(0,1)*epsi*r[1]*Sigma2;
    R0+=ComplexD(0,1)*epsi*r[2]*Sigma3;
    return R0;
};
SU3Matrix RST(double epsi){
    SU2Matrix R0=R(epsi);
    SU2Matrix S0=R(epsi);
    SU2Matrix T0=R(epsi);
    SU3Matrix outr=SU3Matrix::Identity();
    outr.block<2,2>(0,0)=R0;

    SU3Matrix outs=SU3Matrix::Identity();
    outs(0,0)=S0(0,0);
    outs(2,0)=S0(1,0);
    outs(0,2)=S0(0,1);
    outs(2,2)=S0(1,1);

    SU3Matrix outt=SU3Matrix::Identity();
    outt.block<2,2>(1,1)=T0;
    return outr * outs * outt;
}
// acceptance random number
double acpt_rd_num() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> d(0.0, 1.0); 
    return d(gen);
}
// random initialize SU3matrix
SU3Matrix generate_random_su3() {
    // We use a standard normal distribution to fill a 3x3 complex matrix
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> normal_dist(0.0, 1.0);

    // 1. Create a random complex matrix
    Eigen::Matrix3cd M;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            M(i, j) = std::complex<double>(normal_dist(gen), normal_dist(gen));
        }
    }

    // 2. Use QR decomposition to convert the random complex matrix M
    // into a unitary matrix Q, and then normalize it to SU(3).
    // This is a standard way to get a random unitary (U(N)) matrix.
    Eigen::HouseholderQR<Eigen::Matrix3cd> qr(M);
    SU3Matrix U = qr.householderQ();

    // 3. Normalize the determinant to 1 to ensure it's in SU(3)
    // The determinant of a U(N) matrix has magnitude 1, so det = exp(i * phi).
    std::complex<double> det_U = U.determinant();
    double phase = std::arg(det_U);
    std::complex<double> su3_factor = std::exp(std::complex<double>(0.0, -phase / 3.0));

    return U * su3_factor;
}



//periodical add of coordinates
Vector4i Lattice::per_add(Vector4i a,Vector4i b){
        Vector4i c =a+b;
        c[0]=(c[0]+this->Nt)%this->Nt;
        c[1]=(c[1]+this->Ns)%this->Ns;
        c[2]=(c[2]+this->Ns)%this->Ns;
        c[3]=(c[3]+this->Ns)%this->Ns;
        return c;
    };





void Lattice::InitializeFieldToHot() {
    Vector4i cord; // Assuming Vector4i is defined (e.g., Eigen::Vector4i)

    // Loop over all spatial sites (x, y, z, t) and all four directions (mu)
    for (cord[0] = 0; cord[0] < this->Nt; ++cord[0]) {
        for (cord[1] = 0; cord[1] < this->Ns; ++cord[1]) {
            for (cord[2] = 0; cord[2] < this->Ns; ++cord[2]) {
                for (cord[3] = 0; cord[3] < this->Ns; ++cord[3]) {
                    for (int mu = 0; mu < 4; ++mu) {
                        // Assign a new random SU(3) matrix to the link
                        this->set_link(cord, mu) = generate_random_su3();
                    }
                }
            }
        }
    }
}