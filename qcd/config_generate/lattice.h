#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>

#pragma once
using namespace Eigen;
using ComplexD = std::complex<double>;
using SU3Matrix = Matrix<ComplexD, 3, 3>;
using SU2Matrix =Matrix<ComplexD, 2, 2>;
using ComplexD = std::complex<double>;
const int RANK = 5;
using GaugeFieldType = Tensor<SU3Matrix, RANK>;

SU2Matrix R(double epsi);
SU3Matrix RST(double epsi);
class Lattice {
private:
    int Ns;    
    int Nt; 

    // interaction parameter beta=2/g^2
    double Beta;  
    GaugeFieldType gauge_field; 

public:
    //periodical add of coordinates
    Vector4i per_add(Vector4i a,Vector4i b);
    // constructor of lattice
    Lattice(int size_t, int size_s, double beta): 
        Ns(size_s), Nt(size_t) , Beta(beta)
    {
        Eigen::array<int, RANK> dimensions = {Nt, Ns, Ns, Ns, 4};
        gauge_field.resize(dimensions);
        InitializeFieldToIdentity();
    }
    
    // Initialization method
    void InitializeFieldToIdentity() {
        SU3Matrix Identity = SU3Matrix::Identity();
        gauge_field.setConstant(Identity);
    }

    // use this function to save gauge field 
    const GaugeFieldType& get_gaugefield() const{
        return gauge_field;
    }

    // Read (only) direct value of lattice point
    const SU3Matrix& get_link(Vector4i cord, int mu) const {
        return gauge_field(cord[0], cord[1], cord[2], cord[3], mu);
    }
    SU3Matrix& set_link(Vector4i cord, int mu) {
        return gauge_field(cord[0], cord[1], cord[2], cord[3], mu);
    }
    
    // Calcualte the staple matrixes in action 
    SU3Matrix A(Vector4i coord,int mu);

    // change SU3Matrix at certain coordinate accordint to a certain prior epsi
    void update(Vector4i cord,int mu, double epsi);
    
    // update all links in 1 function using optimization, update half of lattice point in one time.
    void update_all(double epsi,int try_each);
};
