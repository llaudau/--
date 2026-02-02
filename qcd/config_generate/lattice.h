#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <complex>
#include <random>
#include <cmath>
#include <algorithm>


#pragma once
using namespace Eigen;
using ComplexD = std::complex<double>;
using SU3Matrix = Matrix<ComplexD, 3, 3>;
using SU2Matrix =Matrix<ComplexD, 2, 2>;
using ComplexD = std::complex<double>;
const int RANK = 5;
const int RANKshit=2;
using GaugeFieldType = Tensor<SU3Matrix, RANK>;

SU2Matrix R(double epsi);
SU3Matrix RST(double epsi);
SU3Matrix generate_random_su3();
double acpt_rd_num();


class Lattice {
private:
    int Ns;    
    int Nt; 
    

    // interaction parameter beta=2/g^2
    double Beta;  
    GaugeFieldType gauge_field; 

public:
    int successtime=0;
    double action=0;
    //periodical add of coordinates
    Vector4i per_add(Vector4i a,Vector4i b);
    // constructor of lattice
    Lattice(int size_t, int size_s, double beta): 
        Ns(size_s), Nt(size_t) , Beta(beta)
    {
        Eigen::array<int, RANK> dimensions = {Nt, Ns, Ns, Ns, 4};
        gauge_field.resize(dimensions);
        InitializeFieldToHot();
    }
    
    // Initialization method
    void InitializeFieldToIdentity() {
        SU3Matrix Identity = SU3Matrix::Identity();
        gauge_field.setConstant(Identity);
    }
    void InitializeFieldToHot();
    
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

    void Action_i(Vector4i coord,int mu);
    void Action_all();


    Tensor<ComplexD,RANKshit> Wilsonloopshit();

    ComplexD Wilsonloop(int dt,int dx, int dy,int dz);

    ComplexD Wilsonloop_i(Vector4i cord,int dt,int dx, int dy,int dz);
    // change SU3Matrix at certain coordinate accordint to a certain prior epsi
    void update(Vector4i cord,int mu, double epsi,int times);
    
    // update all links in 1 function using optimization, update half of lattice point in one time.
    void update_all(double epsi,int try_each);

    // Topological charge computation (clover definition)
    double topological_charge() const;
    double topological_charge_density(Vector4i site) const;
    SU3Matrix compute_plaquette(Vector4i site, int mu, int nu) const;
    SU3Matrix field_strength_from_plaquette(SU3Matrix plaq) const;

    // include thermalize and 
};
