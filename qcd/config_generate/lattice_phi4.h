#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

#pragma once

using namespace Eigen;
const int RANKphi4 = 4;
using phi4_field = Tensor<double, RANKphi4>;

double rd_gen(double epsi);

class phi4_Lattice{
    private :
        int Ns;
        int Nt;

        double Chi;
        double Lada;

        phi4_field field;

    public:
        //periodical add of coordinates
        Vector4i per_add(Vector4i a,Vector4i b);
        phi4_Lattice(int size_t, int size_s, double chi_value,double lada_value): 
        Ns(size_s), Nt(size_t) , Chi(chi_value), Lada (lada_value)
    {
        Eigen::array<int, RANKphi4> dimensions = {Nt, Ns, Ns, Ns};

        field.resize(dimensions);
        InitializeFieldToZeros();
    }
    
    // Initialization method
    void InitializeFieldToZeros() {
        field.setZero();   
    }

    // use this function to save gauge field 
    const phi4_field& get_gaugefield() const{
        return field;
    }

    // Read (only) direct value of lattice point
    const double& get_element(Vector4i cord) const {
        return field(cord[0], cord[1], cord[2], cord[3]);
    }
    double& set_element(Vector4i cord) {
        return field(cord[0], cord[1], cord[2], cord[3]);
    }
    
    // Calcualte the neibor in action 
    double neibor(Vector4i coord);

    // change point at certain coordinate accordint to a certain prior epsi
    void update(Vector4i cord, double epsi,int times);
    
    // update all links in 1 function using optimization, update half of lattice point in one time.
    void update_all(double epsi,int try_each);
};