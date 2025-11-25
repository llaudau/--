#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <cmath>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#pragma once

using namespace Eigen;
const int RANKphi4 = 4;
using phi4_field = Tensor<double, RANKphi4>;

// double rd_gen(double epsi0);


class phi4_Lattice{
    private :
        int Ns;
        int Nt;

        double Chi;
        double Lada;

        phi4_field field;
        phi4_field mom_field;

    public:
        int accept=0;
        double expdeltaH=0;
        int (*neighbor_1D_indices)[8];
        //periodical add of coordinates
        Vector4i per_add(Vector4i a,Vector4i b);
        phi4_Lattice(int size_t, int size_s, double chi_value,double lada_value): 
        Ns(size_s), Nt(size_t) , Chi(chi_value), Lada (lada_value)
    {
        Eigen::array<int, RANKphi4> dimensions = {Nt, Ns, Ns, Ns};

        field.resize(dimensions);
        mom_field.resize(dimensions);
        InitializeFieldToRandom();
        InitializeMoMToRandom();
        initialize_neighbor_indices();
    }
    
    // Initialization method
    void InitializeFieldToZeros() {
        field.setZero();   
    }
    void InitializeFieldToRandom(){
        field.setRandom();
    }
    void InitializeMoMToRandom();
    // initialize the indice list
    Vector4i oneD_to_fourD(int i) const;
    void initialize_neighbor_indices();
    int fourD_to_oneD(const Vector4i& cord) const;

    // use this function to save gauge field 
    phi4_field& get_phi4field_write(){
        return field;
    }
    phi4_field& get_momfield_write(){
        return mom_field;
    }
    const phi4_field& get_phi4field()const{
        return field;
    }
    const phi4_field& get_momfield()const{
        return mom_field;
    }

    // Read (only) direct value of lattice point
    const double& get_phi4element(Vector4i cord) const {
        return field(cord[0], cord[1], cord[2], cord[3]);
    }
    double& set_phi4element(Vector4i cord) {
        return field(cord[0], cord[1], cord[2], cord[3]);
    }
    const double& get_momelement(Vector4i cord) const {
        return mom_field(cord[0], cord[1], cord[2], cord[3]);
    }
    double& set_momelement(Vector4i cord) {
        return mom_field(cord[0], cord[1], cord[2], cord[3]);
    }

    // Calcualte the neibor in action 
    double action();
    double Hamiltonian();

    // update according to HMC algorithm :3 steps :update phi, pi, phi(the same as the first step)
    void update1(double epsi);
    void update2(double epsi);
    void update_HMC(int times,double epsi);
    void update_HMC_acc(int times, double epsi);
    void update_HMC_whole_process(int thermal, int ntraj, int times, double epsi);

    // calculate <m> of the lattice 
    double Magnetic();
    double Magnetic_sqa();
    double Magnetic_sqasqa();
    double Binder_cu();
};