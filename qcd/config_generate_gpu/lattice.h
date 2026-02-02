#ifndef LATTICE_H
#define LATTICE_H
#include <curand_kernel.h>
#include <iostream>
class Lattice {
private:
    int Ns, Nt,volume;
    int threadsPerBlock = 256;
    int blocksPerGrid;
    double lambda,beta;
    

    // field and momentum field
    double* data;  // field value per site 
    double* mom;

    int* hopping_field;

    double* d_data = nullptr;
    double* d_old_data=nullptr;
    double* m_data = nullptr;
    double *d_energy_array=nullptr;
    int* hopping_data=nullptr;
    curandState* d_rng = nullptr;



    // initialize hopping field
    int calculate_neighbor(int v,int i);
    void calculate_hopping();
    inline int index(int x, int y, int z, int t) const;
public:
    double expdh=0.0;

    Lattice(int Ns_, int Nt_, double lambda_,double beta_);
    ~Lattice();
    
    //index methods
    double& operator()(int x, int y, int z, int t);

    void randomize(unsigned int seed = 1234);
    void cuda_init(unsigned long seed);
    // void cuda_update_sweep();
    bool cuda_update_trajectory(double epsi, int num);
    void cuda_finalize();
    double Hamiltonian(double* data, double* mom, int Ns, int Nt);

};

#endif
