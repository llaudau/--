#include "lattice_phi4.h"
#include <algorithm> // For std::copy
#include <vector>
#include <chrono>

double rd_gen0to1() {
    static auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // 2. The generator is initialized ONCE using the time-based seed.
    static std::mt19937 generator(seed);
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}
bool Accept(double delta_H){
    if (delta_H<0){
        return true;
    }
    else{
        if (std::exp(-delta_H)>rd_gen0to1()){
            return true;
        }
        else{
            return false;
        }
    }
    

}

Vector4i phi4_Lattice::per_add(Vector4i a,Vector4i b){
        Vector4i c =a+b;
        c[0]=(c[0]+this->Nt)%this->Nt;
        c[1]=(c[1]+this->Ns)%this->Ns;
        c[2]=(c[2]+this->Ns)%this->Ns;
        c[3]=(c[3]+this->Ns)%this->Ns;
        return c;
    };

void phi4_Lattice::InitializeMoMToRandom(){
    double* mom_field0=this->get_momfield_write().data();
    int volume=this->Ns*this->Ns*this->Ns*this->Nt;
    // Fill momentum with Gaussian(0,1) using Box-Muller. Handle odd volume.
    int i = 0;
    while (i + 1 < volume) {
        double x1 = rd_gen0to1();
        double x2 = rd_gen0to1();
        double r = std::sqrt(-2.0 * std::log(x1));
        mom_field0[i++] = r * std::cos(2.0 * M_PI * x2);
        mom_field0[i++] = r * std::sin(2.0 * M_PI * x2);
    }
    if (i < volume) {
        // one remaining element
        double x1 = rd_gen0to1();
        double x2 = rd_gen0to1();
        mom_field0[i] = std::sqrt(-2.0 * std::log(x1)) * std::cos(2.0 * M_PI * x2);
    }
    return;
}

// Helper Function 1: Convert 1D index to 4D coordinates
Vector4i phi4_Lattice::oneD_to_fourD(int i) const {
    Vector4i cord;
    int volume_s = this->Ns * this->Ns * this->Ns; // Volume of a spatial slice (T-slice)
    int volume_sx = this->Ns * this->Ns;
    int volume_sxy = this->Ns;

    // t-coordinate
    cord(0) = i / volume_s;
    // x-coordinate
    cord(1) = (i % volume_s) / volume_sx;
    // y-coordinate
    cord(2) = (i % volume_sx) / volume_sxy;
    // z-coordinate
    cord(3) = i % volume_sxy;

    return cord;
}
// Helper Function 2: Convert 4D coordinates to 1D index
int phi4_Lattice::fourD_to_oneD(const Vector4i& cord) const {
    // Check boundaries (optional but good practice)
    // You should ensure coordinates are already within bounds [0, N-1]

    int volume_s = this->Ns * this->Ns * this->Ns;
    int volume_sx = this->Ns * this->Ns;
    int volume_sxy = this->Ns;
    
    // Formula: t + x*Nt + y*Nt*Nx + z*Nt*Nx*Ny, but adapted to your structure
    return cord(0) * volume_s + 
           cord(1) * volume_sx + 
           cord(2) * volume_sxy + 
           cord(3);
}

void phi4_Lattice::initialize_neighbor_indices() {
    const int volume = this->Nt * this->Ns * this->Ns * this->Ns;
    
    // Allocate memory for the lookup table (must be done only once)
    // NOTE: If you use std::vector<std::array<int, 8>>, skip manual allocation
    this->neighbor_1D_indices = new int[volume][8];

    // Iterate over every site
    for (int i = 0; i < volume; ++i) {
        
        // Step 1: Get current 4D coordinate
        Vector4i current_cord = this->oneD_to_fourD(i);

        // Step 2 & 3: Calculate the 8 neighbors (+mu and -mu)
        for (int mu = 0; mu < 4; ++mu) {
            
            // --- Positive direction (+mu) ---
            Vector4i plus_cord = current_cord;
            plus_cord(mu) += 1; // Move 1 step in the +mu direction

            // Apply Periodic Boundary Condition (PBC)
            // t-dimension (mu=0) has size Nt, spatial dimensions (mu=1,2,3) have size Ns
            int max_dim = (mu == 0) ? this->Nt : this->Ns;
            
            // PBC: (coord + 1) % N
            if (plus_cord(mu) >= max_dim) {
                plus_cord(mu) = 0; // Wrap around to 0
            }

            // Step 4: Convert back to 1D index and store (even indices: 0, 1, 2, 3)
            this->neighbor_1D_indices[i][mu] = this->fourD_to_oneD(plus_cord);


            // --- Negative direction (-mu) ---
            Vector4i minus_cord = current_cord;
            minus_cord(mu) -= 1; // Move 1 step in the -mu direction

            // Apply Periodic Boundary Condition (PBC)
            // PBC: (coord - 1 + N) % N
            if (minus_cord(mu) < 0) {
                minus_cord(mu) = max_dim - 1; // Wrap around to N-1
            }

            // Step 4: Convert back to 1D index and store (odd indices: 4, 5, 6, 7)
            this->neighbor_1D_indices[i][mu + 4] = this->fourD_to_oneD(minus_cord);
        }
    }
    return;
}



double phi4_Lattice::Hamiltonian(){
    double act;
    const double* mom=this->get_momfield().data();
    int i, volume;
    volume=this->Ns*this->Ns*this->Ns*this->Nt;
    act=this->action();
    for (i=0;i<volume;i++) act+=mom[i]*mom[i]/2.0;
    return act;
}

double phi4_Lattice::action(){
    double phin, phi2, action=0;
    int mu, volume=this->Nt*this->Ns*this->Ns*this->Ns;
    double kappa=this->Chi;
    double lambda=this->Lada;
    // int * hopping=this->neighbor_1D_indices;

    const double * phi4_data=this->get_phi4field().data();

    for (int i=0; i<volume; i++){
        phin=0;
        for (mu=0 ;mu<4 ;mu++ ) phin+=phi4_data[this->neighbor_1D_indices[i][mu]];
        phi2=phi4_data[i]*phi4_data[i];
        action+=-2*kappa*phin*phi4_data[i]+phi2+lambda*(phi2-1.0)*(phi2-1.0);
    }
    return action;
}


void phi4_Lattice::update1(double epsi){
    double* phi4_data = this->get_phi4field_write().data(); 
    const double* mom_data = this->get_momfield().data();
    const int volume = this->Nt * this->Ns * this->Ns * this->Ns;
    // #pragma omp parallel for
    for (int i=0; i<volume;i++ ){
        phi4_data[i]=phi4_data[i]+epsi*mom_data[i];
    }
    return;
}
    
void phi4_Lattice::update2(double epsi){
    int volume=this->Nt*this->Ns*this->Ns*this->Ns;
    const double* phi4_data = this->get_phi4field().data(); 
    double* mom_data = this->get_momfield_write().data();
    // #pragma omp parallel for
    for (int i=0; i<volume;i++){
        double phin=0.0;
        for (int mu=0;mu<8;mu++) phin+=phi4_data[this->neighbor_1D_indices[i][mu]];

        // The force should equal -d(action)/d(phi). For the action implemented
        // the correct contribution from the nearest-neighbor term carries a factor 4.
        double force = 2.0 * this->Chi * phin - 4.0 * this->Lada * (phi4_data[i]*phi4_data[i] - 1.0) * phi4_data[i] - 2.0 * phi4_data[i];

        // momentum update: p <- p + eps * force  (force is -dS/dphi)
        mom_data[i] += epsi * force;
        }
    return;
}

void phi4_Lattice::update_HMC(int times,double epsi){
    for (int i=0; i<times; i++){
        this->update1(epsi/2.);
        this->update2(epsi);
        this->update1(epsi/2.);
    }
    return;
}
void phi4_Lattice::update_HMC_acc(int times, double epsi){

    int volume=this->Ns*this->Ns*this->Ns*this->Nt;
    std::vector<double> phi4_old_state(volume);
    this->InitializeMoMToRandom();

    const double* phi4_data= this->get_phi4field().data();
    for (int i=0; i < volume; i++) phi4_old_state[i] = phi4_data[i];

    

    double H0_old=this->Hamiltonian();

    this->update_HMC(times,epsi);
    
    double H0_new=this->Hamiltonian();
    double dH=H0_new-H0_old;
    // std::cout<<dH<<std::endl;
    this->expdeltaH+=std::exp(-dH);

    if (not Accept(dH)){
        double* phi4_data1 = this->get_phi4field_write().data();
        for (int i=0; i <volume;i++) phi4_data1[i]=phi4_old_state[i];
    }
    else {
        this->accept+=1;
    }
    // std::cout<<"new ? "<<std::endl;
    // std::cout<<phi4_data[0]<<std::endl;
    return;
}

void phi4_Lattice::update_HMC_whole_process(int thermal, int ntraj, int times, double epsi){
    for (int i=0;i<thermal;i++){
        this->update_HMC_acc(times,epsi);
    }
    this->expdeltaH=0;
    this->accept=0;
    for (int i=0;i<ntraj;i++){
        // std::cout<<this->get_phi4element(cord)<<std::endl;
        this->update_HMC_acc(times,epsi);
        if ((i+1)%thermal==0){
            std::cout<<i+1<<std::endl;
            std::cout<<this->accept/(thermal+0.0)<< std::endl;
            std::cout<<this->expdeltaH/(thermal+0.0)<<std::endl;
            this->expdeltaH=0;
            this->accept=0;
        }
    }
    return;
}




double phi4_Lattice::Magnetic(){
    const double  *phi4_data= this->get_phi4field().data();
    int volume= this->Nt * this->Ns * this->Ns * this->Ns;
    double mag=0.;
    for (int i=0 ; i<volume ; i++){
        mag+=phi4_data[i];
    }
    mag=mag/volume;
    return mag;
}
double phi4_Lattice::Magnetic_sqa(){
    const double  *phi4_data= this->get_phi4field().data();
    int volume= this->Nt * this->Ns * this->Ns * this->Ns;
    double mag=0.;
    for (int i=0 ; i<volume ; i++){
        mag+=phi4_data[i]*phi4_data[i];
    }
    mag=mag/volume;
    return mag;
}
double phi4_Lattice::Magnetic_sqasqa(){
    const double  *phi4_data= this->get_phi4field().data();
    int volume= this->Nt * this->Ns * this->Ns * this->Ns;
    double mag=0.;
    for (int i=0 ; i<volume ; i++){
        mag+=phi4_data[i]*phi4_data[i]*phi4_data[i]*phi4_data[i];
    }
    mag=mag/volume;
    return mag;
}
double phi4_Lattice::Binder_cu(){
    return this->Magnetic_sqasqa()/(this->Magnetic_sqa()*this->Magnetic_sqa());
}