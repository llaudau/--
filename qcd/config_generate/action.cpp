#include "lattice.h"


void Lattice::Action_i(Vector4i cord,int mu){
    // origin link 
    SU3Matrix Un0=this->get_link(cord,mu);

    // A matrix for action calculation
    SU3Matrix A_act = this->A(cord,mu);
    double betain=this->Beta;

    this->action+=betain* std::real((SU3Matrix::Identity()*6-Un0*A_act).trace())/3.0;

    return;
}

void Lattice::Action_all(){
    int volume=this->Nt*this->Ns*this->Ns*this->Ns;
    int volumet=this->Ns*this->Ns*this->Ns;
    int volumetx=this->Ns*this->Ns;
    int volumetxy=this->Ns;
    #pragma omp parallel for
    for (int i=0; i<volume;i++ ){
        Vector4i cord;
        cord(0)=i/volumet;
        cord(1)=(i%volumet)/volumetx;
        cord(2)=(i%volumetx)/volumetxy;
        cord(3)=i%volumetxy;
        if ((cord(0)+cord(1)+cord(2)+cord(3))%2==0){
            for (int mu=0; mu<4 ;mu++){
            this->Action_i(cord,mu);
        }
        }
    }
    #pragma omp parallel for
    for (int i=0; i<volume;i++ ){
        Vector4i cord;
        cord(0)=i/volumet;
        cord(1)=i%volumet/volumetx;
        cord(2)=i%volumet%volumetx/volumetxy;
        cord(3)=i%volumet%volumetx%volumetxy;
        if ((cord(0)+cord(1)+cord(2)+cord(3))%2==1){
            for (int mu=0; mu<4 ;mu++){
            this->Action_i(cord,mu);
        }
        }
    }
    return;
}