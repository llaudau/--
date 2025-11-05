#include "lattice.h"
SU3Matrix Lattice::A(Vector4i coord,int mu){
        SU3Matrix A0= SU3Matrix::Zero();
        const int D = 4;
        for (int nu=0 ; nu < D ; nu++){
            if (mu == nu){
                continue;
            };
            // std::cout<<mu<<nu<<std::endl;
            Vector4i vecmu(0,0,0,0);
            Vector4i vecnu(0,0,0,0);
            vecmu[mu]+=1;
            vecnu[nu]+=1;
            A0+=this->get_link(per_add(coord,vecmu),nu)*this->get_link(per_add(coord,vecnu),mu).adjoint()*this->get_link(coord,nu).adjoint();
            A0+=this->get_link(per_add(per_add(coord,vecmu),-vecnu),nu).adjoint()*this->get_link(per_add(coord,-vecnu),mu).adjoint()*this->get_link(per_add(coord,-vecnu),nu);
        }
        return A0;
} ;



void Lattice::Action_i(Vector4i cord,int mu){
    // origin link 
    SU3Matrix Un0=this->get_link(cord,mu);

    // A matrix for action calculation
    SU3Matrix A_act = this->A(cord,mu);
    double betain=this->Beta;

    this->action+=betain* std::real((SU3Matrix::Identity()*6-Un0*A_act).trace());

    return;
}

void Lattice::Action_all(){
    int volume=this->Nt*this->Ns*this->Ns*this->Ns;
    int volumet=this->Ns*this->Ns*this->Ns;
    int volumetx=this->Ns*this->Ns;
    int volumetxy=this->Ns;
    int mu=0;
    #pragma omp parallel for
    for (int i=0; i<volume;i++ ){
        Vector4i cord;
        cord(0)=i/volumet;
        cord(1)=(i%volumet)/volumetx;
        cord(2)=(i%volumetx)/volumetxy;
        cord(3)=i%volumetxy;
        if ((cord(0)+cord(1)+cord(2)+cord(3))%2==0){
            this->Action_i(cord,mu);
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
            this->Action_i(cord,mu);
        }
    }
    this->action=this->action/3.0;
    return;
}