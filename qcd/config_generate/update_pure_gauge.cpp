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



void Lattice::update(Vector4i cord,int mu,double epsi,int times){
    // origin link and candidate link
    SU3Matrix Un0=this->get_link(cord,mu);
    SU3Matrix Un_jump=Un0;

    // A matrix for action calculation
    SU3Matrix A_act = this->A(cord,mu);
    double betain=this->Beta;

    // S'[U']-S[U] (here i update for 'times' times to save the bandwidth of cores)
    for(int i=0;i<times;i++){
        double delt_actnow;
        SU3Matrix rd_su3=RST(epsi);
        delt_actnow=betain* std::real(((-rd_su3*Un_jump+Un_jump)*A_act).trace())/3;
        // accept or not:
        if (acpt_rd_num()<std::min(std::exp(-delt_actnow),1.0) ){
            Un_jump=rd_su3*Un_jump;

            #pragma omp atomic update
            this->successtime+=1;
            
        }
    }
    this->set_link(cord,mu)=Un_jump;
    return;
}

void Lattice::update_all(double epsi,int try_each){
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
            for(int mu=0; mu<4;mu++){
                this->update(cord,mu,epsi,try_each);
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
            for(int mu=0; mu<4;mu++){           
                this->update(cord,mu,epsi,try_each);
            }
        }
    }
    return;
}