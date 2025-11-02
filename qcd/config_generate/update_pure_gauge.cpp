#include "lattice.h"
#include "constant.h"
#include <random>
#include <cmath>

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


//periodical add of coordinates
Vector4i Lattice::per_add(Vector4i a,Vector4i b){
        Vector4i c =a+b;
        c[0]=(c[0]+this->Nt)%this->Nt;
        c[1]=(c[1]+this->Ns)%this->Ns;
        c[2]=(c[2]+this->Ns)%this->Ns;
        c[3]=(c[3]+this->Ns)%this->Ns;
        return c;
    };

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
        delt_actnow=-betain* std::real(((Un_jump*RST(epsi)-Un_jump)*A_act).trace());
        // accept or not:
        if (delt_actnow<0 or acpt_rd_num()<std::exp(-delt_actnow)){
            Un_jump=Un_jump*RST(epsi);
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