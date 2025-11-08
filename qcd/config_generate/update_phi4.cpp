#include "lattice_phi4.h"
double rd_gen(double epsi){
    return 0;
}

Vector4i phi4_Lattice::per_add(Vector4i a,Vector4i b){
        Vector4i c =a+b;
        c[0]=(c[0]+this->Nt)%this->Nt;
        c[1]=(c[1]+this->Ns)%this->Ns;
        c[2]=(c[2]+this->Ns)%this->Ns;
        c[3]=(c[3]+this->Ns)%this->Ns;
        return c;
    };


double phi4_Lattice::neibor(Vector4i coord){
    double a=0;
    for (int i=0 ; i< 4 ; i++){
        Vector4i vecmu(0,0,0,0);
        vecmu[i]+=1;
        a += this->get_element(per_add(coord,vecmu));
    }
    return a ; 
}


void phi4_Lattice::update(Vector4i cord, double epsi,int times){
    return;
}
    

void phi4_Lattice::update_all(double epsi,int try_each){
    return;
}