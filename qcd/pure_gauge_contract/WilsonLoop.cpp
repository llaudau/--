#include "lattice.h"
const Vector4i dir_t(1,0,0,0);
const Vector4i dir_x(0,1,0,0);
const Vector4i dir_y(0,0,1,0);
const Vector4i dir_z(0,0,0,1);


Vector4i Lattice::per_add(Vector4i a,Vector4i b){
        Vector4i c =a+b;
        c[0]=(c[0]+this->Nt)%this->Nt;
        c[1]=(c[1]+this->Ns)%this->Ns;
        c[2]=(c[2]+this->Ns)%this->Ns;
        c[3]=(c[3]+this->Ns)%this->Ns;
        return c;
    };


ComplexD Lattice::Wilsonloop_i(Vector4i cord,int dt,int dx, int dy,int dz){
    SU3Matrix O=this->get_link(cord,1);
    Vector4i cordnow=cord;
    
    for(int i=0;i<dx;i++){
        cordnow=per_add(cordnow,dir_x);
        O=O*this->get_link(cordnow,1);
    }
    for(int i=0;i<dy;i++){
        cordnow=per_add(cordnow,dir_y);
        O=O*this->get_link(cordnow,2);
    }
    for(int i=0;i<dz;i++){
        cordnow=per_add(cordnow,dir_z);
        O=O*this->get_link(cordnow,3);
    }
    for(int i=0;i<dt;i++){
        cordnow=per_add(cordnow,dir_t);
        O=O*this->get_link(cordnow,0);
    }
    for(int i=0;i<dz;i++){
        cordnow=per_add(cordnow,-dir_z);
        O=O*this->get_link(cordnow,3).adjoint();
    }
    for(int i=0;i<dy;i++){
        cordnow=per_add(cordnow,-dir_y);
        O=O*this->get_link(cordnow,2).adjoint();
    }
    for(int i=0;i<dx;i++){
        cordnow=per_add(cordnow,-dir_x);
        O=O*this->get_link(cordnow,1).adjoint();
    }
    for(int i=0;i<dt;i++){
        cordnow=per_add(cordnow,-dir_t);
        O=O*this->get_link(cordnow,0).adjoint();
    }
    return O.trace();
}

ComplexD Lattice::Wilsonloop(int dt,int dx, int dy,int dz){
    Vector4i srce_pt;
    ComplexD average;
    for(int t=0; t<this->Nt;t++){
        for(int x=0;x<this->Ns;x++){
            for(int y=0;y<this->Ns;y++){
                for(int z=0;z<this->Ns;z++){
                    srce_pt<<t,x,y,z;
                    average+=this->Wilsonloop_i(srce_pt,dt,dx,dy,dz);
                }
            }
        }
    }
    double lattice_number=this->Ns*this->Ns*this->Ns*this->Nt;
    return average/(lattice_number);
};