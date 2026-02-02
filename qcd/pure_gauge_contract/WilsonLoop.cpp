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
    SU3Matrix O=SU3Matrix::Identity();
    Vector4i cordnow=cord;
    // if (cord==Vector4i(0,0,0,0) and dx==0){
    //     std::cout<<this->get_link(cordnow,0)*this->get_link(cordnow,0).adjoint()<<std::endl;
    // }
    
    for(int i=0;i<dx;i++){
        O=O*this->get_link(cordnow,1);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {
        //     std::cout<<"x"<<std::endl;
        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        cordnow=per_add(cordnow,dir_x);
        
        
        
    }
    for(int i=0;i<dy;i++){
        O=O*this->get_link(cordnow,2);
        cordnow=per_add(cordnow,dir_y);   
    }
    for(int i=0;i<dz;i++){
        O=O*this->get_link(cordnow,3);
        cordnow=per_add(cordnow,dir_z);
    }
    for(int i=0;i<dt;i++){
        O=O*this->get_link(cordnow,0);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {
            
        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        cordnow=per_add(cordnow,dir_t);
        
        
    }
    
    
    for(int i=0;i<dx;i++){
        cordnow=per_add(cordnow,-dir_x);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {
        //     std::cout<<"x"<<std::endl;
        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        O=O*this->get_link(cordnow,1).adjoint();
    }
    for(int i=0;i<dy;i++){
        cordnow=per_add(cordnow,-dir_y);
        O=O*this->get_link(cordnow,2).adjoint();
    }
    for(int i=0;i<dz;i++){
        cordnow=per_add(cordnow,-dir_z);
        O=O*this->get_link(cordnow,3).adjoint();
    }
    for(int i=0;i<dt;i++){
        cordnow=per_add(cordnow,-dir_t);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {

        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        O=O*this->get_link(cordnow,0).adjoint();
    }
    // if(cord==Vector4i(0,0,0,0) and dx==0) {
    //     std::cout<<O<<std::endl;
            
    //     }
    
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
    double sum_number=this->Ns*this->Ns*this->Ns*this->Nt;
    return average/(sum_number);
};

Tensor<ComplexD,RANKshit> Lattice::Wilsonloopshit(){
        Tensor<ComplexD,RANKshit> shits;

        Eigen::array<int, RANKshit> dimensions = {this->Nt,this->Ns*this->Ns};
        shits.resize(dimensions);
        shits.setZero();

        for (int t=0;t<this->Nt;t++){
            int order=0;
            for (int y=0; y<this->Ns;y++){
                for (int x=0; x<this->Ns;x++){
                    shits(t,order)=this->Wilsonloop(t,x,y,0);
                    order+=1;
                }
            }
            
            // for (int z=0; z<this->Ns;z++){
            //     shits(t,order)=this->Wilsonloop(t,this->Ns,this->Ns,z);           
            //     order+=1;
            // }
        }
        return shits;   
    }


ComplexD Lattice::Plaqutte(){
    ComplexD plaqutte=0;
    plaqutte+=this->Wilsonloop(1,1,0,0);
    // plaqutte+=this->Wilsonloop(1,0,1,0);
    // plaqutte+=this->Wilsonloop(1,0,0,1);
    plaqutte+=this->Wilsonloop(0,1,1,0);
    // plaqutte+=this->Wilsonloop(0,1,0,1);
    plaqutte+=this->Wilsonloop(0,0,1,1);
    // plaqutte/=6.0;
    plaqutte/=3.0;
    return plaqutte;
}