#include "lattice.h"
const Vector4i dir_t(1,0,0,0);
const Vector4i dir_x(0,1,0,0);
const Vector4i dir_y(0,0,1,0);
const Vector4i dir_z(0,0,0,1);




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
    // for(int i=0;i<dy;i++){
    //     cordnow=per_add(cordnow,dir_y);
    //     O=O*this->get_link(cordnow,2);
    // }
    // for(int i=0;i<dz;i++){
    //     cordnow=per_add(cordnow,dir_z);
    //     O=O*this->get_link(cordnow,3);
    // }
    for(int i=0;i<dt;i++){
        O=O*this->get_link(cordnow,0);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {
            
        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        cordnow=per_add(cordnow,dir_t);
        
        
    }
    // for(int i=0;i<dz;i++){
    //     cordnow=per_add(cordnow,-dir_z);
    //     O=O*this->get_link(cordnow,3).adjoint();
    // }
    // for(int i=0;i<dy;i++){
    //     cordnow=per_add(cordnow,-dir_y);
    //     O=O*this->get_link(cordnow,2).adjoint();
    // }
    for(int i=0;i<dx;i++){
        cordnow=per_add(cordnow,-dir_x);
        // if(cord==Vector4i(0,0,0,0) and dx==0) {
        //     std::cout<<"x"<<std::endl;
        //     std::cout<<cordnow.transpose()<<std::endl;
        // }
        O=O*this->get_link(cordnow,1).adjoint();
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

        Eigen::array<int, RANKshit> dimensions = {this->Nt,this->Ns};
        shits.resize(dimensions);

        for (int t=1;t<this->Nt;t++){
            int order=0;
            for (int x=0; x<this->Ns;x++){
                shits(t,order)=this->Wilsonloop(t,x,0,0);
                order+=1;
            }
            // for (int y=0; y<this->Ns;y++){
            //     shits(t,order)=this->Wilsonloop(t,this->Ns,y,0);
            //     order+=1;
            // }
            // for (int z=0; z<this->Ns;z++){
            //     shits(t,order)=this->Wilsonloop(t,this->Ns,this->Ns,z);           
            //     order+=1;
            // }
        }
        return shits;   
    }