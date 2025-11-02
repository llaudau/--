#include "lattice.h"
#include "read.h"
#include <string>
#include "save.h"

int main(){

    Tensor<ComplexD,3> shits;
    int configs_num=2;
    std::string BASE_PATH="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/t16_s10_beta2.0/";
    for (int configs=0;configs<configs_num;configs++){
        std::string index_str=std::to_string(configs);
        LatticeData data0=read_GaugeFieldData(BASE_PATH+"field" + index_str + ".bin");
        Lattice *my_lattice=new Lattice(data0.Lt,data0.Lx,2.0);
        my_lattice->read_from_ext(data0.field); 
        Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
        if (configs==0){
            shits.resize({configs_num,shiti.dimensions()[0],shiti.dimensions()[1]});
        }
        shits.chip(configs,0)=shiti;
    }
    std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string filename="test0";
    std::string full_path = BASE_SAVE_PATH + filename + ".bin";
    save_contracted_data_binary(shits,full_path);
    return 0;
}