#include "lattice.h"
#include "read.h"
#include <string>
#include "save.h"
using Clock = std::chrono::high_resolution_clock;


int main(){

    
    int configs_num=500;
    std::string BASE_PATH="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/t16_s10_beta6.0/";
    auto start_time = Clock::now();

    // initialize shits (bad code here: .Lx*3 is because in Wilsonloop.cpp i use for(x) for(y) for(z) )
    Tensor<ComplexD,3> shits;
    LatticeData data_init=read_GaugeFieldData(BASE_PATH+"field0.bin");
    shits.resize({configs_num,data_init.Lt,data_init.Lx});

    #pragma omp parallel for
    for (int configs=0;configs<configs_num;configs++){
        std::string index_str=std::to_string(configs);
        LatticeData data0=read_GaugeFieldData(BASE_PATH+"field" + index_str + ".bin");
        Lattice *my_lattice=new Lattice(data0.Lt,data0.Lx,6.0);
        my_lattice->read_from_ext(data0.field); 
        Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
        shits.chip(configs,0)=shiti;
        delete my_lattice;
    }
    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
    std::cout << "Time taken: " << duration_us.count() << " microseconds\n";
    std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string filename="test1";
    std::string full_path = BASE_SAVE_PATH + filename + ".bin";
    save_contracted_data_binary(shits,full_path);
    return 0;
}