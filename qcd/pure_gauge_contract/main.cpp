#include "lattice.h"
#include "read.h"
#include <string>
#include "save.h"
using Clock = std::chrono::high_resolution_clock;


int main(){
    
    Tensor<ComplexD,3> shits;
    int configs_num=1;
    auto start_time=Clock::now();

    std::string BASE_PATH="/Users/wangkehe/Git_repository/qcd/config_generate/pure_gauge_bindata/t16_s10_beta6.0/";
    for (int configs=0;configs<configs_num;configs++){
        std::string index_str=std::to_string(configs);
        LatticeData data0=read_GaugeFieldData(BASE_PATH+"field" + index_str + ".bin");
        Lattice *my_lattice=new Lattice(data0.Lt,data0.Lx,6.0);
        my_lattice->read_from_ext(data0.field);
        // my_lattice->Action_all();
        // Action_dstrb[configs]=my_lattice->action;
        Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
        shits.chip(configs,0)=shiti;
        delete my_lattice;
    }
    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
    std::cout << "Time taken: " << duration_us.count() << " microseconds\n";

    std::string BASE_SAVE_PATH="/Users/wangkehe/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string filename="test1";
    std::string full_path = BASE_SAVE_PATH + filename + ".bin";
    std::string fileactionname="action"+filename;
    // std::string full_action_path = BASE_SAVE_PATH + fileactionname + ".txt";
    // save_vector_to_text(Action_dstrb,full_action_path);
    save_contracted_data_binary(shits,full_path);
    return 0;
}
