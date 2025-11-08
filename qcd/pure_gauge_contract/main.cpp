#include "lattice.h"
#include "read.h"
#include <string>
#include "save.h"
using Clock = std::chrono::high_resolution_clock;


int main(){
    
<<<<<<< HEAD
    int configs_num=1000;
    std::string BASE_PATH="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/t8_s4_beta6.0_v2/";
    auto start_time = Clock::now();

    // initialize shits (bad code here: .Lx*3 is because in Wilsonloop.cpp i use for(x) for(y) for(z) )
    Tensor<ComplexD,3> shits;
    LatticeData data_init=read_GaugeFieldData(BASE_PATH+"field0.bin");
    shits.resize({configs_num,data_init.Lt,data_init.Lx});
    std::vector<double> Action_dstrb(configs_num);
    Vector4i a(0,0,0,0);
    #pragma omp parallel for
=======
    Tensor<ComplexD,3> shits;
    int configs_num=1;
    auto start_time=Clock::now();

    std::string BASE_PATH="/Users/wangkehe/Git_repository/qcd/config_generate/pure_gauge_bindata/t16_s10_beta6.0/";
>>>>>>> 368af64 (checkpy)
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
<<<<<<< HEAD
    std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string filename="test5";
=======

    std::string BASE_SAVE_PATH="/Users/wangkehe/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string filename="test1";
>>>>>>> 368af64 (checkpy)
    std::string full_path = BASE_SAVE_PATH + filename + ".bin";
    std::string fileactionname="action"+filename;
    // std::string full_action_path = BASE_SAVE_PATH + fileactionname + ".txt";
    // save_vector_to_text(Action_dstrb,full_action_path);
    save_contracted_data_binary(shits,full_path);
    return 0;
}
