#include "lattice.h"
#include "read.h"
#include <string>
#include "save.h"
using Clock = std::chrono::high_resolution_clock;


int main(){
    
    std::string configname="t8_s4_beta6.0_v2/";

    int configs_num=1000;

    std::string BASE_PATH="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/"+configname;
    auto start_time = Clock::now();

    // initialize shits (bad code here: .Lx*3 is because in Wilsonloop.cpp i use for(x) for(y) for(z) )
    Tensor<ComplexD,3> shits;
    LatticeData data_init=read_GaugeFieldData(BASE_PATH+"field0.bin");
    shits.resize({configs_num,data_init.Lt,data_init.Lx});
    // std::vector<double> Action_dstrb(configs_num);
    std::vector<double> Plaqutte_distrb_re(configs_num);
    std::vector<double> Plaqutte_distrb_im(configs_num);
    Vector4i a(0,0,0,0);
    #pragma omp parallel for
    for (int configs=0;configs<configs_num;configs++){
        std::string index_str=std::to_string(configs);
        LatticeData data0=read_GaugeFieldData(BASE_PATH+"field" + index_str + ".bin");
        Lattice *my_lattice=new Lattice(data0.Lt,data0.Lx,6.0);
        my_lattice->read_from_ext(data0.field);


        // my_lattice->Action_all();
        // Action_dstrb[configs]=my_lattice->action;

        Plaqutte_distrb_re[configs]=my_lattice->Plaqutte().real();
        Plaqutte_distrb_im[configs]=my_lattice->Plaqutte().imag();

        Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
        // std::cout<<shiti<<std::endl;
        shits.chip(configs,0)=shiti;
        delete my_lattice;
    }
    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
    std::cout << "Time taken: " << duration_us.count() << " microseconds\n";
    std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
    std::string full_path = BASE_SAVE_PATH + configname + "Wilsonloop.bin";
    std::string Plaqu_path_re=BASE_SAVE_PATH+configname+"plaqutte_re.txt";
    std::string Plaqu_path_im=BASE_SAVE_PATH+configname+"plaqutte_im.txt";

    // std::cout<<shits.chip(1,0)<<std::endl;

    // std::string fileactionname="action"+filename;
    // std::string full_action_path = BASE_SAVE_PATH + fileactionname + ".txt";
    // save_vector_to_text(Action_dstrb,full_action_path);

    save_vector_to_text(Plaqutte_distrb_re,Plaqu_path_re);
    save_vector_to_text(Plaqutte_distrb_im,Plaqu_path_im);
    save_contracted_data_binary(shits,full_path);
    // std::cout<<shits.chip(1,0)<<std::endl;
    return 0;
}
