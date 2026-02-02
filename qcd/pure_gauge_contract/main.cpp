#include "lattice.h"
#include "read.h"
#include <string>
#include <filesystem>
#include <format>
#include <iostream>
#include "save.h"
using Clock = std::chrono::high_resolution_clock;
namespace fs=std::filesystem;

// calculate wilson_loop
int main(){
    int T=8;
    int S=4;
    const std::string BASE_PATH0 = "/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/";

    int configs_num=1000;

        // // Define the path
        double betasss=6.0;
        std::string betassstr=std::format("{:.1f}",betasss);
        std::string path_suffix="t8_s4_beta"+betassstr+"/";
        std::string folderName = BASE_PATH0+path_suffix;
    


        auto start_time = Clock::now();

        // initialize shits (bad code here: .Lx*3 is because in Wilsonloop.cpp i use for(x) for(y) for(z) )
        Tensor<ComplexD,3> shits;
        LatticeData data_init=read_GaugeFieldData(folderName+"field0.bin");
        shits.resize({configs_num,data_init.Lt,data_init.Lx*data_init.Lx});
        std::vector<double> Action_dstrb(configs_num);
        std::vector<double> Plaqutte_distrb_re(configs_num);
        std::vector<double> Plaqutte_distrb_im(configs_num);
        // Vector4i a(0,0,0,0);
        #pragma omp parallel for
        for (int configs=0;configs<configs_num;configs++){
            std::string index_str=std::to_string(configs);
            LatticeData data0=read_GaugeFieldData(folderName+"field" + index_str + ".bin");
            // std::cout<<data0.Lt<<std::endl;
            // std::cout<<data0.Lx<<std::endl;
            auto my_lattice=std::make_unique<Lattice>(data0.Lt,data0.Lx,betasss);
            my_lattice->read_from_ext(data0.field);
            Plaqutte_distrb_re[configs]=my_lattice->Plaqutte().real();
            Plaqutte_distrb_im[configs]=my_lattice->Plaqutte().imag();

            Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
            // std::cout<<shiti<<std::endl;
            shits.chip(configs,0)=shiti;
        }
        auto end_time = Clock::now();
        auto duration_ns = end_time - start_time;
        auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
        std::cout << "Time taken: " << duration_us.count() << " microseconds\n";
        std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
        
        
        std::string folderName1 = BASE_SAVE_PATH+path_suffix;
        if (!fs::exists(folderName1)) {
            // 2. Create the directory
            if (fs::create_directory(folderName1)) {
                std::cout << "Folder created successfully: " << folderName1 << std::endl;
            } else {
                std::cerr << "Failed to create folder." << std::endl;
            }
        } else {
            std::cout << "Folder already exists." << std::endl;
        }
        std::string full_path = BASE_SAVE_PATH + path_suffix + "Wilsonloop.bin";
        std::string Plaqu_path_re=BASE_SAVE_PATH+path_suffix+"plaqutte_re.txt";
        std::string Plaqu_path_im=BASE_SAVE_PATH+path_suffix+"plaqutte_im.txt";

        // std::cout<<shits.chip(1,0)<<std::endl;
        save_vector_to_text(Plaqutte_distrb_re,Plaqu_path_re);
        save_vector_to_text(Plaqutte_distrb_im,Plaqu_path_im);
        save_contracted_data_binary(shits,full_path);
        // std::cout<<shits.chip(1,0)<<std::endl;
    return 0;
}


// auto correlate time calculate
// int main(){
//     int T=8;
//     int S=4;
//     const std::string BASE_PATH0 = "/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/";
//     int autocorelate_sample_num=10;
//     int configs_num=8000;
//     for (int deltabeta=0;deltabeta<autocorelate_sample_num;deltabeta+=1){
//         double betasss=5.5+(6.5-5.5)/autocorelate_sample_num *deltabeta;
//     // double betasss=6.01;
//         // double betasss=6.01;
//         std::string betassstr=std::format("{:.2f}",betasss);
//         // Define the path
//         std::string path_suffix = "t" + std::to_string(T) + 
//                                 "_s" + std::to_string(S) + 
//                                 "_beta"+betassstr+"atcrlt_lgth" + "/";
        
//         std::string folderName = BASE_PATH0+path_suffix;
//         auto start_time = Clock::now();

//         LatticeData data_init=read_GaugeFieldData(folderName+"field0.bin");
//         std::vector<double> Plaqutte_distrb_re(configs_num);
//         std::vector<double> Plaqutte_distrb_im(configs_num);
//         for (int configs=0;configs<configs_num;configs++){
//             std::string index_str=std::to_string(configs);
//             LatticeData data0=read_GaugeFieldData(folderName+"field" + index_str + ".bin");
//             auto my_lattice=std::make_unique<Lattice>(data0.Lt,data0.Lx,betasss);
//             my_lattice->read_from_ext(data0.field);
//             Plaqutte_distrb_re[configs]=my_lattice->Plaqutte().real();
//             Plaqutte_distrb_im[configs]=my_lattice->Plaqutte().imag();
//         }
//         auto end_time = Clock::now();
//         auto duration_ns = end_time - start_time;
//         auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
//         std::cout << "Time taken: " << duration_us.count() << " microseconds\n";
//         std::string BASE_SAVE_PATH="/home/khw/Documents/Git_repository/qcd/pure_gauge_contract/contracted_data/";
//         std::string folderName1 = BASE_SAVE_PATH+path_suffix;
//         if (!fs::exists(folderName1)) {
//             // 2. Create the directory
//             if (fs::create_directory(folderName1)) {
//                 std::cout << "Folder created successfully: " << folderName1 << std::endl;
//             } else {
//                 std::cerr << "Failed to create folder." << std::endl;
//             }
//         } else {
//             std::cout << "Folder already exists." << std::endl;
//         }
        
//         std::string Plaqu_path_re=BASE_SAVE_PATH+path_suffix+"plaqutte_re.txt";
//         std::string Plaqu_path_im=BASE_SAVE_PATH+path_suffix+"plaqutte_im.txt";
//         save_vector_to_text(Plaqutte_distrb_re,Plaqu_path_re);
//         save_vector_to_text(Plaqutte_distrb_im,Plaqu_path_im);
//     }
//     return 0;
// }
