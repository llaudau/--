#include <iostream>
#include <format>
#include <Eigen/Dense>
#include <string>
#include <filesystem>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <chrono>

#include "lattice.h"
// #include "lattice_phi4.h"
#include "save.h"


using Eigen::MatrixXd;
using ComplexD = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;
using namespace std;
namespace fs=filesystem;

int main() {
    // Create a 8*8^3 lattice with 4 links per site
    int T=8;
    int S=4;
    int autocorelate_sample_num=10;
    const std::string BASE_PATH0 = "/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/";

    
    // for (int deltabeta=0;deltabeta<autocorelate_sample_num;deltabeta++){

    //     double betasss=5.5+(6.5-5.5)/autocorelate_sample_num *deltabeta;
    double betasss=6.01;
        std::string betassstr=std::format("{:.2f}",betasss);
        // Define the path
        std::string path_suffix = "t" + std::to_string(T) + 
                                "_s" + std::to_string(S) + 
                                "_beta"+betassstr+"atcrlt_lgth" + "/";
        
        std::string folderName = BASE_PATH0+path_suffix;

        // 1. Check if the folder exists
        if (!fs::exists(folderName)) {
            // 2. Create the directory
            if (fs::create_directory(folderName)) {
                std::cout << "Folder created successfully: " << folderName << std::endl;
            } else {
                std::cerr << "Failed to create folder." << std::endl;
            }
        } else {
            std::cout << "Folder already exists." << std::endl;
        }

        std::string BASE_PATH = BASE_PATH0 + path_suffix;
        

        // configuration number definition part
        int sampling_interval=5;
        int thermalization=8000;
        int ntraj=8000*sampling_interval;
   
        double epsi=0.10;
        int each_link_trial_num=5;


        int test_loop_number=thermalization+ntraj;
        Lattice *my_lattice=new Lattice(T,S,betasss); 
        auto start_time = Clock::now();
        Vector4i cord;

        for (int shit=0;shit<test_loop_number;shit++){
            
            if (shit>=thermalization and (shit-thermalization)%sampling_interval==0){
                // Tensor<ComplexD,RANKshit> shiti=my_lattice->Wilsonloopshit();
                // cout<<shiti<<endl;
                string index_str=to_string((shit-thermalization)/sampling_interval);
                string filename=BASE_PATH+"field" + index_str + ".bin";
                save_gauge_field_binary(my_lattice->get_gaugefield(),filename);
            }
            my_lattice->update_all(epsi,each_link_trial_num);
        }
        cout<<"success update times"<<my_lattice->successtime<<endl;
        cout<<"total try times"<<(thermalization+ntraj)*T*S*S*S*4*each_link_trial_num<<endl;
        auto end_time = Clock::now();
        auto duration_ns = end_time - start_time;
        auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
        cout << "Time taken: " << duration_us.count() << " microseconds\n";
        delete my_lattice;

    // }
    
    
    


    return 0;
}











// phi4_field

// int main(){
//     int T=4;
//     int S=4;
//     double lada=1.3282;
//     double chi=0.18169;//0.185825
//     phi4_Lattice *a= new phi4_Lattice(S,T,chi,lada);

//     // const double* phi4_data = a->get_phi4field().data();
//     // // print the first element as a quick sanity check
//     // std::cout << "phi4_data[0] = " << phi4_data[0] << std::endl;

//     // double * b= a->get_phi4field_write().data();
//     // for (int mu=0;mu<8;mu++)b[a->neighbor_1D_indices[0][mu]]=0.02;
//     // b[0]=0.01;
//     // std::cout << "phi4_data[0] = " << phi4_data[0] << std::endl;
//     // std::cout << "phi4_data_neighber = " << a->get_momelement(Vector4i(1,1,0,0)) << std::endl;
//     // example: run HMC whole process (uncomment when needed)
//     a->update_HMC_whole_process(1000,10000,10,0.1);
//     delete a;
//     return 0;
// }