#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <chrono>
#include <string>
#include "lattice.h"
#include "save.h"


using Eigen::MatrixXd;
using ComplexD = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;
using namespace std;


const std::string BASE_PATH ="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/t16_s10_beta6.0/";
int main() {
    // Create a 8*8^3 lattice with 4 links per site
    int T=16;
    int S=10;
    
    
    int thermalization=500;
    int configurations=500;
    int sampling_interval=40;
    int test_loop_number=thermalization+configurations*sampling_interval;//500+500*10;5+5*0

    double epsi=0.05;
    int each_link_trial_num=5;
    
    Lattice *my_lattice=new Lattice(T,S,6.0); 


    auto start_time = Clock::now();
    Vector4i cord;

    for (int shit=0;shit<test_loop_number;shit++){
        my_lattice->update_all(epsi,each_link_trial_num);
        if (shit>=thermalization and (shit-thermalization)%sampling_interval==0){
            string index_str=to_string((shit-thermalization)/sampling_interval);
            string filename=BASE_PATH+"field" + index_str + ".bin";
            save_gauge_field_binary(my_lattice->get_gaugefield(),filename);
        }
    }
    cout<<"success update times"<<my_lattice->successtime<<endl;
    cout<<"total try times"<<thermalization*16*1000*4*each_link_trial_num<<endl;
    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
    cout << "Time taken: " << duration_us.count() << " microseconds\n";
    Vector4i cooood;
    cooood<<0,0,0,0;
    cout <<my_lattice->get_link(cooood,0)<<endl;
    delete my_lattice;
    return 0;
}