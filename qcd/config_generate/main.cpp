#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <chrono>
#include "lattice.h"
#include "save.h"


using Eigen::MatrixXd;
using ComplexD = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;
using namespace std;


const std::string BASE_PATH ="/home/khw/Documents/Git_repository/qcd/config_generate/pure_gauge_bindata/";
int main() {
    // Create a 8*5^3 lattice with 4 links per site
    int T=8;
    int S=8;
    int each_link_trial_num=5;
    int test_loop_number=500+500*10;
    Lattice *my_lattice=new Lattice(T,S,2.0); 


    auto start_time = Clock::now();
    Vector4i cord;

    for (int shit=0;shit<test_loop_number;shit++){
        for (int t=0;t<T;t++){
            for (int x=0; x<S;x++){
                for (int y=0; y<S;y++){
                    for (int z=0; z<S;z++){
                        for (int mu=0;mu<4;mu++){
                            cord<<t,x,y,z;
                            for (int i=0;i<each_link_trial_num;i++){
                            my_lattice->update(cord,mu,0.1);
                        }
                        }
                    }
                }
            }
        }
        if (shit>=499 and (shit-499)%10==0){
            string index_str=to_string((shit-499)/10);
            string filename=BASE_PATH+"field" + index_str + ".bin";
            save_gauge_field_binary(my_lattice->get_gaugefield(),filename);
        }
    }    

    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::milliseconds>(duration_ns);
    cout << "Time taken: " << duration_us.count() << " microseconds\n";
        
    delete my_lattice;
    return 0;
}