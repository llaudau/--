#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include <chrono>
#include <vector>
#include "lattice.h"


using Eigen::MatrixXd;
using ComplexD = std::complex<double>;
using Clock = std::chrono::high_resolution_clock;
using namespace std;

int main() {
    // Create a 8*5^3 lattice with 4 links per site
    int T=8;
    int S=5;
    int each_link_trial_num=5;
    int test_loop_number=10;//500+500*10;
    Lattice *my_lattice=new Lattice(T,S,2.0); 

    auto start_time = Clock::now();
    Vector4i cord;
    // cord<<7,4,4,4;
    // int mu =0;
    // my_lattice->update(cord,mu,0.1);
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
    }    
    vector<int> v;
    for(auto element: v){
        
    }
    auto end_time = Clock::now();
    auto duration_ns = end_time - start_time;
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration_ns);
    cout << "Time taken: " << duration_us.count() << " microseconds\n";
        
        
    return 0;
}