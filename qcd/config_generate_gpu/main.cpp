// main.cpp

#include "lattice.h"
#include <iostream>
#include <chrono>


struct Params {
    int Ns = 10;
    int Nt = 10;
    double lambda=1.1;
    double beta = 0.3;
};

int main()
{   
    cudaSetDevice(0);
    auto start0 = std::chrono::high_resolution_clock::now();
    int num_sweeps=100000;

    //initialize phi4field 
    Params p;
    Lattice phi4lattice(p.Ns,p.Nt,p.lambda,p.beta);
    phi4lattice.randomize();
    phi4lattice.cuda_init(1234);
    
    auto start1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_sweeps; ++i) {
        phi4lattice.cuda_update_trajectory(0.1,10);
    }

    auto end1 = std::chrono::high_resolution_clock::now();
   

    
    phi4lattice.cuda_finalize();

    auto end0 = std::chrono::high_resolution_clock::now();

    auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    auto elapsed_ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count();
    std::cout<<phi4lattice.expdh/num_sweeps<<std::endl;
    std::cout << "Total time for " << num_sweeps << " sweeps: "
              << elapsed_ms1 << " ms\n";
    
    std::cout << "Total time for sweeps snf HOST to DEVICE trans :"
              << elapsed_ms0 -elapsed_ms1 << " ms\n";


    return 0;
}