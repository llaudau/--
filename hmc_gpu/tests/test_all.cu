#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <format>
#include "lattice.cuh"
namespace fs=std::filesystem;

int main(){

    



    for (int i=0; i<7;i++){
        cudaSetDevice(0);

        cudaEvent_t start0,start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&start0);
        cudaEventCreate(&stop);
        int T=10;
        int S=8;
        int thermal=1000;
        int interval=1;
        int ntraj=10000;
    
        num_type beta=5.7+(num_type)i*0.1;

        //path structure 
        const std::string BASE_PATH0 = "/home/khw/Documents/Git_repository/hmc_gpu/results/";

        std::stringstream ss;
        ss << "t" << T << "_s" << S << "_beta" << std::fixed << std::setprecision(1) << beta << "/";
        std::string path_suffix = ss.str();
        std::string folderName = BASE_PATH0+path_suffix;


        std::string folderName0 = BASE_PATH0+path_suffix;
        if (!fs::exists(folderName0)) {
            // 2. Create the directory
            if (fs::create_directory(folderName0)) {
                std::cout << "Folder created successfully: " << folderName0 << std::endl;
            } else {
                std::cerr << "Failed to create folder." << std::endl;
            }
        } else {
            std::cout << "Folder already exists." << std::endl;
        }








        cudaEventRecord(start0);
        GaugeField Lattice=GaugeField(S,S,S,T,beta);
        cudaEventRecord(start);
        cudaEventSynchronize(start);
        std::vector<num_type> Plaq_out= Lattice.full_update(thermal,ntraj,interval,0.2,50);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        float milliseconds1 = 0;
        cudaEventElapsedTime(&milliseconds1, start, stop);
        cudaEventElapsedTime(&milliseconds, start0, start);
        printf("Construction took: %f ms\n", milliseconds);
        printf("HMC Update took: %f ms\n", milliseconds1);


        // Save the Plaq array

        std::ofstream outfile(folderName0+"plaquette_results.txt");
        if (outfile.is_open()) {
            for (int i = 0; i < Plaq_out.size(); i++) {
                outfile << std::setprecision(10) << Plaq_out[i] << "\n";
            }
            outfile.close();
            printf("Saved %zu data points to plaquette_results.txt\n", Plaq_out.size());
        } else {
            printf("Failed to open file!\n");
        }

        // Clean up
        cudaEventDestroy(start);
        cudaEventDestroy(start0);
        cudaEventDestroy(stop);
    }
    return 0;
}