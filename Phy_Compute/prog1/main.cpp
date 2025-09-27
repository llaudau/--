#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "sum_ispc.h"
void serialsum(double x, int i, double result[] ){
    double partial=0.0;
    for(int j=1;j< 1000000000;j++){
        partial+=1/(j*(j+x));
    }
    result[i]=partial;
};
int main() {
    double  x1=pow(2,1.f/2);
    const int N=6;
    double x[N]={0.f,0.5f,1.f,x1,100.f,300.f};
    double result[N];
    for(int i=0;i<N;i++){
        
        auto start = std::chrono::high_resolution_clock::now();
        // serialsum(x[i],i,result);
        ispc::sum1(x[i],i,result);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

        std::cout<<"x="<<x[i]<<", psi(x) = "<<std::setprecision(10)<<result[i]<<std::endl;
    }
    return 0;}