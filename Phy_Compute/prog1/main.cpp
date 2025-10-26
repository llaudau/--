#include <iostream>
#include <cmath>
#include <iomanip>
void serialsum(double x, int i, double result[] ){
    double partial=0.0;
    double c=0.0;
    for(int j=1;j< 10000000;j++){
        double term=1/(j*(j+x))-c;
        double t=partial+term;
        c=(t-partial)-term;
        partial=t;
    }
    result[i]=partial+c;
};
int main() {
    double  x1=pow(2,1.f/2);
    const int N=6;
    double x[N]={0.f,0.5f,1.f,x1,100.f,300.f};
    double result[N];
    for(int i=0;i<N;i++){

        serialsum(x[i],i,result);
    
        std::cout<<"x="<<x[i]<<", psi(x) = "<<std::setprecision(10)<<result[i]<<std::endl;}
    return 0;}