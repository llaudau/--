#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>
#include "lattice.h"
using Eigen::MatrixXd;
using ComplexD = std::complex<double>;
using namespace std;
int main() {
    // Create a 4x16^3 lattice with 4 links per site
    Lattice *my_lattice=new Lattice(8,10,2.0); 
    Vector4i cord;
    cord<<1,1,1,1;
    cout<<my_lattice->get_link(cord,0)<<endl;
    for (int i=0;i<10;i++){
        my_lattice->update(cord,0,0.1);
    }
    cout<<my_lattice->get_link(cord,0)<<endl;
    return 0;
}