#include <iostream>
#include <complex>
#include "complex.cuh"
#include "math_helper.cuh"
#include <iostream>

int main() {
    using namespace qcdcuda;

    // Test complex<double>
    complex<double> a(3.0, 4.0);
    complex<double> b(1.0, 2.0);

    // Test Addition (uses math_helper and complex logic)
    complex<double> c = a + b;
    
    // Test Multiplication (The complex logic: (3+4i)*(1+2i) = -5 + 10i)
    complex<double> d = a * b;

    std::cout << "Testing Complex Math:" << std::endl;
    std::cout << "a + b = (" << c.real() << ", " << c.imag() << ")" << std::endl;
    std::cout << "a * b = (" << d.real() << ", " << d.imag() << ")" << std::endl;

    // Verify against manual math
    if (d.real() == -5.0 && d.imag() == 10.0) {
        std::cout << "SUCCESS: Complex multiplication is correct!" << std::endl;
    } else {
        std::cout << "FAIL: Check your complex operator* logic." << std::endl;
    }

    return 0;
}