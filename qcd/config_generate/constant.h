#include <Eigen/Dense>
#include <complex>
using ComplexD = std::complex<double>;
using Matrix2c = Eigen::Matrix<ComplexD, 2, 2>;




inline static const Matrix2c Sigma1 = (Matrix2c() << 
    ComplexD(0, 0), ComplexD(1, 0),
    ComplexD(1, 0), ComplexD(0, 0)
).finished();

inline static const Matrix2c Sigma2 = (Matrix2c() << 
    ComplexD(0, 0), ComplexD(0, -1),
    ComplexD(0, 1), ComplexD(0, 0)
).finished();

inline static const Matrix2c Sigma3 = (Matrix2c() << 
    ComplexD(1, 0), ComplexD(0, 0),
    ComplexD(0, 0), ComplexD(-1, 0)
).finished();
