#pragma once
#include "complex.hpp"
#include <cassert>
#include <ostream>

namespace qcd {

// Fixed-size N×N matrix of complex numbers
template<int N>
struct Matrix {
    complex data[N*N];

    Matrix() { for(int i=0;i<N*N;i++) data[i]=complex(0.0); }

    complex& operator()(int r, int c)       { return data[r*N+c]; }
    const complex& operator()(int r, int c) const { return data[r*N+c]; }

    // Arithmetic
    Matrix operator+(const Matrix& o) const {
        Matrix res; for(int i=0;i<N*N;i++) res.data[i]=data[i]+o.data[i]; return res;
    }
    Matrix operator-(const Matrix& o) const {
        Matrix res; for(int i=0;i<N*N;i++) res.data[i]=data[i]-o.data[i]; return res;
    }
    Matrix& operator+=(const Matrix& o) {
        for(int i=0;i<N*N;i++) data[i]+=o.data[i]; return *this;
    }
    Matrix& operator-=(const Matrix& o) {
        for(int i=0;i<N*N;i++) data[i]-=o.data[i]; return *this;
    }
    Matrix operator*(double s) const {
        Matrix res; for(int i=0;i<N*N;i++) res.data[i]=data[i]*s; return res;
    }
    Matrix& operator*=(double s) {
        for(int i=0;i<N*N;i++) data[i]*=s; return *this;
    }
    Matrix operator*(complex s) const {
        Matrix res; for(int i=0;i<N*N;i++) res.data[i]=data[i]*s; return res;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& o) const {
        Matrix res;
        for(int r=0;r<N;r++)
            for(int c=0;c<N;c++) {
                complex sum;
                for(int k=0;k<N;k++) sum += (*this)(r,k)*o(k,c);
                res(r,c) = sum;
            }
        return res;
    }
    Matrix& operator*=(const Matrix& o) { *this = *this * o; return *this; }

    // Hermitian conjugate (dagger)
    Matrix dagger() const {
        Matrix res;
        for(int r=0;r<N;r++) for(int c=0;c<N;c++) res(r,c)=std::conj((*this)(c,r));
        return res;
    }

    // Trace
    complex trace() const {
        complex s;
        for(int i=0;i<N;i++) s += (*this)(i,i);
        return s;
    }

    // Real part of trace
    double re_trace() const { return trace().real(); }

    // Identity
    static Matrix identity() {
        Matrix m;
        for(int i=0;i<N;i++) m(i,i)=complex(1.0);
        return m;
    }

    // Zero
    static Matrix zero() { return Matrix(); }
};

template<int N>
inline Matrix<N> operator*(double s, const Matrix<N>& m) { return m*s; }
template<int N>
inline Matrix<N> operator*(complex s, const Matrix<N>& m) { return m*s; }

template<int N>
inline std::ostream& operator<<(std::ostream& os, const Matrix<N>& m) {
    for(int r=0;r<N;r++) {
        for(int c=0;c<N;c++) os << m(r,c) << " ";
        os << "\n";
    }
    return os;
}

using SU3 = Matrix<3>;

} // namespace qcd
