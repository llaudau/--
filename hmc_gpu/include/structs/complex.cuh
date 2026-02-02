// include/models/complex.h

#pragma once


#ifndef HMC_COMPLEX_H
#define HMC_COMPLEX_H
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include "math_helper.cuh"


namespace qcdcuda{
    
  __host__ __device__ inline float conj(float x) { return x; }
  __host__ __device__ inline double conj(double x) { return x; }
  template <typename T> struct complex;





  // basic complex operators :
  //|z|
  template <typename ValueType> __host__ __device__ ValueType abs(const complex<ValueType> &z);
  // |z|^2
  template <typename ValueType> __host__ __device__ ValueType norm(const complex<ValueType> &z);


  //arithmetic operators
  // multiplication
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType> &lhs, const complex<ValueType> &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType> &lhs, const ValueType &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const ValueType &lhs, const complex<ValueType> &rhs);
  // Division
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator/(const complex<ValueType> &lhs, const complex<ValueType> &rhs);
  template <> __host__ __device__ inline complex<float> operator/(const complex<float> &lhs, const complex<float> &rhs);
  template <>
  __host__ __device__ inline complex<double> operator/(const complex<double> &lhs, const complex<double> &rhs);

  // Addition
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &lhs, const complex<ValueType> &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &lhs, const ValueType &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const ValueType &lhs, const complex<ValueType> &rhs);
  // Subtraction
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &lhs, const complex<ValueType> &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &lhs, const ValueType &rhs);
  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const ValueType &lhs, const complex<ValueType> &rhs);

  // Unary plus and minus
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &rhs);
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &rhs);







  template <typename ValueType> __host__ __device__ inline complex<ValueType> conj(const complex<ValueType> &z)
  {
    return complex<ValueType>(z.real(), -z.imag());
  }
  

  template <typename ValueType> 
  struct complex{
      typedef ValueType value_type;
      __host__ __device__ inline  ValueType real() const ;
      __host__ __device__ inline  ValueType imag() const ;
      __host__ __device__ inline void real(ValueType);
      __host__ __device__ inline void imag(ValueType);

      __host__ __device__ inline complex(const ValueType &re = ValueType(), const ValueType &im = ValueType()){
      real(re);
      imag(im);
      }
      template <class X> __host__ __device__ inline complex(const complex<X> &z)
      {
      real(z.real());
      imag(z.imag());
      }

      template <class X> __host__ __device__ inline complex(const std::complex<X> &z)
      {
      real(z.real());
      imag(z.imag());
      }
      template <typename T> __host__ __device__ inline complex<ValueType> &operator=(const complex<T> &z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

      __host__ __device__ inline complex<ValueType> &operator+=(const complex<ValueType> z)
      {
      real(real() + z.real());
      imag(imag() + z.imag());
      return *this;
      }

      __host__ __device__ inline complex<ValueType> &operator-=(const complex<ValueType> z)
      {
      real(real() - z.real());
      imag(imag() - z.imag());
      return *this;
      }

      __host__ __device__ inline complex<ValueType> &operator*=(const complex<ValueType> z)
      {
      *this = *this * z;
      return *this;
      }

      __host__ __device__ inline complex<ValueType> &operator/=(const complex<ValueType> z)
      {
      *this = *this / z;
      return *this;
      }

      __host__ __device__ inline complex<ValueType> &operator*=(const ValueType z)
      {
      real(real() * z);
      imag(imag() * z);
      return *this;
      }

  };
    template <> struct complex<float> : public float2 {
  public:
    typedef float value_type;
    complex() = default;
    constexpr complex(const float &re, const float &im = float()) : float2 {re, im} { }

    template <typename X>
    constexpr complex(const std::complex<X> &z) : float2 {static_cast<float>(z.real()), static_cast<float>(z.imag())}
    {
    }

    constexpr complex(const float2 &z) : float2(z) { }

    template <typename T> __host__ __device__ inline complex<float> &operator=(const complex<T> &z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }


    __host__ __device__ inline complex<float> &operator*=(const complex<float> &z)
    {
      *this = *this * z;
      return *this;
    }

    __host__ __device__ inline complex<float> &operator/=(const complex<float> &z)
    {
      *this = *this / z;
      return *this;
    }

    __host__ __device__ inline complex<float> &operator*=(const float &z)
    {
      *this = mul2(*this, {z, z});
      return *this;
    }

    constexpr float real() const { return x; }
    constexpr float imag() const { return y; }
    __host__ __device__ inline void real(float re) { x = re; }
    __host__ __device__ inline void imag(float im) { y = im; }

    // cast operators
    inline operator std::complex<float>() const { return std::complex<float>(real(), imag()); }
    template <typename T> inline __host__ __device__ operator complex<T>() const
    {
      return complex<T>(static_cast<T>(real()), static_cast<T>(imag()));
    }
  };

    template <> struct complex<double> : public double2 {
    public:
        typedef double value_type;
        complex() = default;
        constexpr complex(const double &re, const double &im = double()) : double2 {re, im} { }

        template <typename X>
        constexpr complex(const std::complex<X> &z) : double2 {static_cast<double>(z.real()), static_cast<double>(z.imag())}
        {
        }

        constexpr complex(const double2 &z) : double2(z) { }

        template <typename T> __host__ __device__ inline complex &operator=(const complex<T> &z)
        {
        real(z.real());
        imag(z.imag());
        return *this;
        }

        __host__ __device__ inline complex<double> &operator*=(const complex<double> &z)
        {
        *this = *this * z;
        return *this;
        }

        __host__ __device__ inline complex<double> &operator/=(const complex<double> &z)
        {
        *this = *this / z;
        return *this;
        }

        __host__ __device__ inline complex<double> &operator*=(const double &z)
        {
        this->x *= z;
        this->y *= z;
        return *this;
        }

        constexpr double real() const { return x; }
        constexpr double imag() const { return y; }
        __host__ __device__ inline void real(double re) { x = re; }
        __host__ __device__ inline void imag(double im) { y = im; }

        // cast operators
        inline operator std::complex<double>() const { return std::complex<double>(real(), imag()); }
        template <typename T> inline __host__ __device__ operator complex<T>() const
        {
        return complex<T>(static_cast<T>(real()), static_cast<T>(imag()));
        }
    };


  // Binary arithmetic operations

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    return add2(lhs, rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    return add2(lhs, {rhs, rhs});
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator+(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    return add2({lhs, lhs}, rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    return add2(lhs, -rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    return add2(lhs, {-rhs, -rhs});
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator-(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    return add2({lhs, lhs}, -rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    complex<ValueType> rtn = mul2({lhs.real(), lhs.real()}, rhs);
    return fma2({lhs.imag(), lhs.imag()}, {-rhs.imag(), rhs.real()}, rtn);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    return mul2(lhs, {rhs, rhs});
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator*(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    return mul2({lhs, lhs}, rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator/(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    const ValueType cross_norm = lhs.real() * rhs.real() + lhs.imag() * rhs.imag();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm / rhs_norm, (lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / rhs_norm);
  }



    template <typename ValueType> __host__ __device__ inline ValueType abs(const complex<ValueType> &z)
  {
    return hypot(z.real(), z.imag());
  }
  template <> __host__ __device__ inline float abs(const complex<float> &z) { return hypot(z.real(), z.imag()); }
  template <> __host__ __device__ inline double abs(const complex<double> &z) { return hypotf(z.real(), z.imag()); }
    template <typename ValueType> __host__ __device__ inline ValueType norm(const complex<ValueType> &z)
  {
    return z.real() * z.real() + z.imag() * z.imag();
  }

  template <> __host__ __device__ inline complex<float> operator/(const complex<float> &lhs, const complex<float> &rhs)
  {

    float s = abs(rhs.real()) + abs(rhs.imag());
    float oos = 1.0f / s;
    float ars = lhs.real() * oos;
    float ais = lhs.imag() * oos;
    float brs = rhs.real() * oos;
    float bis = rhs.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    return complex<float>(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
  }

  template <>
  __host__ __device__ inline complex<double> operator/(const complex<double> &lhs, const complex<double> &rhs)
  {

    double s = abs(rhs.real()) + abs(rhs.imag());
    double oos = 1.0 / s;
    double ars = lhs.real() * oos;
    double ais = lhs.imag() * oos;
    double brs = rhs.real() * oos;
    double bis = rhs.imag() * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    return complex<double>(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator/(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    return complex<ValueType>(lhs.real() / rhs, lhs.imag() / rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline complex<ValueType> operator/(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    const ValueType cross_norm = lhs * rhs.real();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm / rhs_norm, (-lhs.real() * rhs.imag()) / rhs_norm);
  }

  template <> __host__ __device__ inline complex<float> operator/(const float &lhs, const complex<float> &rhs)
  {
    return complex<float>(lhs) / rhs;
  }
  template <> __host__ __device__ inline complex<double> operator/(const double &lhs, const complex<double> &rhs)
  {
    return complex<double>(lhs) / rhs;
  }

  // Unary arithmetic operations
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator+(const complex<ValueType> &rhs)
  {
    return rhs;
  }
  template <typename ValueType> __host__ __device__ inline complex<ValueType> operator-(const complex<ValueType> &rhs)
  {
    return {-rhs.real(), -rhs.imag()};
  }

  // Equality operators
  template <typename ValueType>
  __host__ __device__ inline bool operator==(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    if (lhs.real() == rhs.real() && lhs.imag() == rhs.imag()) { return true; }
    return false;
  }

  template <typename ValueType>
  __host__ __device__ inline bool operator==(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    if (lhs == rhs.real() && rhs.imag() == 0) { return true; }
    return false;
  }
  template <typename ValueType>
  __host__ __device__ inline bool operator==(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    if (lhs.real() == rhs && lhs.imag() == 0) { return true; }
    return false;
  }

  template <typename ValueType>
  __host__ __device__ inline bool operator!=(const complex<ValueType> &lhs, const complex<ValueType> &rhs)
  {
    return !(lhs == rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline bool operator!=(const ValueType &lhs, const complex<ValueType> &rhs)
  {
    return !(lhs == rhs);
  }

  template <typename ValueType>
  __host__ __device__ inline bool operator!=(const complex<ValueType> &lhs, const ValueType &rhs)
  {
    return !(lhs == rhs);
  }

}

#endif