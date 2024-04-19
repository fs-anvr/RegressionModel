//
// Created by fsanv on 11.03.2024.
//

#ifndef ML_LINEARREGRESSION_VECTOR_HPP
#define ML_LINEARREGRESSION_VECTOR_HPP

#include <type_traits>
#include <vector>

namespace ML::Math {

    struct Vector {
    public:
        Vector();

        ~Vector();

        Vector(const Vector &);

        Vector(Vector &&) noexcept;

        explicit Vector(const std::vector<double>&);

        Vector &operator=(const Vector &);

        Vector &operator=(Vector &&) noexcept;

        double operator[](std::size_t index) const;

        double &operator[](std::size_t index);

        Vector &operator+=(const Vector &);

        Vector operator+(const Vector &) const;

        Vector &operator-=(const Vector &);

        Vector operator-(const Vector &) const;

        //Vector& operator*=(const Vector&);
        Vector &operator*=(const double &);

        //Vector operator*(const Vector&) const;
        Vector operator*(const double &) const;

        Vector &operator/=(const double &);

        Vector operator/(const double &) const;

        [[nodiscard]]
        double norm() const;

        [[nodiscard]]
        std::vector<double> values() const;

    private:
        std::vector<double> data;
    };

    double DotProduct(const Vector &, const Vector &);

    Vector AdamarProduct(const Vector &, const Vector &);

    Vector operator*(const double&, const Vector&);

}

#endif //ML_LINEARREGRESSION_VECTOR_HPP
