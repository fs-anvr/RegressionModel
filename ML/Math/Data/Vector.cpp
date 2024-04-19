//
// Created by fsanv on 11.03.2024.
//

#include "Vector.hpp"

#include <ranges>
#include <cmath>

namespace ML::Math {

    Vector::Vector() = default;

    Vector::~Vector() = default;

    Vector::Vector(const Vector &other) = default;

    Vector::Vector(Vector &&other) noexcept: data(std::move(other.data)) {}

    Vector::Vector(const std::vector<double> &values) : data(values) {}

    Vector &Vector::operator=(const Vector &other) = default;

    Vector &Vector::operator=(Vector &&other) noexcept {
        this->data = std::move(other.data);
        return *this;
    }

    double Vector::operator[](std::size_t index) const {
        return this->data[index];
    }

    double &Vector::operator[](std::size_t index) {
        return this->data[index];
    }

    Vector &Vector::operator+=(const Vector &other) {
        for (auto [val1, val2]: std::views::zip(this->data, other.data)) {
            val1 += val2;
        }
        return *this;
    }

    Vector Vector::operator+(const Vector &other) const {
        auto result = Vector(*this);
        result += other;
        return result;
    }

    Vector &Vector::operator-=(const Vector &other) {
        for (auto [val1, val2]: std::views::zip(this->data, other.data)) {
            val1 -= val2;
        }
        return *this;
    }

    Vector Vector::operator-(const Vector &other) const {
        auto result = Vector(*this);
        result += other;
        return result;
    }

    //Vector& Vector::operator*=(const Vector&);

    Vector &Vector::operator*=(const double &scalar) {
        for (auto &val1: this->data) {
            val1 *= scalar;
        }
        return *this;
    }

    //Vector Vector::operator*(const Vector& other) const;

    Vector Vector::operator*(const double &scalar) const {
        auto result = Vector(*this);
        result *= scalar;
        return result;
    }

    Vector &Vector::operator/=(const double &scalar) {
        for (auto &val1: this->data) {
            val1 /= scalar;
        }
        return *this;
    }

    Vector Vector::operator/(const double &scalar) const {
        auto result = Vector(*this);
        result /= scalar;
        return result;
    }

    [[nodiscard]] double Vector::norm() const {
        return std::sqrt(DotProduct(*this, *this));
    }

    std::vector<double> Vector::values() const {
        return this->data;
    }

    double DotProduct(const Vector &one, const Vector &two) {
        double sum;
        for (const auto &[val1, val2]: std::views::zip(one.values(), two.values())) {
            sum += val1 * val2;
        }
        return sum;
    }

    Vector AdamarProduct(const Vector & one, const Vector & two) {
        auto new_vector = Vector(one);
        for (auto i = 0; i < new_vector.values().size(); i++) {
            new_vector[i] *= two[i];
        }

        return new_vector;
    }

    Vector operator*(const double& scalar, const Vector& vector) {
        return vector * scalar;
    }

}
