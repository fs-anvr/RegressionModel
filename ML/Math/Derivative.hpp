//
// Created by fsanv on 10.03.2024.
//

#ifndef ML_LINEARREGRESSION_DERIVATIVE_HPP
#define ML_LINEARREGRESSION_DERIVATIVE_HPP

#include "FiniteDifference.hpp"
#include "MathConcepts.hpp"
#include "Power.hpp"

namespace ML::Math {

    template<std::size_t n = 1, typename T> requires Summable<T> && ScalarMultipliable<T> && Normalizable<T>
    constexpr double ForwardDerivative(std::function<double(T)> f, T x, T h) {
        return ML::Math::ForwardDifference<n>(f, x, h) / power<n>(h.norm());
    }

    template<std::size_t n = 1, typename T> requires Summable<T> && ScalarMultipliable<T> && Normalizable<T>
    constexpr double BackwardDerivative(std::function<double(T)> f, T x, T h) {
        return ML::Math::BackwardDifference<n>(f, x, h) / power<n>(h.norm());
    }

    template<std::size_t n = 1, typename T> requires Summable<T> && ScalarMultipliable<T> && Normalizable<T>
    constexpr double CentralDerivative(std::function<double(T)> f, T x, T h) {
        return ML::Math::CentralDifference<n>(f, x, h) / power<n>(h.norm());
    }

    template<std::size_t n = 1, typename T> requires Summable<T> && ScalarMultipliable<T> && Normalizable<T>
    constexpr double Derivative(std::function<double(T)> f, T x, T h) {
        return ML::Math::CentralDifference<n>(f, x, h) / power<n>(h.norm());
    }

}

#endif //ML_LINEARREGRESSION_DERIVATIVE_HPP
