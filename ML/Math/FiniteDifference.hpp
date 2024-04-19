//
// Created by fsanv on 08.03.2024.
//

#ifndef ML_LINEARREGRESSION_FINITEDIFFERENCE_HPP
#define ML_LINEARREGRESSION_FINITEDIFFERENCE_HPP

#include <type_traits>
#include <functional>
#include <utility>

#include "Binomial.hpp"

namespace ML::Math {

    namespace detail {

        constexpr inline int sign(std::size_t k) {
            return k % 2 == 0 ? 1 : -1;
        }

        template<typename T, std::size_t m, std::size_t ... i>
        [[nodiscard]] constexpr double _forward_difference(
                std::function<double(T)> f,
                T x,
                T h,
                std::index_sequence<i...>) noexcept {
            return ((sign(m - i) * (double)binomial<m, i>() * f(x + i * h)) + ...);
        }

        template<typename T, std::size_t m, std::size_t ... i>
        [[nodiscard]] constexpr double _backward_difference(
                std::function<double(T)> f,
                T x,
                T h,
                std::index_sequence<i...>) noexcept {
            return ((sign(i) * (double)binomial<m, i>() * f(x - i * h)) + ...);
        }

        template<typename T, std::size_t m, std::size_t ... i>
        [[nodiscard]] constexpr double _central_difference(
                std::function<double(T)> f,
                T x,
                T h,
                std::index_sequence<i...>) noexcept {
            return ((sign(i) * (double)binomial<m, i>() * f(x + (m/2.0 - i) * h)) + ...);
        }

    }

    template<std::size_t m, typename T>
    constexpr double ForwardDifference(std::function<double(T)> f, T x, T h) {
        return detail::_forward_difference<T, m>(f, x, h, std::make_index_sequence<m + 1>{});
    }

    template<std::size_t m, typename T>
    constexpr double BackwardDifference(std::function<double(T)> f, T x, T h) {
        return detail::_backward_difference<T, m>(f, x, h, std::make_index_sequence<m + 1>{});
    }

    template<std::size_t m, typename T>
    constexpr double CentralDifference(std::function<double(T)> f, T x, T h) {
        return detail::_central_difference<T, m>(f, x, h, std::make_index_sequence<m + 1>{});
    }

}

#endif //ML_LINEARREGRESSION_FINITEDIFFERENCE_HPP
