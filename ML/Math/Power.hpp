//
// Created by fsanv on 10.03.2024.
//

#ifndef ML_LINEARREGRESSION_POWER_HPP
#define ML_LINEARREGRESSION_POWER_HPP

#include <utility>

#include "MathConcepts.hpp"

namespace ML::Math {

    namespace detail {

        template<std::size_t i, class T>
        constexpr T _value(T value) {
            return value;
        }

        template<int n, std::size_t ... i, class T>
        constexpr T _power(T value, std::index_sequence<i...>) {
            return (_value<i>(value) * ...);
        }
    }

    template<int n, class T> requires Multipliable<T>
    constexpr T power(T value) {
        return detail::_power<n>(value, std::make_index_sequence<n>{});
    }
}

#endif //ML_LINEARREGRESSION_POWER_HPP
