//
// Created by fsanv on 08.03.2024.
//

#ifndef ML_LINEARREGRESSION_FACTORIAL_HPP
#define ML_LINEARREGRESSION_FACTORIAL_HPP

#include <type_traits>
#include <utility>

namespace ML::Math {

    namespace detail {

        template<std::size_t ... Is>
        [[nodiscard]] constexpr std::size_t _factorial(std::index_sequence<Is...>) noexcept
        {
            return ((Is + 1) * ...);
        }

    }

    template<std::size_t N>
    [[nodiscard]] constexpr std::size_t factorial() noexcept
    {
        return detail::_factorial(std::make_index_sequence<N>{});
    }

    template<>
    [[nodiscard]] constexpr std::size_t factorial<0>() noexcept
    {
        return 1;
    }

}

#endif //ML_LINEARREGRESSION_FACTORIAL_HPP
