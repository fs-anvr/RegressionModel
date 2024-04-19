//
// Created by fsanv on 08.03.2024.
//

#ifndef ML_LINEARREGRESSION_BINOMIAL_HPP
#define ML_LINEARREGRESSION_BINOMIAL_HPP

#include <type_traits>
#include "Factorial.hpp"

namespace ML::Math {

    namespace detail {
        template<std::size_t n, std::size_t k>
        struct Binomial {
            [[nodiscard]] static constexpr std::size_t binomial() noexcept {
                return factorial<n>() / (factorial<k>() * factorial<n-k>());
            }
        };

        template<std::size_t n>
        struct Binomial<n, n> {
            [[nodiscard]] static constexpr std::size_t binomial() noexcept {
                return 1;
            }
        };

        template<std::size_t n>
        struct Binomial<n, 0> {
            [[nodiscard]] static constexpr std::size_t binomial() noexcept {
                return 1;
            }
        };
    }

    template<std::size_t n, std::size_t k>
    [[nodiscard]] constexpr std::size_t binomial() noexcept {
        return detail::Binomial<n, k>::binomial();
    }

}

#endif //ML_LINEARREGRESSION_BINOMIAL_HPP
