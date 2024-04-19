//
// Created by fsanv on 10.03.2024.
//

#ifndef ML_LINEARREGRESSION_MATHCONCEPTS_HPP
#define ML_LINEARREGRESSION_MATHCONCEPTS_HPP

namespace ML::Math {

    template<class T>
    concept Summable = requires(T first, T second) {
        { first + second } -> std::convertible_to<T>;
    };

    template<class T>
    concept Multipliable = requires(T first, T second) {
        { first * second } -> std::convertible_to<T>;
    };

    template<class T>
    concept ScalarMultipliable = requires(T first, double c) {
        { first * c } -> std::convertible_to<T>;
    };

    template<class T>
    concept Normalizable = requires(T value) {
        { value.norm() } -> std::convertible_to<double>;
    };

}

#endif //ML_LINEARREGRESSION_MATHCONCEPTS_HPP
