//
// Created by fsanvr on 19.04.2024.
//

#ifndef ML_LINEARREGRESSION_ADAGRAD_HPP
#define ML_LINEARREGRESSION_ADAGRAD_HPP

#include "../Derivative.hpp"
#include "IGradient.hpp"

#include <functional>

namespace ML::Math {

    class Adagrad : public IGradient {
    public:

        Math::Vector gradient(
                const IApproximation& approximation,
                const ILossFunction& lossFunction,
                const Data::Dataset& features,
                const Math::Vector& predict,
                const Math::Vector& weights,
                double delta,
                std::size_t randomSeed
        ) override {
            auto grad = Math::Vector(std::vector<double>(weights.values().size()));
            for(auto [row, y] : std::views::zip(features.rows(), predict.values()))
            {
                grad += gradientOne(approximation, lossFunction, Math::Vector(row), y, weights, delta);
            }
            grad = grad / static_cast<double>(predict.values().size());

            return grad;
        }

    private:

        static Math::Vector gradientOne(
                const IApproximation& approximation,
                const ILossFunction& lossFunction,
                const Math::Vector& features,
                double predict,
                const Math::Vector& weights,
                double delta) {
            auto length = weights.values().size();
            auto grad = std::vector<double>(length);
            for(auto [index, dw] : std::views::enumerate(grad))
            {
                auto h = Math::Vector(std::vector<double>(length));
                h[index] = delta;
                std::function<double(Math::Vector)> f = [&predict, &approximation, &lossFunction, &features](const Math::Vector& weight) {
                    return lossFunction.error(predict, approximation.approximation(weight, features));
                };

                auto d = Math::Derivative<1>(f, weights, h);
                dw = d;
            }

            return Math::Vector(grad);
        }

    };

}

#endif //ML_LINEARREGRESSION_ADAGRAD_HPP
