//
// Created by fsanvr on 19.03.2024.
//

#ifndef ML_LINEARREGRESSION_STOCHASTICGRADIENT_HPP
#define ML_LINEARREGRESSION_STOCHASTICGRADIENT_HPP

#include "../Derivative.hpp"
#include "IGradient.hpp"

#include <functional>
#include <random>

namespace ML::Math {

    class StochasticGradient : public IGradient {
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
            auto index = randomIndex(predict.values().size(), randomSeed);
            auto row = features.row(index);
            auto y = predict.values()[index];

            auto grad = gradientOne(approximation, lossFunction, Math::Vector(row), y, weights, delta);
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

        static std::size_t randomIndex(std::size_t maxIndex, std::size_t randomSeed) {
            auto seed = randomSeed != 0 ? randomSeed : std::random_device()();
            std::mt19937 generator(seed);
            std::uniform_int_distribution<> distribution(0, (int)maxIndex);
            std::size_t randomIndex = distribution(generator);
            return randomIndex;
        }
    };

}

#endif //ML_LINEARREGRESSION_STOCHASTICGRADIENT_HPP
