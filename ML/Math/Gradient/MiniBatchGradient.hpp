//
// Created by fsanvr on 25.03.2024.
//

#ifndef ML_LINEARREGRESSION_MINIBATCHGRADIENT_HPP
#define ML_LINEARREGRESSION_MINIBATCHGRADIENT_HPP

#include "../Derivative.hpp"
#include "IGradient.hpp"

#include <functional>
#include <random>

namespace ML::Math {

    template<std::size_t batchSize>
    class MiniBatchGradient : public IGradient {
    public:

        MiniBatchGradient() = default;
        ~MiniBatchGradient() override = default;

        Math::Vector gradient(
                const IApproximation& approximation,
                const ILossFunction& lossFunction,
                const Data::Dataset& features,
                const Math::Vector& predict,
                const Math::Vector& weights,
                double delta,
                std::size_t randomSeed
        ) override {
            auto indexes = randomIndexes(predict.values().size(), randomSeed);

            auto grad = Math::Vector(std::vector<double>(weights.values().size()));
            for(auto index : indexes)
            {
                auto row = features.row(index);
                auto y = predict[index];
                grad += gradientOne(approximation, lossFunction, Math::Vector(row), y, weights, delta);
            }
            grad = grad / static_cast<double>(batchSize);

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

        static std::vector<std::size_t> randomIndexes(std::size_t maxIndex, std::size_t randomSeed) {
            auto seed = randomSeed != 0 ? randomSeed : std::random_device()();
            auto indexes = std::vector<std::size_t>(batchSize);
            std::mt19937 generator(seed);
            std::uniform_int_distribution<> distribution(0, (int)maxIndex);
            for (auto k = 0; k < batchSize; k++) {
                std::size_t randomIndex = distribution(generator);
                indexes.push_back(randomIndex);
            }
            return indexes;
        }

    };

}

#endif //ML_LINEARREGRESSION_MINIBATCHGRADIENT_HPP
