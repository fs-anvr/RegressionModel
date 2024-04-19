//
// Created by fsanvr on 25.03.2024.
//

#ifndef ML_LINEARREGRESSION_STOCHASTICAVERAGEGRADIENT_HPP
#define ML_LINEARREGRESSION_STOCHASTICAVERAGEGRADIENT_HPP

#include "../Derivative.hpp"
#include "StochasticGradient.hpp"

#include <functional>
#include <random>

namespace ML::Math {

    class StochasticAverageGradient : public StochasticGradient {
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
            if (!isInit) {
                initData(randomSeed, predict.values().size(), weights.values().size());
            }

            auto index = randomIndex();

            auto grad = Math::Vector(std::vector<double>(weights.values().size()));
            std::size_t iteration = 0;
            for(auto [row, y] : std::views::zip(features.rows(), predict.values()))
            {
                if (iteration == index) {
                    gradientsByObjects[iteration] =
                            gradientOne(approximation, lossFunction, Math::Vector(row), y, weights, delta);
                }
                grad += gradientsByObjects[iteration];
                iteration++;
            }
            grad = grad / static_cast<double>(predict.values().size());

            return grad;
        }

    private:
        bool isInit;
        std::mt19937 generator;
        std::uniform_int_distribution<> distribution;
        std::vector<ML::Math::Vector> gradientsByObjects;

        void initData(std::size_t randomSeed, std::size_t object_count, std::size_t weights_count) {
            auto seed = randomSeed != 0 ? randomSeed : std::random_device()();
            this->generator = std::mt19937(seed);
            this->distribution = std::uniform_int_distribution<>(0, (int)object_count);
            this->gradientsByObjects = std::vector<Math::Vector>(object_count, Math::Vector(std::vector<double>(weights_count)));
            isInit = true;
        }

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

        std::size_t randomIndex() {
            return distribution(generator);
        }
    };

    using SAG = StochasticAverageGradient;

}

#endif //ML_LINEARREGRESSION_STOCHASTICAVERAGEGRADIENT_HPP
