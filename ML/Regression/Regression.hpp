//
// Created by fsanvr on 19.03.2024.
//

#ifndef ML_LINEARREGRESSION_REGRESSION_HPP
#define ML_LINEARREGRESSION_REGRESSION_HPP

#include "../Math/Data/Vector.hpp"
#include "../Data/Dataset.hpp"

#include "../Math/Gradient/IGradient.hpp"
#include "../Math/Gradient/Gradient.hpp"
#include "../Math/Approximation/IApproximation.hpp"
#include "../Math/Gradient/StochasticGradientWithMomentum.hpp"
#include "../Math/Gradient/StochasticAverageGradient.hpp"
#include "../Math/Gradient/StochasticGradient.hpp"
#include "../Math/Gradient/Adagrad.hpp"
#include "../Math/Gradient/RMSPROP.hpp"
#include "../Math/LossFunctions/ILossFunction.hpp"

#include <concepts>


namespace ML::Models {

    template<class GradientType, class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction> &&
    std::derived_from<GradientType, Math::IGradient>
    class Regression {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new GradientType();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            for (std::size_t k = 0;; k++)
            {
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                w -= learningRate * grad;

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;
    };


    template<class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction>
    class Regression<Math::StochasticGradient, ApproximationType, LossFunctionType> {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new Math::StochasticGradient();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            for (std::size_t k = 0;; k++)
            {
                auto realLearningRate = this->learningRate(learningRate, k);
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                w -= realLearningRate * grad;

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;

        double learningRate(double lr, std::size_t iteration) {
            return iteration == 0 ? lr : lr / static_cast<double>(iteration);
        }
    };


    template<class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction>
    class Regression<Math::StochasticAverageGradient, ApproximationType, LossFunctionType> {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new Math::StochasticAverageGradient();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            for (std::size_t k = 0;; k++)
            {
                auto realLearningRate = this->learningRate(learningRate, k);
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                w -= realLearningRate * grad;

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;

        double learningRate(double lr, std::size_t iteration) {
            return iteration == 0 ? lr : lr / static_cast<double>(iteration);
        }
    };


    template<class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction>
    class Regression<Math::StochasticGradientWithMomentum, ApproximationType, LossFunctionType> {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new Math::StochasticGradientWithMomentum();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0,
                double alpha = 0.5
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            auto h = Math::Vector(std::vector<double>(m+1)); // momentum
            for (std::size_t k = 0;; k++)
            {
                auto realLearningRate = this->learningRate(learningRate, k);
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                h = alpha * h + realLearningRate * grad;
                w -= h;

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;

        double learningRate(double lr, std::size_t iteration) {
            return iteration == 0 ? lr : lr / static_cast<double>(iteration);
        }
    };


    template<class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction>
    class Regression<Math::Adagrad, ApproximationType, LossFunctionType> {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new Math::Adagrad();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            auto G = Math::Vector(std::vector<double>(m+1));
            for (std::size_t k = 0;; k++)
            {
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                G += Math::AdamarProduct(grad, grad);
                auto weighted_lr = Math::Vector(std::vector<double>(m+1));
                for (auto i = 0; i < weighted_lr.values().size(); i++) {
                    weighted_lr[i] = learningRate / std::sqrt(G[i] + epsilon);
                }
                w -= Math::AdamarProduct(weighted_lr, grad);

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;
    };


    template<class ApproximationType, class LossFunctionType>
    requires
    std::derived_from<ApproximationType, Math::IApproximation> &&
    std::derived_from<LossFunctionType, Math::ILossFunction>
    class Regression<Math::RMSPROP, ApproximationType, LossFunctionType> {
    public:

        Regression() {
            approximation = new ApproximationType();
            lossFunction = new LossFunctionType();
            gradient = new Math::RMSPROP();
        }

        virtual ~Regression() {
            delete approximation;
            delete lossFunction;
            delete gradient;
        }

        Math::Vector fit(
                const Data::Dataset& sample,
                const Math::Vector& target,
                double learningRate = 0.01,
                std::size_t iterations = 50000,
                double epsilon = 0.000000001,
                std::size_t randomSeed = 0,
                double alpha = 0.5
        ) {
            auto [n, m] = sample.shape();
            auto w = Math::Vector(std::vector<double>(m+1, 1));
            auto G = Math::Vector(std::vector<double>(m+1));
            for (std::size_t k = 0;; k++)
            {
                auto grad = gradient->gradient(*approximation, *lossFunction, sample, target, w, 0.01, randomSeed);
                G = alpha * G + (1 - alpha) * Math::AdamarProduct(grad, grad);
                auto weighted_lr = Math::Vector(std::vector<double>(m+1));

                for (auto i = 0; i < weighted_lr.values().size(); i++) {
                    weighted_lr[i] = learningRate / std::sqrt(G[i] + epsilon);
                }

                auto temp = Math::AdamarProduct(weighted_lr, grad);
                w -= temp;

                if (k == iterations || grad.norm() < epsilon) {
                    break;
                }
            }
            return w;
        }

    private:
        Math::IApproximation* approximation;
        Math::ILossFunction* lossFunction;
        Math::IGradient* gradient;
    };

}

#endif //ML_LINEARREGRESSION_REGRESSION_HPP
