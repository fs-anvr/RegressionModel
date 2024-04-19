#include <iostream>
#include <vector>
#include <ranges>

#include "ML/Math/LossFunctions/SquaredError.hpp"
#include "ML/Regression/LinearRegression.hpp"
#include "ML/Math/Gradient/StochasticGradientWithMomentum.hpp"
#include "ML/Math/Gradient/StochasticAverageGradient.hpp"
#include "ML/Math/Gradient/StochasticGradient.hpp"
#include "ML/Math/Gradient/MiniBatchGradient.hpp"
#include "ML/Math/Gradient/Adagrad.hpp"
#include "ML/Math/Gradient/RMSPROP.hpp"
#include "ML/Math/Gradient/Gradient.hpp"
#include "ML/Math/Data/Vector.hpp"
#include "ML/Data/Dataset.hpp"

int main() {
    auto print_weights = [](const ML::Math::Vector& weights) {
        for (auto [index, weight] : std::views::enumerate(weights.values())) {
            std::cout << "w" + std::to_string(index) + " = " << weight << "  ";
        }
        std::cout << std::endl;
    };

    auto dataset = ML::Data::Dataset(
            std::vector<std::string>{"X", "Y"},
            std::vector<double>{
                    1.0, 5.0,
                    2.0, 1.0,
                    3.0, -2.0,
                    4.0, -10.0,
                    5.0, -15.0,
                    1.0, 0.0,
                    2.0, 0.0,
                    3.0, 0.0,
                    4.0, 0.0,
                    5.0, 0.0,
                    1.0, 5.0,
                    2.0, 1.0,
                    3.0, 2.0,
                    4.0, 10.0,
                    5.0, 15.0
            },
            15,
            2
    );

    auto target = ML::Math::Vector(
            std::vector<double>{17.0, 18.0, 20.0, 17.0, 17.0, 12, 17, 22, 27, 32, 17, 18, 24, 37, 47}
            );

    auto model1 = ML::Models::LinearRegression<ML::Math::Gradient, ML::Math::SquaredError>();
    auto weights1 = model1.fit(dataset, target);
    print_weights(weights1);

    auto model2 = ML::Models::LinearRegression<ML::Math::StochasticGradient, ML::Math::SquaredError>();
    auto weights2 = model2.fit(dataset, target);
    print_weights(weights2);

    auto model3 = ML::Models::LinearRegression<ML::Math::MiniBatchGradient<5>, ML::Math::SquaredError>();
    auto weights3 = model3.fit(dataset, target);
    print_weights(weights3);

    auto model4 = ML::Models::LinearRegression<ML::Math::SAG, ML::Math::SquaredError>();
    auto weights4 = model4.fit(dataset, target);
    print_weights(weights4);

    auto model5 = ML::Models::LinearRegression<ML::Math::SAGM, ML::Math::SquaredError>();
    auto weights5 = model5.fit(dataset, target);
    print_weights(weights5);

    auto model6 = ML::Models::LinearRegression<ML::Math::Adagrad, ML::Math::SquaredError>();
    auto weights6 = model6.fit(dataset, target);
    print_weights(weights6);

    auto model7 = ML::Models::LinearRegression<ML::Math::RMSPROP, ML::Math::SquaredError>();
    auto weights7 = model7.fit(dataset, target);
    print_weights(weights7);

    return 0;
}

