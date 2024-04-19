cmake_minimum_required(VERSION 3.27)
project(RegressionModel)

set(CMAKE_CXX_STANDARD 23)

add_executable(RegressionModel main.cpp
        ML/Regression/LinearRegression.hpp
        ML/Math/FiniteDifference.hpp
        ML/Math/Factorial.hpp
        ML/Math/Binomial.hpp
        ML/Math/Derivative.hpp
        ML/Math/Power.hpp
        ML/Math/MathConcepts.hpp
        ML/Data/Dataset.cpp
        ML/Data/Dataset.hpp
        ML/Math/Data/Vector.cpp
        ML/Math/Data/Vector.hpp
        ML/Math/Gradient/IGradient.hpp
        ML/Math/Gradient/Gradient.hpp
        ML/Math/LossFunctions/ILossFunction.hpp
        ML/Math/Approximation/IApproximation.hpp
        ML/Math/Approximation/LinearApproximation.hpp
        ML/Math/LossFunctions/SquaredError.hpp
        ML/Regression/Regression.hpp
        ML/Math/Gradient/StochasticGradient.hpp
        ML/Math/Gradient/MiniBatchGradient.hpp
        ML/Math/Gradient/StochasticAverageGradient.hpp
        ML/Math/Gradient/StochasticGradientWithMomentum.hpp
        ML/Math/Gradient/Adagrad.hpp
        ML/Math/Gradient/RMSPROP.hpp
)
