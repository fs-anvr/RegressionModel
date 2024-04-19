//
// Created by fsanv on 03.03.2024.
//

#ifndef ML_LINEARREGRESSION_LINEARREGRESSION_HPP
#define ML_LINEARREGRESSION_LINEARREGRESSION_HPP

#include "../Math/Approximation/LinearApproximation.hpp"
#include "../Math/Approximation/IApproximation.hpp"
#include "../Math/LossFunctions/ILossFunction.hpp"
#include "../Math/Gradient/StochasticGradient.hpp"
#include "../Math/Gradient/IGradient.hpp"
#include "../Math/Gradient/Gradient.hpp"
#include "../Math/Data/Vector.hpp"
#include "../Data/Dataset.hpp"
#include "Regression.hpp"

#include <concepts>

namespace ML::Models {

    template<class GradientType, class LossFunctionType>
            requires
            std::derived_from<LossFunctionType, Math::ILossFunction> &&
            std::derived_from<GradientType, Math::IGradient>
    class LinearRegression
            : public Regression<GradientType, Math::LinearApproximation, LossFunctionType> {};

}

#endif //ML_LINEARREGRESSION_LINEARREGRESSION_HPP
