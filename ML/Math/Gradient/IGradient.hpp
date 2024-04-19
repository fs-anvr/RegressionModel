//
// Created by fsanvr on 17.03.2024.
//

#ifndef ML_LINEARREGRESSION_IGRADIENT_HPP
#define ML_LINEARREGRESSION_IGRADIENT_HPP

#include "../../Data/Dataset.hpp"
#include "../Data/Vector.hpp"

#include "../Approximation/IApproximation.hpp"
#include "../LossFunctions/ILossFunction.hpp"

#include <concepts>

namespace ML::Math {

    class IGradient {
    public:

        IGradient() = default;
        virtual ~IGradient() = default;

        virtual Math::Vector gradient(
                const IApproximation& approximation,
                const ILossFunction& lossFunction,
                const Data::Dataset& features,
                const Math::Vector& predict,
                const Math::Vector& weights,
                double delta,
                std::size_t randomSeed
                ) = 0;
    };

}

#endif //ML_LINEARREGRESSION_IGRADIENT_HPP
