//
// Created by fsanvr on 17.03.2024.
//

#ifndef ML_LINEARREGRESSION_IAPPROXIMATION_HPP
#define ML_LINEARREGRESSION_IAPPROXIMATION_HPP

#include "../Data/Vector.hpp"

namespace ML::Math {

    class IApproximation {
    public:
        IApproximation() = default;
        virtual ~IApproximation() = default;

        [[nodiscard]]
        virtual double approximation(const Math::Vector& W, const Math::Vector& X) const = 0;
    };

}

#endif //ML_LINEARREGRESSION_IAPPROXIMATION_HPP
