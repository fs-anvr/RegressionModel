//
// Created by fsanvr on 17.03.2024.
//

#ifndef ML_LINEARREGRESSION_SQUAREDERROR_HPP
#define ML_LINEARREGRESSION_SQUAREDERROR_HPP

#include "ILossFunction.hpp"

namespace ML::Math {

    class SquaredError : public ILossFunction {
    public:
        SquaredError() = default;
        ~SquaredError() override = default;

        [[nodiscard]]
        double error(double target, double value) const override {
            auto se = (target - value);
            return se * se;
        }
    };

}

#endif //ML_LINEARREGRESSION_SQUAREDERROR_HPP
