//
// Created by fsanvr on 17.03.2024.
//

#ifndef ML_LINEARREGRESSION_LINEARAPPROXIMATION_HPP
#define ML_LINEARREGRESSION_LINEARAPPROXIMATION_HPP

#include "IApproximation.hpp"

namespace ML::Math {

    class LinearApproximation : public IApproximation {
    public:
        LinearApproximation() = default;
        ~LinearApproximation() override = default;

        [[nodiscard]]
        double approximation(const Math::Vector& W, const Math::Vector& X) const override {
            auto length = X.values().size();
            double sum = 0;
            auto w_values = W.values();
            auto x_values = X.values();
            for (auto i = 0; i < length; i++) {
                sum += w_values[i] * x_values[i];
            }
            sum += w_values.back();

            return sum;
        }
    };

}

#endif //ML_LINEARREGRESSION_LINEARAPPROXIMATION_HPP
