//
// Created by fsanvr on 17.03.2024.
//

#ifndef ML_LINEARREGRESSION_ILOSSFUNCTION_HPP
#define ML_LINEARREGRESSION_ILOSSFUNCTION_HPP

namespace ML::Math {

    class ILossFunction {
    public:
        ILossFunction() = default;
        virtual ~ILossFunction() = default;

        [[nodiscard]]
        virtual double error(double target, double value) const = 0;
    };

}

#endif //ML_LINEARREGRESSION_ILOSSFUNCTION_HPP
