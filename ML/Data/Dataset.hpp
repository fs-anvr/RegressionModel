//
// Created by fsanv on 11.03.2024.
//

#ifndef ML_LINEARREGRESSION_DATASET_HPP
#define ML_LINEARREGRESSION_DATASET_HPP

#include <string>
#include <vector>
#include <ranges>
#include <tuple>
#include <cmath>

#include "../Math/Data/Vector.hpp"

namespace ML::Data {

    struct Dataset {
    public:
        Dataset();

        ~Dataset();

        Dataset(const Dataset&);

        Dataset(Dataset&&) noexcept;

        Dataset(const std::vector<std::string>&, const std::vector<double>&, std::size_t, std::size_t) noexcept;

        Dataset(std::vector<std::string>&&, std::vector<double>&&, std::size_t, std::size_t) noexcept;

        Dataset &operator=(const Dataset&);

        Dataset &operator=(Dataset&&) noexcept;

        double operator[](std::size_t, std::size_t) const;

        double &operator[](std::size_t, std::size_t);

        std::vector<std::reference_wrapper<double>> ref_row(size_t);

        [[nodiscard]] std::vector<double> row(size_t) const;

        std::vector<std::reference_wrapper<double>> ref_column(size_t);

        [[nodiscard]] std::vector<double> column(size_t) const;

        [[nodiscard]] std::tuple<std::size_t, std::size_t> shape() const;

        [[nodiscard]] std::vector<std::vector<double>> rows() const;

        [[nodiscard]] std::vector<std::vector<double>> columns() const;

    private:
        std::vector<std::string> names;
        std::vector<double> data;
        std::size_t shape_row, shape_col;
    };
}

#endif //ML_LINEARREGRESSION_DATASET_HPP
