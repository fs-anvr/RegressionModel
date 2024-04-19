//
// Created by fsanv on 03.03.2024.
//

#ifndef ML_DATA_DATASET_CPP
#define ML_DATA_DATASET_CPP

#include "Dataset.hpp"

namespace ML::Data {

    Dataset::Dataset() = default;

    Dataset::~Dataset() = default;

    Dataset::Dataset(const Dataset& other) = default;

    Dataset::Dataset(Dataset&& other) noexcept :
        names(std::move(other.names)),
        data(std::move(other.data)),
        shape_row(other.shape_row),
        shape_col(other.shape_col) { }

    Dataset::Dataset(
            const std::vector<std::string>& names,
            const std::vector<double>& data,
            std::size_t shape_row,
            std::size_t shape_col
            ) noexcept :
        names(names), data(data), shape_row(shape_row), shape_col(shape_col) {}

    Dataset::Dataset(
            std::vector<std::string>&& names,
            std::vector<double>&& data,
            std::size_t shape_row,
            std::size_t shape_col
            ) noexcept :
            names(std::move(names)), data(std::move(data)), shape_row(shape_row), shape_col(shape_col) {}

    Dataset &Dataset::operator=(const Dataset& other) {
        this->names = other.names;
        this->data = other.data;
        this->shape_row = other.shape_row;
        this->shape_col = other.shape_col;

        return *this;
    }

    Dataset &Dataset::operator=(Dataset&& other) noexcept {
        this->names = std::move(other.names);
        this->data = std::move(other.data);
        this->shape_row = other.shape_row;
        this->shape_col = other.shape_col;

        return *this;
    }

    double Dataset::operator[](std::size_t row, std::size_t column) const {
        return this->data[shape_col*row + column];
    }

    double &Dataset::operator[](std::size_t row, std::size_t column) {
        return this->data[shape_col*row + column];
    }

    std::vector<std::reference_wrapper<double>> Dataset::ref_row(size_t index) {
        auto start = data.begin() + shape_col * index;
        return {start, start + shape_col};
    }

    std::vector<double> Dataset::row(size_t index) const {
        auto start = data.begin() + shape_col * index;
        return {start, start + shape_col};
    }

    std::vector<std::reference_wrapper<double>> Dataset::ref_column(size_t index) {
        std::vector<std::reference_wrapper<double>> result;
        result.reserve(shape_row);
        for (auto k = 0; k < shape_row; k++) {
            result.push_back((*this)[k, index]);
        }
        return result;
    }

    std::vector<double> Dataset::column(size_t index) const {
        std::vector<double> result;
        result.reserve(shape_row);
        for (auto k = 0; k < shape_row; k++) {
            result.push_back((*this)[k, index]);
        }
        return result;
    }

    [[nodiscard]]
    std::tuple<std::size_t, std::size_t> Dataset::shape() const {
        return std::tuple{shape_row, shape_col};
    }

    [[nodiscard]]
    std::vector<std::vector<double>> Dataset::rows() const {
        std::vector<std::vector<double>> result;
        result.reserve(shape_row);
        for (auto k = 0; k < shape_row; k++) {
            result.push_back(this->row(k));
        }
        return result;
    }

    [[nodiscard]]
    std::vector<std::vector<double>> Dataset::columns() const {
        std::vector<std::vector<double>> result;
        result.reserve(shape_col);
        for (auto k = 0; k < shape_col; k++) {
            result.push_back(this->column(k));
        }
        return result;
    }
}

#endif //ML_DATA_DATASET_CPP