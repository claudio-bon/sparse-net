#include "sparselayer.h"
#include <cmath>
#include <random>
#include <omp.h>
#include <stdexcept>


template <typename T>
SparseLayer<T>::SparseLayer(std::size_t n, std::size_t r, bool rand, T min, T max): weights(n,r,rand,min,max) {
    if (rand) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(min, max);
            this->bias = dis(gen);
    }
    else this->bias = 1.0;
}

template <typename T>
void SparseLayer<T>::set_bias(T bias) {
    this->bias = bias;
}
template <typename T>
void SparseLayer<T>::set_weight(std::size_t n, std::size_t r, T value) {
    this->weights(n, r) = value;
}
template<typename T>
std::size_t SparseLayer<T>::get_shape(std::size_t dim) {
    return this->weights.get_size(dim);
}


template <typename T>
T sigmoid(T x) { return 1 / (1 + std::exp(-x)); }

template <typename T>
void SparseLayer<T>::forward(std::vector<T>& in_vec, std::vector<T>& out_vec, parallelism_t parallelism) {
    if (out_vec.size() != this->weights.get_size(0)) out_vec.resize(this->weights.get_size(0));

    //Full parallelism
    if (parallelism == FULL) {
        int i;
#pragma omp parallel for shared(in_vec, out_vec) schedule(static)
        for (i = 0; i < this->weights.get_size(0); i++) {
            int r;
            T result = this->bias;
#pragma omp parallel for reduction(+:result) shared(in_vec, out_vec) schedule(static)
            for (r = 0; r < this->weights.get_size(1); r++) {
                result += in_vec[i + r] * this->weights(i, r);
            }

            out_vec[i] = sigmoid(result);
        }
    }
    //Outer loop parallelism
    else if (parallelism == OUTER) {
        int i;
#pragma omp parallel for shared(in_vec, out_vec) schedule(static)
        for (i = 0; i < this->weights.get_size(0); i++) {

            T result = this->bias;
            for (int r = 0; r < this->weights.get_size(1); r++) {
                result += in_vec[i + r] * this->weights(i, r);
            }

            out_vec[i] = sigmoid(result);
        }
    }
    //Inner loop parallelism
    else if (parallelism == INNER) {
        for (int i = 0; i < this->weights.get_size(0); i++) {
            int r;
            T result = this->bias;
#pragma omp parallel for reduction(+:result) shared(in_vec, out_vec) schedule(static)
            for (r = 0; r < this->weights.get_size(1); r++) {
                result += in_vec[i + r] * this->weights(i, r);
            }

            out_vec[i] = sigmoid(result);
        }
    }
    //Sequential
    else if (parallelism == SEQUENTIAL) {
        int i;
        for (i = 0; i < this->weights.get_size(0); i++) {

            T result = this->bias;
            for (int r = 0; r < this->weights.get_size(1); r++) {
                result += in_vec[i + r] * this->weights(i, r);
            }

            out_vec[i] = sigmoid(result);
        }
    }
}


template class SparseLayer<float>;
template class SparseLayer<double>;