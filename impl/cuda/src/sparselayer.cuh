#ifndef SPARSELAYER_H
#define SPARSELAYER_H

#include "mat.h"
#include <vector>
#include <cstddef>
#include <memory>

template <typename T>
class SparseLayer {
    Mat<T> weights;
    T bias;
    std::shared_ptr<float> time;

public:
    SparseLayer(std::size_t n, std::size_t r, bool rand=true, T min=-2.0, T max=2.0);
    void set_bias(T bias);
    void set_weight(std::size_t n, std::size_t r, T value);
    std::size_t get_shape(std::size_t dim);
    float get_time(bool reset=false);
    std::size_t get_nops();
    void forward(T* d_in_vec, T* d_out_vec, T* d_weights, bool sequential_accumulation=true);
};


#endif