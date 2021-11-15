#ifndef SPARSELAYER_H
#define SPARSELAYER_H

#include "mat.h"
#include <vector>
#include <cstddef>


enum parallelism_t {FULL, OUTER, INNER, SEQUENTIAL};


template <typename T>
class SparseLayer {
private:
    Mat<T> weights;
    float bias;

public:
    SparseLayer(std::size_t n, std::size_t r, bool rand=true, T min = -2.0, T max = 2.0);
    void set_bias(T bias);
    void set_weight(std::size_t n, std::size_t r, T value);
    std::size_t get_shape(std::size_t dim);
    void forward(std::vector<T>& in_vec, std::vector<T>& out_vec, parallelism_t parallelism = OUTER);
};


#endif