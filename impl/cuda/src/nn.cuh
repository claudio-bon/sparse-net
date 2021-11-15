#ifndef NN_H
#define NN_H

#include "sparselayer.cuh"
#include "mat.h"
#include <vector>
#include <string>
#include <cstddef>

#define _CRT_SECURE_NO_DEPRECATE


const std::size_t R = 3;


template <typename T>
class NN {
    std::vector<SparseLayer<T> > layers;

public:
    NN();
    NN(std::size_t first_layer_size, std::size_t num_layers, bool rand=true, T min=-2.0, T max=2.0);
    NN(std::string filename);

    SparseLayer<T>& operator[](std::size_t index);
    std::size_t get_num_layers();
    float get_time();
    std::size_t get_nops();

    void forward(std::vector<T>& in_vec, std::vector<T>& out_vec, bool sequential_accumulation=true);
    std::vector<T> forward(std::vector<T>& in_vec, bool sequential_accumulation=true);
};


#endif