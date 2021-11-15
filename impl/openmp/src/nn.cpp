#include "nn.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <sstream>


template <typename T>
NN<T>::NN() {}

template <typename T>
NN<T>::NN(std::size_t first_layer_size, std::size_t num_layers, bool rand, T min, T max) {
    if (first_layer_size <= (num_layers-1) * (R - 1)) throw std::invalid_argument("Defined neural network is either too deep or the starting layer size is too small");

    for (std::size_t i = 1; i < num_layers; i++) {
        this->layers.push_back(SparseLayer<T>(first_layer_size - i * (R - 1), R, rand, min, max));
    }
}

template <typename T>
NN<T>::NN(std::string filename) {
    std::ifstream data_file;
    data_file.open(filename);

    if (data_file.is_open()) {
        std::size_t n, r, num_layers;
        T bias, weight;

        data_file >> num_layers;
        for (std::size_t i = 1; i < num_layers; i++) {
            data_file >> n;
            data_file >> r;
            if (r != R) {
                throw std::length_error("Weights dimension R does not match with the one defined in the file");
            }

            SparseLayer<T> layer(n, r);

            data_file >> bias;
            layer.set_bias(bias);
            for (std::size_t j = 0; j < n; j++) {
                for (std::size_t k = 0; k < r; k++) {
                    data_file >> weight;
                    layer.set_weight(j, k, weight);
                }
            }
            this->layers.push_back(layer);
        }
    }
    else {
        std::string error_msg = "Couldn't open ";
        error_msg += filename;
        throw std::runtime_error(error_msg);
    }

    data_file.close();
}

template <typename T>
SparseLayer<T>& NN<T>::operator[](std::size_t index) {
    return this->layers[index];
}

template <typename T>
std::size_t NN<T>::get_num_layers() {
    return this->layers.size();
}

template <typename T>
void NN<T>::forward(std::vector<T>& in_vec, std::vector<T>& out_vec, parallelism_t parallelism) {
    //Check if the input size is correct
    if (in_vec.size() - (R - 1) != this->layers[0].get_shape(0)) {
        std::stringstream err;
        err << "Input length should be " << this->layers[0].get_shape(0)  + R - 1 << ", but it actually is " << in_vec.size();
        throw std::invalid_argument(err.str());
    }

    copy(in_vec.begin(), in_vec.end(), back_inserter(out_vec));
    std::vector<T> tmp;

    for (auto& layer : this->layers) {
        tmp.resize(layer.get_shape(0));

        layer.forward(out_vec, tmp, parallelism);

        //Use this Layer's output as the next Layer's input
        std::swap(out_vec, tmp);
    }
}
template <typename T>
std::vector<T> NN<T>::forward(std::vector<T>& in_vec, parallelism_t parallelism) {
    std::vector<T> out_vec;
    this->forward(in_vec, out_vec, parallelism);
    return out_vec;
}


template class NN<float>;
template class NN<double>;