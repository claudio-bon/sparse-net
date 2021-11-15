#include "nn.cuh"
#include "helper_cuda.cuh"
#include <fstream>
#include <stdexcept>
#include <sstream>


template <typename T>
NN<T>::NN() {}

template <typename T>
NN<T>::NN(std::size_t first_layer_size, std::size_t num_layers, bool rand, T min, T max) {
    //first_layer_size - (num_layers-1)*(R - 1) <= 0
    if (first_layer_size <= (num_layers-1)*(R - 1)) throw std::invalid_argument("Defined neural network is either too deep or the starting layer size is too small");

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
float NN<T>::get_time() {
    float total_time = 0.0f;
    for (auto& layer : this->layers) total_time += layer.get_time();
    return total_time;
}
template <typename T>
std::size_t NN<T>::get_nops() {
    std::size_t total_ops = 0;
    for (auto& layer : this->layers) total_ops += layer.get_nops();
    return total_ops;
}

template <typename T>
void NN<T>::forward(std::vector<T>& in_vec, std::vector<T>& out_vec, bool sequential_accumulation) {
    //Check if the input size is correct
    if (in_vec.size() - (R - 1) != this->layers[0].get_shape(0)) {
        std::stringstream err;
        err << "Input length should be " << this->layers[0].get_shape(0)  + R - 1 << ", but it actually is " << in_vec.size();
        throw std::invalid_argument(err.str());
    }

    T *d_in_vec, *d_out_vec, *d_weights;

    //Allocate device memory
    cudaSafeCall( cudaMalloc((void**)&d_in_vec, in_vec.size()*sizeof(T)) );
    cudaSafeCall( cudaMalloc((void**)&d_out_vec, (in_vec.size()-(R-1))*sizeof(T)) );
    cudaSafeCall( cudaMalloc((void**)&d_weights, (in_vec.size()-(R-1))*R*sizeof(T)) );

    //Transfer the input to the device
    cudaSafeCall( cudaMemcpy(d_in_vec, in_vec.data(), in_vec.size()*sizeof(T), cudaMemcpyHostToDevice) );

    for (auto& layer : this->layers) {
        layer.forward(d_in_vec, d_out_vec, d_weights, sequential_accumulation);

        //Use this Layer's output as the next Layer's input
        std::swap(d_in_vec, d_out_vec);
    }

    //Transfer the output back to the host
    std::size_t out_size = in_vec.size() - this->layers.size()*(R-1);
    if (out_vec.size() != out_size) out_vec.resize(out_size);
    cudaSafeCall( cudaMemcpy(out_vec.data(), d_in_vec, out_vec.size()*sizeof(T), cudaMemcpyDeviceToHost) );

    //Free device memory
    cudaSafeCall( cudaFree(d_in_vec) );
    cudaSafeCall( cudaFree(d_out_vec) );
    cudaSafeCall( cudaFree(d_weights) );
}
template <typename T>
std::vector<T> NN<T>::forward(std::vector<T>& in_vec, bool sequential_accumulation) {
    std::vector<T> out_vec(in_vec.size() - this->layers.size()*(R-1));
    this->forward(in_vec, out_vec, sequential_accumulation);
    return out_vec;
}


template class NN<float>;
template class NN<double>;