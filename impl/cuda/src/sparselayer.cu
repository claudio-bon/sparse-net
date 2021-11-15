#include "sparselayer.cuh"
#include "sparselayer_forward_kernel.cuh"
#include "helper_cuda.cuh"
#include <cmath>
#include <random>
#include <algorithm>
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
template<typename T>
float SparseLayer<T>::get_time(bool reset) {
    if (this->time == nullptr) throw std::logic_error("Function \"forward\" must be executed before calling \"get_time\"");
    float t = *(this->time);
    if (reset) this->time.reset();
    return t;
}
template<typename T>
std::size_t SparseLayer<T>::get_nops() {
    std::size_t N = this->get_shape(0);
    std::size_t R = this->get_shape(1);
    //Accumulation is R and not R-1 because the bias is also summed
    //3*N given by the activation function (exponential + summation + division)
    return (2*R+3)*N;
}

template <typename T>
void SparseLayer<T>::forward(T* d_in_vec, T* d_out_vec, T* d_weights, bool sequential_accumulation) {
    std::size_t N = this->get_shape(0);
    std::size_t R = this->get_shape(1);

    //Get the device properties to make sure to stay within the hardware limitations
    int device;
    cudaSafeCall( cudaGetDevice(&device) );
    cudaDeviceProp deviceProp;
    cudaSafeCall( cudaGetDeviceProperties(&deviceProp, device) );

    std::size_t output_elements_per_block = std::min(N, (std::size_t)(deviceProp.maxThreadsPerBlock / R)); //Number of output elements each block has to compute

    std::size_t input_elements_per_block = output_elements_per_block + R - 1; //Number of input elements a block has to use
    std::size_t weights_per_block = output_elements_per_block * R; //Number of weights a block has to use
    std::size_t shared_mem_size = input_elements_per_block + weights_per_block; //Number of elements the shared memory has to hold

    //The selected output elements per block's need of shared memory must not go over the max capacity
    if (shared_mem_size*sizeof(T) > deviceProp.sharedMemPerBlock) {
        std::size_t n_elements = deviceProp.sharedMemPerBlock/sizeof(T);
        output_elements_per_block = (n_elements - R + 1)/(R + 1);
        input_elements_per_block = output_elements_per_block + R - 1;
        weights_per_block = output_elements_per_block * R;
        shared_mem_size = input_elements_per_block + weights_per_block;
    }
    
    std::size_t block_dim = R*output_elements_per_block;
    std::size_t grid_dim = std::min(deviceProp.maxGridSize[0], (int)std::ceil((T)N/output_elements_per_block));

    //Transfer the weights to the device
    cudaSafeCall( cudaMemcpy(d_weights, this->weights.data(), N*R*sizeof(T), cudaMemcpyHostToDevice) );

    cudaEvent_t start, end;
    cudaSafeCall( cudaEventCreate(&start) );
    cudaSafeCall( cudaEventCreate(&end) );

    cudaSafeCall( cudaEventRecord(start) );
    sparselayer_forward_kernel<T><<< grid_dim, block_dim, shared_mem_size*sizeof(T) >>>(d_in_vec, d_out_vec, d_weights, this->bias, R, N,
                                                                                        output_elements_per_block, input_elements_per_block,
                                                                                        sequential_accumulation);
    cudaSafeCall(cudaEventRecord(end))
    
    cudaSafeCall(cudaEventSynchronize(end));
    float msecTotal;
    cudaSafeCall(cudaEventElapsedTime(&msecTotal, start, end))
    if (this->time == nullptr) this->time = std::make_shared<float>();
    *(this->time) = msecTotal/1000;
}



template class SparseLayer<float>;
template class SparseLayer<double>;