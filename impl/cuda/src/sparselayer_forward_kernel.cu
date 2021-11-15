#include "sparselayer_forward_kernel.cuh"
#include "sharedmem.cuh"
#include <cstddef>
#include <cmath>



template <typename T>
__device__ T sigmoid(T x) {
    return 1 / (1 + std::exp(-x));
}


template <typename T>
__global__ void sparselayer_forward_kernel(T* in_vec, T* out_vec, T* weights, T bias, std::size_t R, std::size_t N,
                                           std::size_t output_elements_per_block, std::size_t input_elements_per_block,
                                           bool sequential_accumulation) {

    if (threadIdx.x < output_elements_per_block * R) {
        SharedMemory<T> smem;
        T* shared_input = smem.getPointer();
        T* shared_weights = &shared_input[input_elements_per_block];

        std::size_t thread_group_id = threadIdx.x/R; //threads that compute the same output element have the same group's id
        std::size_t member_id = threadIdx.x % R; //thread's index inside the group, it lies in the interval [0, R-1]
        std::size_t block_start = blockIdx.x*output_elements_per_block; //starting output's index the block has to process (one block may compute more outputs)
        std::size_t thread_start = block_start + thread_group_id; //starting output's index the thread has to process (more threads compute a single output)

        for(std::size_t i = thread_start, j = block_start; i < N; i += gridDim.x*output_elements_per_block, j += gridDim.x*output_elements_per_block) {

            //Copy data to shared memory
            if (threadIdx.x < input_elements_per_block) {
                shared_input[threadIdx.x] = in_vec[j+threadIdx.x];
            }
            shared_weights[threadIdx.x] = weights[i * R + member_id];

            __syncthreads();

            //Multiplication
            shared_weights[threadIdx.x] = shared_input[thread_group_id + member_id] * shared_weights[threadIdx.x];

            __syncthreads();

            //Sequential Accumulation
            if (sequential_accumulation) {
                if (member_id == 0) {
                    T acc = bias;
                    for (std::size_t k = threadIdx.x; k < threadIdx.x+R; k++) {
                        acc += shared_weights[k];
                    }
                    out_vec[i] = sigmoid(acc);
                }
            }
            //Parallel Accumulation
            else {
                for (std::size_t len=R; len>=2; len = std::ceil((T)len/2)) {
                    if (member_id < len/2)
                        shared_weights[threadIdx.x] += shared_weights[threadIdx.x + (std::size_t)std::ceil((T)len/2)];
                    __syncthreads();
                }

                if (member_id == 0) {
                    out_vec[i] = sigmoid(bias + shared_weights[threadIdx.x]);
                }
            }
        }
    }
}


template __global__ void sparselayer_forward_kernel<float>(float* in_vec, float* out_vec, float* weights, float bias, std::size_t R, std::size_t N,
                                                           std::size_t output_elements_per_block, std::size_t input_elements_per_block, bool sequential_accumulation);
template __global__ void sparselayer_forward_kernel<double>(double* in_vec, double* out_vec, double* weights, double bias, std::size_t R, std::size_t N,
                                                            std::size_t output_elements_per_block, std::size_t input_elements_per_block, bool sequential_accumulation);