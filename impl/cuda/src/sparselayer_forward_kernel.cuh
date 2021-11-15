template <typename T>
__global__ void sparselayer_forward_kernel(T* in_vec, T* out_vec, T* weights, T bias, std::size_t R, std::size_t N,
                                           std::size_t output_elements_per_block, std::size_t input_elements_per_block,
                                           bool sequential_accumulation=true);