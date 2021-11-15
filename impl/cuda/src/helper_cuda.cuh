#ifndef CUDA_NN_CUDA_UTILS_H
#define CUDA_NN_CUDA_UTILS_H

#include <iostream>

#define cudaSafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr<<"GPU assert: "<<cudaGetErrorString(code)<<" "<<file<<" "<<line<<"\n";
        if (abort) exit(code);
    }
}

#endif