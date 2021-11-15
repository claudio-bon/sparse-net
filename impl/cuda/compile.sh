FOLDER="src/"
nvcc -std=c++11 ${FOLDER}main.cpp ${FOLDER}nn.cu ${FOLDER}sparselayer.cu ${FOLDER}mat.cpp ${FOLDER}sparselayer_forward_kernel.cu ${FOLDER}vector_utils.cpp -o nn_cuda