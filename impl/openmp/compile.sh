FOLDER="src/"
g++ --std=c++11 -fopenmp ${FOLDER}main.cpp ${FOLDER}nn.cpp ${FOLDER}sparselayer.cpp ${FOLDER}mat.cpp ${FOLDER}vector_utils.cpp -o nn_openmp