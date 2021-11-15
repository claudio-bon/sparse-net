#include "vector_utils.h"
#include <fstream>
#include <random>
#include <stdexcept>
#include <iostream>


template <typename T>
std::vector<T> read_vector(std::string filename) {
    std::ifstream data_file;
    data_file.open(filename);

    std::vector<T> vec;
    if (data_file.is_open()) {
        std::size_t N;
        data_file >> N;
        vec.resize(N);
        for (auto elem = vec.begin(); elem!=vec.end(); ++elem) {
            data_file >> *elem;
        }
    }
    else {
        std::string error_msg = "Couldn't open ";
        error_msg += filename;
        throw std::runtime_error(error_msg);
    }

    data_file.close();
    return vec;
}

template <typename T>
void write_vector(std::string filename, std::vector<T> vec) {
    std::ofstream data_file;
    data_file.open(filename);

    if (data_file.is_open()) {
        data_file << vec.size();
        data_file << "\n";
        for (auto elem = vec.begin(); elem!=vec.end(); ++elem) {
            data_file << *elem;
            data_file << "\n";
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
void init_vector(std::vector<T>& vec, bool rand, T min, T max) {
    if (min > max) throw std::invalid_argument("min cannot be greater than max");

    if (rand) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);
        for (auto& elem : vec) {
            elem = dis(gen);
        }
    }
    else {
        T sect = (max - min) / (vec.size()-1);
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = min + i*sect;
        }
    }
}

template <typename T>
std::vector<T> create_vector(std::size_t N, bool rand, T min, T max) {
    std::vector<T> vec(N);
    init_vector(vec, rand, min, max);
    return vec;
}



template std::vector<float> read_vector(std::string filename);
template std::vector<double> read_vector(std::string filename);

template void write_vector(std::string filename, std::vector<float> vec);
template void write_vector(std::string filename, std::vector<double> vec);

template void init_vector(std::vector<float>& vec, bool rand, float min, float max);
template void init_vector(std::vector<double>& vec, bool rand, double min, double max);

template std::vector<float> create_vector(std::size_t N, bool rand, float min, float max);
template std::vector<double> create_vector(std::size_t N, bool rand, double min, double max);