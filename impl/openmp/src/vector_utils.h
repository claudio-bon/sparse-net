#ifndef VECTORUTILS_H
#define VECTORUTILS_H

#include <string>
#include <vector>
#include <cstddef>

template <typename T>
std::vector<T> read_vector(std::string filename);

template <typename T>
void write_vector(std::string filename, std::vector<T> vec);

template <typename T>
void init_vector(std::vector<T>& vec, bool rand=true, T min = -2.0, T max = 2.0);

template <typename T>
std::vector<T> create_vector(std::size_t N, bool rand=true, T min = -2.0, T max = 2.0);

#endif