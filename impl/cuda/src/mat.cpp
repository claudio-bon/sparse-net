#include "mat.h"
#include "vector_utils.h"
#include <stdexcept>


template <typename T>
Mat<T>::Mat() {}
template <typename T>
Mat<T>::Mat(std::size_t n, std::size_t r, bool rand, T min, T max) : mat(n*r), size{n,r} {
    init_vector<T>(this->mat, rand, min, max);
}

template <typename T>
void Mat<T>::resize(std::size_t n, std::size_t r) {
    this->size = std::vector<std::size_t>{n, r};
    this->mat.resize(n * r);
}

template <typename T>
T& Mat<T>::operator()(std::size_t n, std::size_t r) {
    if (n >= this->size[0] || r >= this->size[1])
        throw std::out_of_range("Indices out of range");
    return this->mat[n * this->size[1] + r];
}
template <typename T>
const T Mat<T>::operator()(std::size_t n, std::size_t r) const {
    if (n >= this->size[0] || r >= this->size[1])
        throw std::out_of_range("Indices out of range");
    return this->mat[n * this->size[1] + r];
}

template <typename T>
std::size_t Mat<T>::get_size(std::size_t dim) {
    if (dim >= this->size.size())
        throw std::out_of_range("Index out of range");
    return this->size[dim];
}

template <typename T>
T* Mat<T>::data() {
    return this->mat.data();
}



template class Mat<float>;
template class Mat<double>;