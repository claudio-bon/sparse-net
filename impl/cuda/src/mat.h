#ifndef MAT_H
#define MAT_H

#include <vector>
#include <cstddef>

template <typename T>
class Mat {
    std::vector<T> mat;
    std::vector<std::size_t> size;

public:
    Mat();
    Mat(std::size_t n, std::size_t r, bool rand=true, T min=-2.0, T max=2.0);
    void resize(std::size_t n, std::size_t r);
    T& operator()(std::size_t n, std::size_t r);
    const T operator()(std::size_t n, std::size_t r) const;
    std::size_t get_size(std::size_t dim);
    T* data();
};


#endif