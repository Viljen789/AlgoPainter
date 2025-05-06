#include "Fitness.h"
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <SFML/Graphics/Image.hpp>
#include <iostream>

#ifndef NO_OPENMP
#include <omp.h>
#endif

float computeFitness(const sf::Image &rendered, const sf::Image &target) {
    auto size = target.getSize();
    float sum = 0;
    for (unsigned int y = 0; y < size.y; y++) {
        for (unsigned int x = 0; x < size.x; x++) {
            auto c1 = rendered.getPixel(x, y);
            auto c2 = target.getPixel(x, y);
            float dr = static_cast<float>(c1.r) - c2.r;
            float dg = static_cast<float>(c1.g) - c2.g;
            float db = static_cast<float>(c1.b) - c2.b;
            sum -= dr * dr + dg * dg + db * db;
        }
    }
    return sum;
}

float computeFitness(const std::vector<Pixel> &rendered, const std::vector<Pixel> &target,
                     unsigned width, unsigned height) {
    float sum = 0;
    const size_t size = static_cast<size_t>(width) * height;

    if (rendered.size() != size || target.size() != size) {
        std::cerr << "Fitness size mismatch! Rendered: " << rendered.size() << ", Target: " << target.size() <<
                ", Expected: " << size << std::endl;
        return -1e12f;
    }

    const size_t CHUNK_SIZE = 4096;

#ifndef NO_OPENMP
#pragma omp parallel for reduction(-:sum) schedule(static, CHUNK_SIZE)
#endif
    for (size_t i = 0; i < size; i += CHUNK_SIZE) {
        float localSum = 0.0f;
        const size_t end = std::min(i + CHUNK_SIZE, size);

#ifdef __AVX__
#pragma omp simd reduction(-:localSum)
    for (size_t j = i; j < end; j++) {
        const Pixel &c1 = rendered[j];
        const Pixel &c2 = target[j];
        int dr = static_cast<int>(c1.r) - c2.r;
        int dg = static_cast<int>(c1.g) - c2.g;
        int db = static_cast<int>(c1.b) - c2.b;
        localSum -= static_cast<float>(dr * dr + dg * dg + db * db);
    }

#else
#ifndef NO_OPENMP
#pragma omp simd reduction(-:localSum)
#endif
        for (size_t j = i; j < end; j++) {
            const Pixel &c1 = rendered[j];
            const Pixel &c2 = target[j];
            int dr = static_cast<int>(c1.r) - c2.r;
            int dg = static_cast<int>(c1.g) - c2.g;
            int db = static_cast<int>(c1.b) - c2.b;
            localSum -= static_cast<float>(dr * dr + dg * dg + db * db);
        }
#endif

        sum += localSum;
    }

    return sum;
}
