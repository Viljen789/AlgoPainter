//
// Created by vilje on 03/05/2025.
//

#include "Fitness.h"
#include <immintrin.h> // For SIMD intrinsics (x86 specific)
#include <vector>
#include <algorithm> // For std::min
#include <SFML/Graphics/Image.hpp> // Include specific header for sf::Image
#include <iostream>

#ifndef NO_OPENMP
#include <omp.h> // Include OpenMP
#endif


// Original function for backward compatibility (uses sf::Image)
float computeFitness(const sf::Image &rendered, const sf::Image &target) {
    auto size = target.getSize(); // size is sf::Vector2u
    float sum = 0;
    for (unsigned int y = 0; y < size.y; y++) {
        for (unsigned int x = 0; x < size.x; x++) {
            // Fix: Use sf::Vector2u({x, y}) or pass x, y as separate args as required by SFML 2.x
            // SFML 2.x getPixel takes unsigned int x, unsigned int y
            auto c1 = rendered.getPixel(x, y);
            auto c2 = target.getPixel(x, y);
            float dr = static_cast<float>(c1.r) - c2.r; // Cast differences to float before squaring
            float dg = static_cast<float>(c1.g) - c2.g;
            float db = static_cast<float>(c1.b) - c2.b;
            sum -= dr * dr + dg * dg + db * db;
        }
    }
    return sum;
}

// Optimized version that works directly with pixel buffers (CPU version)
float computeFitness(const std::vector<Pixel> &rendered, const std::vector<Pixel> &target,
                     unsigned width, unsigned height) {
    float sum = 0;
    const size_t size = static_cast<size_t>(width) * height; // Cast to size_t for calculation

    if (rendered.size() != size || target.size() != size) {
        // Handle size mismatch error - this indicates a serious problem
        std::cerr << "Fitness size mismatch! Rendered: " << rendered.size() << ", Target: " << target.size() <<
                ", Expected: " << size << std::endl;
        return -1e12f; // Return a very low fitness to indicate error, use float literal
    }

    // Process pixels in chunks for better cache utilization and OpenMP/SIMD
    const size_t CHUNK_SIZE = 4096; // Increased chunk size

#ifndef NO_OPENMP
#pragma omp parallel for reduction(-:sum) schedule(static, CHUNK_SIZE)
#endif
    for (size_t i = 0; i < size; i += CHUNK_SIZE) {
        float localSum = 0.0f; // Use float literal
        const size_t end = std::min(i + CHUNK_SIZE, size);

#ifdef __AVX__ // Check for AVX support (more common than just SSE)
    // Potential SIMD implementation here... but requires careful vectorization.
    // Sticking to OpenMP SIMD pragma for broader compiler support.
#pragma omp simd reduction(-:localSum)
    for (size_t j = i; j < end; j++) {
        const Pixel &c1 = rendered[j];
        const Pixel &c2 = target[j];
        int dr = static_cast<int>(c1.r) - c2.r;
        int dg = static_cast<int>(c1.g) - c2.g;
        int db = static_cast<int>(c1.b) - c2.b;
        localSum -= static_cast<float>(dr * dr + dg * dg + db * db); // Cast to float explicitly
    }

#else // Fallback if AVX is not available or NO_OPENMP is defined
        // Use OpenMP SIMD or a simple loop
#ifndef NO_OPENMP
#pragma omp simd reduction(-:localSum)
#endif
        for (size_t j = i; j < end; j++) {
            const Pixel &c1 = rendered[j];
            const Pixel &c2 = target[j];
            int dr = static_cast<int>(c1.r) - c2.r;
            int dg = static_cast<int>(c1.g) - c2.g;
            int db = static_cast<int>(c1.b) - c2.b;
            localSum -= static_cast<float>(dr * dr + dg * dg + db * db); // Cast to float explicitly
        }
#endif // __AVX__

        sum += localSum;
    }

    return sum;
}
