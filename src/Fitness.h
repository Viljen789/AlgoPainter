//
// Created by vilje on 03/05/2025.
//

#pragma once
#include <SFML/Graphics.hpp>
#include "Pixel.h" // Include Pixel definition
#include <vector>

// Original function for backward compatibility (uses sf::Image)
float computeFitness(const sf::Image &rendered, const sf::Image &target);

// New, optimized function using direct pixel buffers (CPU version)
float computeFitness(const std::vector<Pixel> &rendered, const std::vector<Pixel> &target,
                     unsigned width, unsigned height);

// Note: The CUDA fitness kernel is declared and defined in the CUDA files.
// It does not need a declaration here unless you call it directly from a .cpp file,
// which you shouldn't. The CudaRasterizer calls it.
