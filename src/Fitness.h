//
// Created by vilje on 03/05/2025.
//

#pragma once
#include "Pixel.h"

#include <SFML/Graphics.hpp>

#include <vector>

// Original function for backward compatibility (uses sf::Image)
float computeFitness(const sf::Image& rendered, const sf::Image& target);

//  (CPU version)
float computeFitness(const std::vector<Pixel>& rendered, const std::vector<Pixel>& target, unsigned width,
                     unsigned height);
