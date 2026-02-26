//
// Created by Viljen Apalset Vassbø on 18/02/2026.
//

#ifndef ALGOPAINTER_IRASTERIZER_H
#define ALGOPAINTER_IRASTERIZER_H

#pragma once
#include "Individual.h"
#include "Pixel.h"

#include <vector>

class IRasterizer {
  public:
    virtual ~IRasterizer() = default;

    // The core genetic algorithm pipeline methods
    virtual void uploadPopulation(const std::vector<Individual>& population, const std::vector<Pixel>& targetImage) = 0;
    virtual void renderAndEvaluate(const std::vector<Pixel>& targetImage, std::vector<float>& fitnessResults) = 0;
    virtual std::vector<Pixel> getRenderedImage(int index) = 0;
    virtual void resize(unsigned width, unsigned height, const std::vector<Pixel>& targetImage) = 0;

    virtual unsigned int getWidth() const = 0;
    virtual unsigned int getHeight() const = 0;
};

#endif // ALGOPAINTER_IRASTERIZER_H