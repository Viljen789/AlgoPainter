#ifndef APPLICATION_H
#define APPLICATION_H

#pragma once
#include <random>
#include <SFML/Graphics.hpp>

#include "Gene.h"
#include "Individual.h"
#include "Rasterizer.h"
#include "CudaRasterizer.h"
#include "Pixel.h"

class Application {
public:
    Application();
    void run();

private:
    void processEvents();
    void update();
    void render();
    void saveCurrentImage();

    sf::RenderWindow window;
    sf::Image targetImage;
    unsigned canvasW, canvasH;

    static constexpr int POPULATION_SIZE = 100;
    static constexpr int GENES_PER_INDIVIDUAL = 250;
    static constexpr float MUTATION_RATE = 0.05f;
    static constexpr int TOURNAMENT_SIZE = 4;

    static constexpr int DISPLAY_FREQUENCY = 10;
    static constexpr bool SHOW_STATS = true;
    static constexpr bool USE_GPU = true;

    static constexpr bool USE_PROGRESSIVE_RENDERING = true;
    static constexpr int INITIAL_RESOLUTION_FACTOR = 16;
    static constexpr int PROGRESSIVE_RESOLUTION_FREQUENCY = 100;
    int currentResolutionFactor = INITIAL_RESOLUTION_FACTOR;

    int generationCount = 0;
    int lastDisplayedGeneration = 0;
    sf::Clock performanceClock;

    Rasterizer rasterizer;
    CudaRasterizer cudaRasterizer;

    std::vector<Pixel> targetPixels;
    std::vector<Pixel> downscaledTargetPixels;

    std::vector<Individual> population;
    std::vector<float> fitnessValues;
    Individual bestIndividual;

    std::mt19937 rng;

    void downscaleTargetImage(int factor);
    void increaseResolution();
};

#endif //APPLICATION_H
