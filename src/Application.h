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

    static constexpr int POPULATION_SIZE = 200;
    static constexpr int GENES_PER_INDIVIDUAL = 150;
    static constexpr float MUTATION_RATE = 0.02f;
    static constexpr int TOURNAMENT_SIZE = 5;

    static constexpr int DISPLAY_FREQUENCY = 20;
    static constexpr bool SHOW_STATS = true;
    static constexpr bool USE_GPU = true;

    static constexpr bool USE_PROGRESSIVE_RENDERING = true;
    static constexpr int INITIAL_RESOLUTION_FACTOR = 8;
    static constexpr int PROGRESSIVE_RESOLUTION_FREQUENCY = 50;
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
