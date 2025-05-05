//
// Created by vilje on 03/05/2025.
//

#ifndef APPLICATION_H
#define APPLICATION_H

#pragma once
#include <random>
#include <SFML/Graphics.hpp>

#include "Gene.h"
#include "Individual.h"
#include "Rasterizer.h" // CPU fallback
#include "CudaRasterizer.h" // GPU version
#include "Pixel.h" // Pixel struct definition
// #include "GAUtils.h" // GA constants are now defined here or passed

class Application {
public:
    Application();

    void run();

private:
    void processEvents();

    void update();

    void render();

    sf::RenderWindow window;
    // sf::RenderTexture renderTexture; // No longer used for main display rendering
    sf::Image targetImage; // Original full-res target image
    unsigned canvasW, canvasH;

    // GA Configuration - Moved constants here
    static constexpr int POPULATION_SIZE = 200; // Increased population size for GPU
    static constexpr int GENES_PER_INDIVIDUAL = 150; // More genes per individual
    static constexpr float MUTATION_RATE = 0.02f; // Adjusted mutation rate (per gene parameter)
    static constexpr int TOURNAMENT_SIZE = 5;

    // Display configuration
    static constexpr int DISPLAY_FREQUENCY = 20; // How often to update the display
    static constexpr bool SHOW_STATS = true;
    static constexpr bool USE_GPU = true; // Set to false to force CPU fallback

    // Progressive evolution
    static constexpr bool USE_PROGRESSIVE_RENDERING = true;
    static constexpr int INITIAL_RESOLUTION_FACTOR = 8; // Start at 1/8 resolution
    static constexpr int PROGRESSIVE_RESOLUTION_FREQUENCY = 50; // Generations between resolution increases
    int currentResolutionFactor = INITIAL_RESOLUTION_FACTOR;

    int generationCount = 0;
    int lastDisplayedGeneration = 0;
    sf::Clock performanceClock;

    // CPU Rasterizer (fallback or for single-image rendering)
    Rasterizer rasterizer;

    // GPU Rasterizer (main rendering/evaluation)
    CudaRasterizer cudaRasterizer;
    // bool gpuInitialized is managed internally by CudaRasterizer

    std::vector<Pixel> targetPixels; // Full-res target pixels
    std::vector<Pixel> downscaledTargetPixels; // Current resolution target pixels

    std::vector<Individual> population;
    std::vector<float> fitnessValues;
    Individual bestIndividual;

    std::mt19937 rng;

    // Methods for progressive rendering
    void downscaleTargetImage(int factor);

    void increaseResolution();
};

#endif //APPLICATION_H
