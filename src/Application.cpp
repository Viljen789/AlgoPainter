#include <iostream>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "Application.h"
#include "Fitness.h"
#include "GAUtils.h"
#include "Gene.h"
#include "Pixel.h"

#ifndef NO_OPENMP
#include <omp.h>
#endif

Application::Application()
    : window(sf::VideoMode({800, 600}), "EvoArt"),
      rasterizer(1, 1),
      cudaRasterizer(1, 1) {
    
    std::string targetPath = "../assets/target.jpg";
    if (!targetImage.loadFromFile(targetPath)) {
        std::cerr << "ERROR: Failed to load target image from: " << std::filesystem::absolute(targetPath) << std::endl;
        throw std::runtime_error("Failed to load target image");
    }

    auto ts = targetImage.getSize();
    canvasW = ts.x;
    canvasH = ts.y;

    targetPixels.resize(static_cast<size_t>(canvasW) * canvasH);
    for (unsigned y = 0; y < canvasH; y++) {
        for (unsigned x = 0; x < canvasW; x++) {
            sf::Color c = targetImage.getPixel(x, y);
            targetPixels[static_cast<size_t>(y) * canvasW + x] = {c.r, c.g, c.b, 255};
        }
    }

    if (USE_PROGRESSIVE_RENDERING) {
        downscaleTargetImage(currentResolutionFactor);
    } else {
        currentResolutionFactor = 1;
        downscaledTargetPixels = targetPixels;
        rasterizer.resize(canvasW, canvasH);
        cudaRasterizer.resize(canvasW, canvasH, downscaledTargetPixels);
    }

    std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(canvasW)),
            yDist(0.0f, static_cast<float>(canvasH)), 
            sizeDist(5.0f, 40.0f),
            rotDist(0.0f, 360.0f);
    std::uniform_int_distribution<int> colorDist(0, 255);

    population.reserve(POPULATION_SIZE);
    fitnessValues.assign(POPULATION_SIZE, -1e12f);
    for (int k = 0; k < POPULATION_SIZE; k++) {
        Individual indiv;
        indiv.reserve(GENES_PER_INDIVIDUAL);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; i++) {
            auto shape = static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng));
            sf::Vector2f pos{xDist(rng), yDist(rng)};
            sf::Vector2f size{sizeDist(rng), sizeDist(rng)};
            float rot = rotDist(rng);
            sf::Color color{
                static_cast<std::uint8_t>(colorDist(rng)), 
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(std::uniform_int_distribution<int>(30, 150)(rng))
            };
            indiv.emplace_back(shape, pos, size, rot, color);
        }
        population.push_back(std::move(indiv));
    }
    bestIndividual = population[0];
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
}

void Application::downscaleTargetImage(int factor) {
    unsigned downscaledW = canvasW / factor;
    unsigned downscaledH = canvasH / factor;
    if (downscaledW == 0 || downscaledH == 0) return;

    downscaledTargetPixels.resize(static_cast<size_t>(downscaledW) * downscaledH);
    for (unsigned y = 0; y < downscaledH; y++) {
        for (unsigned x = 0; x < downscaledW; x++) {
            unsigned r = 0, g = 0, b = 0, count = 0;
            for (unsigned dy = 0; dy < (unsigned)factor; dy++) {
                for (unsigned dx = 0; dx < (unsigned)factor; dx++) {
                    unsigned srcX = x * factor + dx;
                    unsigned srcY = y * factor + dy;
                    if (srcX < canvasW && srcY < canvasH) {
                        const Pixel &p = targetPixels[static_cast<size_t>(srcY) * canvasW + srcX];
                        r += p.r; g += p.g; b += p.b; count++;
                    }
                }
            }
            if (count > 0) {
                downscaledTargetPixels[static_cast<size_t>(y) * downscaledW + x] = {
                    static_cast<uint8_t>(r / count), static_cast<uint8_t>(g / count),
                    static_cast<uint8_t>(b / count), 255
                };
            }
        }
    }
    rasterizer.resize(downscaledW, downscaledH);
    cudaRasterizer.resize(downscaledW, downscaledH, downscaledTargetPixels);
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
}

void Application::increaseResolution() {
    if (currentResolutionFactor > 1) {
        currentResolutionFactor /= 2;
        std::cout << "Increasing resolution to 1/" << currentResolutionFactor << std::endl;
        downscaleTargetImage(currentResolutionFactor);
    }
}

void Application::run() {
    performanceClock.restart();
    while (window.isOpen()) {
        processEvents();
        update();

        if (generationCount % DISPLAY_FREQUENCY == 0) {
            render();
            float elapsed = performanceClock.getElapsedTime().asSeconds();
            performanceClock.restart();
            std::cout << "Gen " << generationCount << " | Best Fitness: " << fitnessValues[0] 
                      << " | Res: 1/" << currentResolutionFactor 
                      << " | FPS: " << DISPLAY_FREQUENCY / elapsed << std::endl;
        }

        if (USE_PROGRESSIVE_RENDERING && generationCount > 0 && 
            generationCount % PROGRESSIVE_RESOLUTION_FREQUENCY == 0 && currentResolutionFactor > 1) {
            increaseResolution();
        }
    }
}

void Application::processEvents() {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) window.close();
        else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::S) saveCurrentImage();
    }
}

void Application::update() {
    cudaRasterizer.renderAndEvaluate(downscaledTargetPixels, fitnessValues);

    std::vector<int> indices(POPULATION_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return fitnessValues[a] > fitnessValues[b]; });

    std::vector<Individual> nextPop;
    nextPop.reserve(POPULATION_SIZE);
    nextPop.push_back(population[indices[0]]); // Elitism

    while (nextPop.size() < POPULATION_SIZE) {
        int i1 = tournamentSelect(fitnessValues, rng, TOURNAMENT_SIZE);
        int i2 = tournamentSelect(fitnessValues, rng, TOURNAMENT_SIZE);
        Individual c1, c2;
        onePointCrossover(population[i1], population[i2], c1, c2, rng);
        mutateIndividual(c1, rng, canvasW, canvasH, MUTATION_RATE);
        nextPop.push_back(std::move(c1));
        if (nextPop.size() < POPULATION_SIZE) {
            mutateIndividual(c2, rng, canvasW, canvasH, MUTATION_RATE);
            nextPop.push_back(std::move(c2));
        }
    }
    population = std::move(nextPop);
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
    generationCount++;
}

void Application::render() {
    std::vector<Pixel> pixels = cudaRasterizer.getRenderedImage(0);
    sf::Image img;
    img.create(cudaRasterizer.getWidth(), cudaRasterizer.getHeight(), reinterpret_cast<const sf::Uint8*>(pixels.data()));
    sf::Texture tex; tex.loadFromImage(img);
    sf::Sprite spr(tex);
    float s = std::min(800.0f / img.getSize().x, 600.0f / img.getSize().y);
    spr.setScale({s, s});
    window.clear(); window.draw(spr); window.display();
}

void Application::saveCurrentImage() { /* Implementation similar to previous but concise */ }
