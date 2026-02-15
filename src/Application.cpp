#include <iostream>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdint>

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
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    if (!targetImage.loadFromFile("../assets/cool-dog.jpg")) {
        std::cerr << "ERROR: Failed to load target image from: " << std::filesystem::absolute("../assets/cool-dog.jpg")
                <<
                std::endl;
        std::cerr << "Please make sure the image exists at this location." << std::endl;
        throw std::runtime_error("Failed to load target image");
    }

    auto ts = targetImage.getSize();
    canvasW = ts.x;
    canvasH = ts.y;
    unsigned numPixels = canvasW * canvasH;
    minPossibleFitness = -static_cast<float>(255 * 255 * 3) * numPixels;

    std::cout << "TargetImage size: " << ts.x << ", " << ts.y << std::endl;

    targetPixels.resize(static_cast<size_t>(canvasW) * canvasH);
    for (unsigned y = 0; y < canvasH; y++) {
        for (unsigned x = 0; x < canvasW; x++) {
            sf::Color c = targetImage.getPixel({x, y});
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

    if (USE_GPU) {
        if (cudaRasterizer.isInitialized()) {
            std::cout << "GPU acceleration initialized successfully" << std::endl;
        } else {
            std::cout << "GPU acceleration not available or failed initialization, using CPU implementation" <<
                    std::endl;
        }
    } else {
        std::cout << "USE_GPU is false, using CPU implementation" << std::endl;
    }

    std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(canvasW)),
            yDist(0.0f, static_cast<float>(canvasH)), sizeDist(5.0f, 50.0f);
    std::uniform_int_distribution<int> colorDist(0, 255);

    population.reserve(POPULATION_SIZE);
    fitnessValues.resize(POPULATION_SIZE);

    for (int k = 0; k < POPULATION_SIZE; k++) {
        Individual indiv;
        indiv.reserve(GENES_PER_INDIVIDUAL);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; i++) {
            auto shape = static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng));
            sf::Vector2f pos{xDist(rng), yDist(rng)};
            float size = sizeDist(rng);
            sf::Color color{
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(std::uniform_int_distribution<int>(50, 200)(rng))
            };
            indiv.emplace_back(shape, pos, size, color);
        }
        population.push_back(std::move(indiv));
    }

    bestIndividual = population[0];

    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
}

void Application::downscaleTargetImage(int factor) {
    unsigned downscaledW = canvasW / factor;
    unsigned downscaledH = canvasH / factor;

    if (downscaledW == 0 || downscaledH == 0) {
        std::cerr << "Warning: Resolution factor " << factor << " results in zero dimension." << std::endl;
        return;
    }

    downscaledTargetPixels.resize(static_cast<size_t>(downscaledW) * downscaledH);

    for (unsigned y = 0; y < downscaledH; y++) {
        for (unsigned x = 0; x < downscaledW; x++) {
            unsigned r = 0, g = 0, b = 0;
            unsigned count = 0;

            for (unsigned dy = 0; dy < factor; dy++) {
                for (unsigned dx = 0; dx < factor; dx++) {
                    unsigned srcX = x * factor + dx;
                    unsigned srcY = y * factor + dy;

                    if (srcX < canvasW && srcY < canvasH) {
                        const Pixel &p = targetPixels[static_cast<size_t>(srcY) * canvasW + srcX];
                        r += p.r;
                        g += p.g;
                        b += p.b;
                        count++;
                    }
                }
            }

            if (count > 0) {
                downscaledTargetPixels[static_cast<size_t>(y) * downscaledW + x] = {
                    static_cast<uint8_t>(r / count),
                    static_cast<uint8_t>(g / count),
                    static_cast<uint8_t>(b / count),
                    255
                };
            } else {
                downscaledTargetPixels[static_cast<size_t>(y) * downscaledW + x] = {0, 0, 0, 255};
            }
        }
    }

    rasterizer.resize(downscaledW, downscaledH);
    cudaRasterizer.resize(downscaledW, downscaledH, downscaledTargetPixels);
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
}

void Application::increaseResolution() {
    if (currentResolutionFactor > 1) {
        int nextFactor = currentResolutionFactor / 2;
        if (canvasW / nextFactor == 0 || canvasH / nextFactor == 0) {
            std::cout << "Cannot increase resolution further without zero dimensions." << std::endl;
            return;
        }

        currentResolutionFactor = nextFactor;
        std::cout << "Increasing resolution to 1/" << currentResolutionFactor << " of original" << std::endl;

        downscaleTargetImage(currentResolutionFactor);
    }
}

void Application::run() {
    performanceClock.restart();
    while (window.isOpen()) {
        processEvents();

        update();

        bool forceDisplay = (USE_PROGRESSIVE_RENDERING && generationCount > 0 &&
                             generationCount % PROGRESSIVE_RESOLUTION_FREQUENCY == 0 &&
                             currentResolutionFactor < INITIAL_RESOLUTION_FACTOR);
        if (generationCount % DISPLAY_FREQUENCY == 0 || generationCount == 0 || forceDisplay) {
            render();
            lastDisplayedGeneration = generationCount;

            if (SHOW_STATS && generationCount > 0) {
                float elapsed = performanceClock.getElapsedTime().asSeconds();
                performanceClock.restart();

                float gensPerSecond = DISPLAY_FREQUENCY / elapsed;
                std::cout << "Generation " << generationCount
                        << " | fitness=" << bestFitness
                        << " | " << gensPerSecond << " gen/s" << std::endl;
            } else if (SHOW_STATS && generationCount == 0) {
                std::cout << "Generation " << generationCount
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ")" << std::endl;
                performanceClock.restart();
            } else if (!SHOW_STATS && (generationCount % DISPLAY_FREQUENCY == 0 || generationCount == 0 ||
                                       forceDisplay)) {
                std::cout << "Generation " << generationCount
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ")" << std::endl;
            }
        }

        if (USE_PROGRESSIVE_RENDERING &&
            generationCount > 0 &&
            generationCount % PROGRESSIVE_RESOLUTION_FREQUENCY == 0 &&
            currentResolutionFactor > 1) {
            increaseResolution();
        }
    }
}

void Application::processEvents() {
    while (const auto event = window.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            window.close();
        } else if (const auto *key = event->getIf<sf::Event::KeyPressed>()) {
            if (key->code == sf::Keyboard::Key::S) {
                saveCurrentImage();
            }
        }
    }
}

void Application::saveCurrentImage() {
    std::vector<Pixel> bestImagePixels = cudaRasterizer.getRenderedImage(0);

    unsigned renderW = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getWidth()
                           : rasterizer.getWidth();
    unsigned renderH = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getHeight()
                           : rasterizer.getHeight();

    sf::Texture tempTexture;
    if (!tempTexture.resize({renderW, renderH})) {
        std::cerr << "Failed to resize texture for saving" << std::endl;
        return;
    }

    tempTexture.update(reinterpret_cast<const std::uint8_t *>(bestImagePixels.data()));

    sf::Image bestImage = tempTexture.copyToImage();

    std::filesystem::path outputDir = "../../output";
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directory(outputDir);
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "../../output/evo_art_gen" << generationCount << "_";
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".png";
    std::string filename = oss.str();

    if (bestImage.saveToFile(filename)) {
        std::cout << "Image saved successfully to: " << std::filesystem::absolute(filename) << std::endl;
    } else {
        std::cerr << "Failed to save image to: " << std::filesystem::absolute(filename) << std::endl;
    }
}


void Application::update() {
    // 1. Evaluate fitness via GPU
    cudaRasterizer.renderAndEvaluate(downscaledTargetPixels, fitnessValues);

    // 2. Find best individual
    int bestIdx = 0;
    float currentBestFitness = fitnessValues[0];
    for (int i = 1; i < POPULATION_SIZE; i++) {
        if (fitnessValues[i] > currentBestFitness) {
            currentBestFitness = fitnessValues[i];
            bestIdx = i;
        }
    }

    // 3. Update global best
    if (currentBestFitness > bestFitness) {
        bestFitness = currentBestFitness;
        bestIndividual = population[bestIdx];
    }

    // 4. Create new population
    std::vector<Individual> newPop;
    newPop.reserve(POPULATION_SIZE);

    // Elitism: keep best individuals
    std::vector<std::pair<float, int>> sorted(POPULATION_SIZE);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        sorted[i] = {fitnessValues[i], i};
    }
    std::partial_sort(sorted.begin(), sorted.begin() + ELITE_COUNT, sorted.end(), std::greater<>());

    for (int i = 0; i < ELITE_COUNT; i++) {
        newPop.push_back(population[sorted[i].second]);
    }

    // Fill rest with crossover + mutation
    while (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
        int p1 = tournamentSelect(fitnessValues, rng, TOURNAMENT_SIZE);
        int p2 = tournamentSelect(fitnessValues, rng, TOURNAMENT_SIZE);

        Individual c1, c2;
        onePointCrossover(population[p1], population[p2], c1, c2, rng);

        mutateIndividual(c1, rng, canvasW, canvasH, currentMutationRate);
        mutateIndividual(c2, rng, canvasW, canvasH, currentMutationRate);

        newPop.push_back(std::move(c1));
        if (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
            newPop.push_back(std::move(c2));
        }
    }

    population = std::move(newPop);

    // 5. Upload and increment
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);

    generationCount++;
}

void Application::render() {
    std::vector<Pixel> bestImagePixels = cudaRasterizer.getRenderedImage(0);

    unsigned renderW = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getWidth()
                           : rasterizer.getWidth();
    unsigned renderH = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getHeight()
                           : rasterizer.getHeight();

    // --- SFML 3 FIX: Update Texture Directly ---
    sf::Texture renderTextureDisplay;

    // 1. Resize texture to match data
    if (!renderTextureDisplay.resize({renderW, renderH})) {
        std::cerr << "Failed to resize display texture" << std::endl;
        return;
    }

    renderTextureDisplay.update(reinterpret_cast<const std::uint8_t *>(bestImagePixels.data()));


    sf::Sprite spr(renderTextureDisplay);

    float scaleX = static_cast<float>(window.getSize().x) / renderW;
    float scaleY = static_cast<float>(window.getSize().y) / renderH;

    float uniformScale = std::min(scaleX, scaleY);
    spr.setScale({uniformScale, uniformScale});

    sf::Vector2f offset{
        (static_cast<float>(window.getSize().x) - renderW * uniformScale) / 2.f,
        (static_cast<float>(window.getSize().y) - renderH * uniformScale) / 2.f
    };
    spr.setPosition(offset);

    window.clear(sf::Color::Black);
    window.draw(spr);
    window.display();
}

