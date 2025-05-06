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
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    if (!targetImage.loadFromFile("../assets/AstridOgKnut.jpg")) {
        std::cerr << "ERROR: Failed to load target image from: " << std::filesystem::absolute("../assets/SexyEmil.jpg")
                <<
                std::endl;
        std::cerr << "Please make sure the image exists at this location." << std::endl;
        throw std::runtime_error("Failed to load target image");
    }

    auto ts = targetImage.getSize();
    canvasW = ts.x;
    canvasH = ts.y;

    std::cout << "TargetImage size: " << ts.x << ", " << ts.y << std::endl;

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
    population.emplace_back();
    population[0].reserve(GENES_PER_INDIVIDUAL);
    for (int k = 1; k < POPULATION_SIZE; k++) {
        Individual indiv;
        indiv.reserve(GENES_PER_INDIVIDUAL);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; i++) {
            auto shape = static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng));
            sf::Vector2f pos{xDist(rng), yDist(rng)};
            float size = sizeDist(rng);
            sf::Color color{
                static_cast<std::uint8_t>(colorDist(rng)), static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(std::uniform_int_distribution<int>(50, 200)(rng))
            };
            indiv.emplace_back(shape, pos, size, color);
        }
        population.push_back(std::move(indiv));
    }

    if (POPULATION_SIZE > 1) {
        bestIndividual = population[1];
    } else {
        bestIndividual.reserve(GENES_PER_INDIVIDUAL);
    }

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
                        << " best fitness = " << fitnessValues[0]
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ") | Perf: " << gensPerSecond << " generations/sec" << std::endl;
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
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window.close();
        } else if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::S) {
                saveCurrentImage();
            }
        }
    }
}

void Application::saveCurrentImage() {
    std::vector<Pixel> bestImagePixels = cudaRasterizer.getRenderedImage(0);

    sf::Image bestImage;
    unsigned renderW = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getWidth()
                           : rasterizer.getWidth();
    unsigned renderH = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getHeight()
                           : rasterizer.getHeight();

    bestImage.create(renderW, renderH, reinterpret_cast<const sf::Uint8 *>(bestImagePixels.data()));

    // Create output directory if it doesn't exist
    std::filesystem::path outputDir = "../../output";
    if (!std::filesystem::exists(outputDir)) {
        std::filesystem::create_directory(outputDir);
    }

    // Generate filename with timestamp and generation number
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
    cudaRasterizer.renderAndEvaluate(downscaledTargetPixels, fitnessValues);

    int bestIndex = 0;
    float bestFit = fitnessValues[bestIndex];
    for (int k = 1; k < POPULATION_SIZE; k++) {
        if (fitnessValues[k] > bestFit) {
            bestFit = fitnessValues[k];
            bestIndex = k;
        }
    }
    bestIndividual = population[bestIndex];

    std::vector<Individual> newPop;
    newPop.reserve(POPULATION_SIZE);
    newPop.push_back(bestIndividual);

    if (bestIndividual.size() != GENES_PER_INDIVIDUAL) {
        std::cerr << "Warning: Best individual has " << bestIndividual.size()
                << " genes, expected " << GENES_PER_INDIVIDUAL << std::endl;

        if (bestIndividual.empty()) {
            bestIndividual.clear();
            bestIndividual.reserve(GENES_PER_INDIVIDUAL);

            std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(canvasW));
            std::uniform_real_distribution<float> yDist(0.0f, static_cast<float>(canvasH));
            std::uniform_real_distribution<float> sizeDist(5.0f, 50.0f);
            std::uniform_int_distribution<int> colorDist(0, 255);

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
                bestIndividual.emplace_back(shape, pos, size, color);
            }

            newPop[0] = bestIndividual;
        } else if (bestIndividual.size() < GENES_PER_INDIVIDUAL) {
            while (bestIndividual.size() < GENES_PER_INDIVIDUAL) {
                int srcIdx = std::uniform_int_distribution<int>(0, bestIndividual.size() - 1)(rng);
                bestIndividual.push_back(bestIndividual[srcIdx]);
            }
            newPop[0] = bestIndividual;
        } else if (bestIndividual.size() > GENES_PER_INDIVIDUAL) {
            bestIndividual.resize(GENES_PER_INDIVIDUAL);
            newPop[0] = bestIndividual;
        }
    }

    Individual child1, child2;
    child1.reserve(GENES_PER_INDIVIDUAL);
    child2.reserve(GENES_PER_INDIVIDUAL);

    std::vector<Individual> validatedPopulation;
    validatedPopulation.reserve(POPULATION_SIZE);
    validatedPopulation.push_back(bestIndividual);

    for (int i = 1; i < POPULATION_SIZE && i < population.size(); i++) {
        if (population[i].size() == GENES_PER_INDIVIDUAL) {
            validatedPopulation.push_back(population[i]);
        } else {
            Individual newIndiv = bestIndividual;
            mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 5);
            validatedPopulation.push_back(newIndiv);
        }
    }

    while (validatedPopulation.size() < POPULATION_SIZE) {
        Individual newIndiv = bestIndividual;
        mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 10);
        validatedPopulation.push_back(newIndiv);
    }

    while (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
        int i1 = tournamentSelect(fitnessValues, rng);
        int i2 = tournamentSelect(fitnessValues, rng);

        if (i1 < 0 || i2 < 0 || i1 >= validatedPopulation.size() || i2 >= validatedPopulation.size()) {
            i1 = 0;
            i2 = 0;
        }

        onePointCrossover(validatedPopulation[i1], validatedPopulation[i2], child1, child2, rng);

        if (child1.size() != GENES_PER_INDIVIDUAL) {
            child1 = validatedPopulation[0];
            mutateIndividual(child1, rng, canvasW, canvasH, MUTATION_RATE * 2);
        }

        if (child2.size() != GENES_PER_INDIVIDUAL) {
            child2 = validatedPopulation[0];
            mutateIndividual(child2, rng, canvasW, canvasH, MUTATION_RATE * 2);
        }

        mutateIndividual(child1, rng, canvasW, canvasH, MUTATION_RATE);
        mutateIndividual(child2, rng, canvasW, canvasH, MUTATION_RATE);

        newPop.push_back(child1);
        if (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
            newPop.push_back(child2);
        }
    }

    population.swap(newPop);

    bool populationValid = true;
    for (const auto &individual: population) {
        if (individual.size() != GENES_PER_INDIVIDUAL) {
            populationValid = false;
            std::cerr << "Invalid individual size after update: " << individual.size() << std::endl;
            break;
        }
    }

    if (!populationValid) {
        std::cerr << "Warning: Invalid population detected. Regenerating..." << std::endl;
        std::vector<Individual> regeneratedPop;
        regeneratedPop.reserve(POPULATION_SIZE);
        regeneratedPop.push_back(population[0]);

        for (int i = 1; i < POPULATION_SIZE; i++) {
            Individual newIndiv = population[0];
            mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 5);
            regeneratedPop.push_back(newIndiv);
        }

        population.swap(regeneratedPop);
    }

    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);

    generationCount++;
}

void Application::render() {
    std::vector<Pixel> bestImagePixels = cudaRasterizer.getRenderedImage(0);

    sf::Image bestImage;
    unsigned renderW = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getWidth()
                           : rasterizer.getWidth();
    unsigned renderH = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getHeight()
                           : rasterizer.getHeight();

    bestImage.create(renderW, renderH, reinterpret_cast<const sf::Uint8 *>(bestImagePixels.data()));

    sf::Texture renderTextureDisplay;
    if (!renderTextureDisplay.loadFromImage(bestImage)) {
        std::cerr << "Failed to load rendered image into texture" << std::endl;
        return;
    }
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
