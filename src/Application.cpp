//
// Created by vilje on 03/05/2025.
//

#include <iostream>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <filesystem> // Include filesystem for detailed error messages

#include "Application.h"
#include "Fitness.h"
#include "GAUtils.h" // Includes GA constants now
#include "Gene.h"
#include "Pixel.h" // Include Pixel definition

#ifndef NO_OPENMP
#include <omp.h>
#endif

Application::Application()
    : window(sf::VideoMode({800, 600}), "EvoArt"), // width, height, title
      rasterizer(1, 1), // Temporary initialization, will be resized below
      cudaRasterizer(1, 1) // Temporary initialization, will be resized below
{
    // Removed renderTexture initialization here, as it's not a member anymore

    // Print the current working directory
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    // Print more detailed error message if image fails to load
    if (!targetImage.loadFromFile("../assets/SexyEmil.jpg")) {
        std::cerr << "ERROR: Failed to load target image from: " << std::filesystem::absolute("../assets/SexyEmil.jpg")
                <<
                std::endl;
        std::cerr << "Please make sure the image exists at this location." << std::endl;
        throw std::runtime_error("Failed to load target image");
    }

    auto ts = targetImage.getSize(); // sf::Vector2u
    canvasW = ts.x; // unsigned int
    canvasH = ts.y; // unsigned int
    // renderTexture = sf::RenderTexture{ts}; // This line should be removed as renderTexture is not a member

    std::cout << "TargetImage size: " << ts.x << ", " << ts.y << std::endl;

    // Convert target image to pixel buffer once
    targetPixels.resize(static_cast<size_t>(canvasW) * canvasH); // Cast canvasW to size_t
    for (unsigned y = 0; y < canvasH; y++) {
        for (unsigned x = 0; x < canvasW; x++) {
            // Fix: SFML 2.x getPixel takes unsigned int x, unsigned int y
            sf::Color c = targetImage.getPixel(x, y);
            targetPixels[static_cast<size_t>(y) * canvasW + x] = {c.r, c.g, c.b, 255}; // Cast y to size_t
        }
    }

    // Initialize progressive rendering and downscale target
    if (USE_PROGRESSIVE_RENDERING) {
        downscaleTargetImage(currentResolutionFactor);
    } else {
        currentResolutionFactor = 1; // Ensure factor is 1 if not progressive
        downscaledTargetPixels = targetPixels;
        // Initialize rasterizers at full resolution
        rasterizer.resize(canvasW, canvasH);
        cudaRasterizer.resize(canvasW, canvasH, downscaledTargetPixels); // Resize GPU rasterizer
    }


    // Initialize GPU if enabled
    // cudaRasterizer constructor calls initialize() internally
    if (USE_GPU) {
        // Check USE_GPU constant
        if (cudaRasterizer.isInitialized()) {
            // Check actual initialization status
            std::cout << "GPU acceleration initialized successfully" << std::endl;
        } else {
            std::cout << "GPU acceleration not available or failed initialization, using CPU implementation" <<
                    std::endl;
        }
    } else {
        std::cout << "USE_GPU is false, using CPU implementation" << std::endl;
    }


    std::uniform_real_distribution<float> xDist(0.0f, static_cast<float>(canvasW)),
            yDist(0.0f, static_cast<float>(canvasH)), sizeDist(5.0f, 50.0f); // Use float literals and cast canvasW/H
    std::uniform_int_distribution<int> colorDist(0, 255);

    population.reserve(POPULATION_SIZE);
    fitnessValues.resize(POPULATION_SIZE);
    // The first individual will be replaced by the best later
    population.emplace_back(); // Add a placeholder individual
    population[0].reserve(GENES_PER_INDIVIDUAL); // Reserve space in the placeholder
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
                static_cast<std::uint8_t>(std::uniform_int_distribution<int>(50, 200)(rng)) // Random alpha
            };
            indiv.emplace_back(shape, pos, size, color);
        }
        population.push_back(std::move(indiv));
    }

    // Now initialize bestIndividual based on the first generated one, or copy one later
    if (POPULATION_SIZE > 1) {
        bestIndividual = population[1]; // Start with the first actual generated individual
    } else {
        bestIndividual.reserve(GENES_PER_INDIVIDUAL); // Keep it empty or add default genes if POPULATION_SIZE is 1
    }


    // Upload initial population to GPU (or CPU)
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels); // Pass target for GPU upload
}

void Application::downscaleTargetImage(int factor) {
    unsigned downscaledW = canvasW / factor;
    unsigned downscaledH = canvasH / factor;

    if (downscaledW == 0 || downscaledH == 0) {
        std::cerr << "Warning: Resolution factor " << factor << " results in zero dimension." << std::endl;
        return; // Avoid zero dimensions
    }

    downscaledTargetPixels.resize(static_cast<size_t>(downscaledW) * downscaledH); // Cast downscaledW to size_t

    // Simple box filter downsampling
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
                        // Cast srcY to size_t
                        r += p.r;
                        g += p.g;
                        b += p.b;
                        count++;
                    }
                }
            }

            if (count > 0) {
                downscaledTargetPixels[static_cast<size_t>(y) * downscaledW + x] = {
                    // Cast y to size_t
                    static_cast<uint8_t>(r / count),
                    static_cast<uint8_t>(g / count),
                    static_cast<uint8_t>(b / count),
                    255 // Downscaled target remains opaque
                };
            } else {
                downscaledTargetPixels[static_cast<size_t>(y) * downscaledW + x] = {0, 0, 0, 255};
                // Black if no source pixels
            }
        }
    }

    // Resize rasterizers for current resolution
    rasterizer.resize(downscaledW, downscaledH);
    // GPU rasterizer needs target image data uploaded again
    cudaRasterizer.resize(downscaledW, downscaledH, downscaledTargetPixels);

    // Re-upload population after resize, as GPU buffers are reallocated
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels);
}

void Application::increaseResolution() {
    if (currentResolutionFactor > 1) {
        int nextFactor = currentResolutionFactor / 2;
        if (canvasW / nextFactor == 0 || canvasH / nextFactor == 0) {
            // Stop if next resolution step leads to zero dimensions
            std::cout << "Cannot increase resolution further without zero dimensions." << std::endl;
            return;
        }

        currentResolutionFactor = nextFactor;
        std::cout << "Increasing resolution to 1/" << currentResolutionFactor << " of original" << std::endl;

        downscaleTargetImage(currentResolutionFactor);

        // downscaleTargetImage already calls cudaRasterizer.resize and uploadPopulation
    }
}

void Application::run() {
    performanceClock.restart();
    while (window.isOpen()) {
        processEvents();

        // Limit update frequency if needed, but usually you want to run GA as fast as possible
        // sf::Time elapsed = performanceClock.getElapsedTime();
        // if (elapsed > sf::seconds(1.0f/60.0f)) { // Limit to 60 updates per second
        //    update();
        //    performanceClock.restart();
        // } else {
        //    sf::sleep(sf::seconds(1.0f/60.0f) - elapsed);
        // }

        update();


        // Only render periodically to improve performance
        // Ensure the best individual is rendered after resolution increase, even if DISPLAY_FREQUENCY is large
        bool forceDisplay = (USE_PROGRESSIVE_RENDERING && generationCount > 0 &&
                             generationCount % PROGRESSIVE_RESOLUTION_FREQUENCY == 0 &&
                             currentResolutionFactor < INITIAL_RESOLUTION_FACTOR);
        if (generationCount % DISPLAY_FREQUENCY == 0 || generationCount == 0 || forceDisplay) {
            // Render on generation 0
            render();
            lastDisplayedGeneration = generationCount;

            if (SHOW_STATS && generationCount > 0) {
                // Calculate elapsed time since last stats display
                float elapsed = performanceClock.getElapsedTime().asSeconds();
                // Restart clock after displaying stats
                performanceClock.restart();

                float gensPerSecond = DISPLAY_FREQUENCY / elapsed; // Simple average
                // If forcing display, the actual number of gens might be less than DISPLAY_FREQUENCY since the last display
                if (forceDisplay) {
                    // Re-calculate how many generations actually passed since last display
                    // This is tricky if DISPLAY_FREQUENCY is not a divisor of PROGRESSIVE_RESOLUTION_FREQUENCY
                    // For simplicity, let's just report average over DISPLAY_FREQUENCY
                }
                std::cout << "Generation " << generationCount
                        << " best fitness = " << fitnessValues[0] // Assuming best is always at index 0 after update()
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ") | Perf: " << gensPerSecond << " generations/sec" << std::endl;
            } else if (SHOW_STATS && generationCount == 0) {
                // Display initial state and start performance clock
                std::cout << "Generation " << generationCount
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ")" << std::endl;
                performanceClock.restart(); // Start timing after initial display
            } else if (!SHOW_STATS && (generationCount % DISPLAY_FREQUENCY == 0 || generationCount == 0 ||
                                       forceDisplay)) {
                // Display basic info even if stats are off
                std::cout << "Generation " << generationCount
                        << " (resolution: 1/" << currentResolutionFactor
                        << ", mode: " << (cudaRasterizer.isInitialized() ? "GPU" : "CPU")
                        << ")" << std::endl;
            }
        }

        // Progressive resolution increases (check AFTER potential display)
        if (USE_PROGRESSIVE_RENDERING &&
            generationCount > 0 && // Don't increase before first generation
            generationCount % PROGRESSIVE_RESOLUTION_FREQUENCY == 0 && // Use a dedicated constant
            currentResolutionFactor > 1) {
            // Ensure we haven't reached full resolution
            increaseResolution();
        }
    }
}

void Application::processEvents() {
    sf::Event event; // Event object needs to be passed to pollEvent
    while (window.pollEvent(event)) {
        // Fix: pollEvent takes an sf::Event&
        if (event.type == sf::Event::Closed) {
            // Access event type via .type
            window.close();
        }
    }
}

void Application::update() {
    // renderAndEvaluate handles CPU/GPU selection internally
    cudaRasterizer.renderAndEvaluate(downscaledTargetPixels, fitnessValues);

    int bestIndex = 0;
    float bestFit = fitnessValues[bestIndex]; // Use float literal
    for (int k = 1; k < POPULATION_SIZE; k++) {
        if (fitnessValues[k] > bestFit) {
            bestFit = fitnessValues[k];
            bestIndex = k;
        }
    }
    bestIndividual = population[bestIndex];

    std::vector<Individual> newPop;
    newPop.reserve(POPULATION_SIZE);
    newPop.push_back(bestIndividual); // Elitism - keep the best

    // Ensure bestIndividual has the correct number of genes
    if (bestIndividual.size() != GENES_PER_INDIVIDUAL) {
        std::cerr << "Warning: Best individual has " << bestIndividual.size()
                << " genes, expected " << GENES_PER_INDIVIDUAL << std::endl;

        // Fix the best individual if it has wrong number of genes
        if (bestIndividual.empty()) {
            // Create a new individual with random genes if best is empty
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

            // Update the first individual in newPop
            newPop[0] = bestIndividual;
        } else if (bestIndividual.size() < GENES_PER_INDIVIDUAL) {
            // Add more genes if needed
            while (bestIndividual.size() < GENES_PER_INDIVIDUAL) {
                // Copy a random gene from the existing ones
                int srcIdx = std::uniform_int_distribution<int>(0, bestIndividual.size() - 1)(rng);
                bestIndividual.push_back(bestIndividual[srcIdx]);
            }
            newPop[0] = bestIndividual;
        } else if (bestIndividual.size() > GENES_PER_INDIVIDUAL) {
            // Remove excess genes
            bestIndividual.resize(GENES_PER_INDIVIDUAL);
            newPop[0] = bestIndividual;
        }
    }

    // Preallocate children (already done in constructor, but ensure capacity)
    Individual child1, child2;
    child1.reserve(GENES_PER_INDIVIDUAL);
    child2.reserve(GENES_PER_INDIVIDUAL);

    // Create a copy of current population with validated gene counts
    std::vector<Individual> validatedPopulation;
    validatedPopulation.reserve(POPULATION_SIZE);
    validatedPopulation.push_back(bestIndividual); // First is the best individual

    // Validate the remaining individuals
    for (int i = 1; i < POPULATION_SIZE && i < population.size(); i++) {
        if (population[i].size() == GENES_PER_INDIVIDUAL) {
            validatedPopulation.push_back(population[i]);
        } else {
            // Create a copy of best individual with slight mutations
            Individual newIndiv = bestIndividual;
            mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 5); // Higher mutation rate
            validatedPopulation.push_back(newIndiv);
        }
    }

    // Fill up to POPULATION_SIZE if needed
    while (validatedPopulation.size() < POPULATION_SIZE) {
        // Copy and mutate the best individual
        Individual newIndiv = bestIndividual;
        mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 10); // Even higher mutation rate
        validatedPopulation.push_back(newIndiv);
    }

    while (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
        // Cast size_t to int for comparison
        int i1 = tournamentSelect(fitnessValues, rng);
        int i2 = tournamentSelect(fitnessValues, rng);

        // Handle potential -1 return from tournamentSelect for empty population
        if (i1 < 0 || i2 < 0 || i1 >= validatedPopulation.size() || i2 >= validatedPopulation.size()) {
            // Use the best individual instead
            i1 = 0;
            i2 = 0;
        }

        // Pass preallocated children to crossover
        onePointCrossover(validatedPopulation[i1], validatedPopulation[i2], child1, child2, rng);

        // Verify children have correct number of genes
        if (child1.size() != GENES_PER_INDIVIDUAL) {
            child1 = validatedPopulation[0]; // Use best as fallback
            mutateIndividual(child1, rng, canvasW, canvasH, MUTATION_RATE * 2);
        }

        if (child2.size() != GENES_PER_INDIVIDUAL) {
            child2 = validatedPopulation[0]; // Use best as fallback
            mutateIndividual(child2, rng, canvasW, canvasH, MUTATION_RATE * 2);
        }

        // Mutate children
        mutateIndividual(child1, rng, canvasW, canvasH, MUTATION_RATE);
        mutateIndividual(child2, rng, canvasW, canvasH, MUTATION_RATE);

        newPop.push_back(child1);
        if (static_cast<int>(newPop.size()) < POPULATION_SIZE) {
            // Cast size_t to int
            newPop.push_back(child2);
        }
    }

    population.swap(newPop);

    // Verify all individuals have the correct number of genes
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
        // Regenerate the entire population except for the best
        std::vector<Individual> regeneratedPop;
        regeneratedPop.reserve(POPULATION_SIZE);
        regeneratedPop.push_back(population[0]); // Keep the best

        for (int i = 1; i < POPULATION_SIZE; i++) {
            Individual newIndiv = population[0]; // Copy the best
            mutateIndividual(newIndiv, rng, canvasW, canvasH, MUTATION_RATE * 5); // Higher mutation rate
            regeneratedPop.push_back(newIndiv);
        }

        population.swap(regeneratedPop);
    }

    // Upload new population to GPU/CPU
    cudaRasterizer.uploadPopulation(population, downscaledTargetPixels); // Pass target

    generationCount++;
}

void Application::render() {
    // Get the best individual's rendered image from the rasterizer (GPU or CPU)
    // This will render at the *current* resolution (which might be downscaled)
    // Assumes bestIndividual is at index 0 of population_ inside CudaRasterizer after uploadPopulation
    std::vector<Pixel> bestImagePixels = cudaRasterizer.getRenderedImage(0); // Get image for index 0
    // Get current resolution directly from rasterizer/cudaRasterizer

    sf::Image bestImage;
    // Get current resolution directly from rasterizer/cudaRasterizer
    unsigned renderW = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getWidth()
                           : rasterizer.getWidth();
    unsigned renderH = cudaRasterizer.isInitialized()
                           ? cudaRasterizer.getHeight()
                           : rasterizer.getHeight();
    // This should be cudaRasterizer.getHeight() or rasterizer.getHeight()


    bestImage.create(renderW, renderH, reinterpret_cast<const sf::Uint8 *>(bestImagePixels.data()));

    // Create an SFML Texture and Sprite to display it
    sf::Texture renderTextureDisplay;
    // Fix: loadFromImage returns bool
    if (!renderTextureDisplay.loadFromImage(bestImage)) {
        std::cerr << "Failed to load rendered image into texture" << std::endl;
        return;
    }
    sf::Sprite spr(renderTextureDisplay);

    // Scale the sprite to fit the window while maintaining aspect ratio
    float scaleX = static_cast<float>(window.getSize().x) / renderW;
    float scaleY = static_cast<float>(window.getSize().y) / renderH;

    float uniformScale = std::min(scaleX, scaleY);
    spr.setScale({uniformScale, uniformScale});

    // Center the sprite in the window
    sf::Vector2f offset{
        (static_cast<float>(window.getSize().x) - renderW * uniformScale) / 2.f, // Cast to float
        (static_cast<float>(window.getSize().y) - renderH * uniformScale) / 2.f // Cast to float
    };
    spr.setPosition(offset);

    // Draw the sprite to the window
    window.clear(sf::Color::Black); // Clear window background
    window.draw(spr);

    // Display the rendered frame
    window.display();
}

