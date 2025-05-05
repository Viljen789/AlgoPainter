//
// Created by vilje on 04/05/2025.
//
#include "GAUtils.h"
#include "Individual.h"
// Remove circular include
// #include "Application.h" 
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm> // For std::min/max
#include <iostream> // For potential debugging output

// Update implementation to use the parameter
int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng, int tournamentSize) {
    // Ensure tournament size doesn't exceed population size
    int populationSize = static_cast<int>(fitnessValues.size()); // Cast size_t to int
    tournamentSize = std::min(tournamentSize, populationSize);
    if (tournamentSize <= 0) {
        // Handle empty or tiny population - might return 0 or an error index
        if (populationSize > 0) return 0; // Return the first index if population exists but is tiny
        return -1; // Indicate error if population is empty
    }

    std::uniform_int_distribution<int> dist(0, populationSize - 1); // Distribute over valid indices
    int bestIndex = dist(rng); // Pick initial random contestant

    for (int i = 1; i < tournamentSize; i++) {
        int challengerIndex = dist(rng);
        // In case of ties, keep the older individual (the one picked first)
        if (fitnessValues[challengerIndex] > fitnessValues[bestIndex]) {
            bestIndex = challengerIndex;
        }
    }
    return bestIndex;
}

// Original crossover that creates new individuals (kept for reference, not used in Application)
std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng) {
    // Note: This version allocates new individuals. The optimized version is preferred.
    int n = static_cast<int>(A.size()); // Cast size_t to int
    if (n != B.size() || n == 0) {
        // Handle error or mismatch
        std::cerr << "Crossover size mismatch or empty individuals!" << std::endl;
        return {A, B}; // Return copies in case of error
    }

    std::uniform_int_distribution<int> dist(1, n - 1); // Crossover point excludes 0 and n
    int cx = dist(rng);

    Individual C;
    C.reserve(n);
    Individual D;
    D.reserve(n);

    C.insert(C.end(), A.begin(), A.begin() + cx);
    D.insert(D.end(), B.begin(), B.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), A.begin() + cx, A.end());
    return {C, D};
}

// Optimized version that reuses existing individuals to avoid allocations
void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng) {
    int n = static_cast<int>(A.size()); // Cast size_t to int
    int m = static_cast<int>(B.size()); // Cast size_t to int

    if (n != m || n == 0 || m == 0) {
        // Handle error or mismatch
        std::cerr << "Optimized crossover size mismatch or empty individuals! A:" << n << " B:" << m << std::endl;

        // If either parent is valid, copy it to both children
        if (n > 0) {
            C = A;
            D = A;
        } else if (m > 0) {
            C = B;
            D = B;
        } else {
            // Both parents empty - leave children empty
            C.clear();
            D.clear();
        }
        return;
    }

    // Ensure C and D have enough capacity
    C.reserve(n); // Should already be reserved in Application, but safety check
    D.reserve(n); // Should already be reserved in Application, but safety check

    // Always use a valid crossover point
    std::uniform_int_distribution<int> dist(1, n - 1); // Crossover point excludes 0 and n
    int cx = n > 1 ? dist(rng) : 0; // If n is 1, use 0 as crossover point

    // Clear the output individuals but maintain their capacity
    C.clear();
    D.clear();

    // Perform the crossover
    C.insert(C.end(), A.begin(), A.begin() + cx);
    D.insert(D.end(), B.begin(), B.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), A.begin() + cx, A.end());

    // Verify the children have the expected size
    if (C.size() != n || D.size() != n) {
        std::cerr << "Warning: After crossover, child sizes don't match parent size! "
                << "A:" << n << " B:" << m << " C:" << C.size() << " D:" << D.size() << std::endl;

        // Fix the children if necessary
        if (C.size() != n) {
            C = A; // Fallback to copying parent A
        }
        if (D.size() != n) {
            D = B; // Fallback to copying parent B
        }
    }
}

// Mutation function - returns true if a gene was mutated (for future potential gene count changes)
// Added parameter for mutation rate to make it configurable
bool mutateIndividual(Individual &I, std::mt19937 &rng, unsigned canvasW, unsigned canvasH, float mutationRate) {
    bool mutated = false; // Track if any gene was mutated
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> posDeltaDist(-20.0f, 20.0f); // Small position perturbation
    std::uniform_real_distribution<float> sizeDeltaDist(-10.0f, 10.0f); // Small size perturbation

    // Define distributions outside the loop for efficiency
    std::uniform_real_distribution<float> posReplaceDistX(0.0f, static_cast<float>(canvasW));
    // Use float literal and cast
    std::uniform_real_distribution<float> posReplaceDistY(0.0f, static_cast<float>(canvasH));
    // Use float literal and cast
    std::uniform_real_distribution<float> sizeReplaceDist(5.0f, 50.0f); // Use float literals
    std::uniform_int_distribution<int> colDeltaDist(-20, 20);
    std::uniform_int_distribution<int> colReplaceDist(0, 255);


    // Mutation operators:
    // 0: Perturb Position
    // 1: Perturb Size
    // 2: Perturb Color
    // 3: Randomize Shape
    // 4: Randomize Position
    // 5: Randomize Size
    // 6: Randomize Color

    std::uniform_int_distribution<int> mutationTypeDist(0, 6); // 7 different mutation types

    for (auto &gene: I) {
        if (prob(rng) < mutationRate) {
            mutated = true;
            int mutationType = mutationTypeDist(rng);

            switch (mutationType) {
                case 0: {
                    // Perturb position
                    sf::Vector2f currentPos = gene.getPos();
                    float newX = currentPos.x + posDeltaDist(rng);
                    float newY = currentPos.y + posDeltaDist(rng);
                    // Clamp position to canvas bounds
                    newX = std::max(0.0f, std::min(static_cast<float>(canvasW), newX)); // Cast to float for comparison
                    newY = std::max(0.0f, std::min(static_cast<float>(canvasH), newY)); // Cast to float for comparison
                    gene = Gene{gene.getType(), {newX, newY}, gene.getSize(), gene.getColor()};
                    break;
                }
                case 1: {
                    // Perturb size
                    float currentSize = gene.getSize();
                    float newSize = currentSize + sizeDeltaDist(rng);
                    // Clamp size to reasonable bounds (e.g., min 1, max 100)
                    newSize = std::max(1.0f, std::min(100.0f, newSize)); // Use float literal
                    gene = Gene{gene.getType(), gene.getPos(), newSize, gene.getColor()};
                    break;
                }
                case 2: {
                    // Perturb color
                    sf::Color currentColor = gene.getColor();
                    int newR = std::max(0, std::min(255, static_cast<int>(currentColor.r) + colDeltaDist(rng)));
                    // Cast to int for arithmetic
                    int newG = std::max(0, std::min(255, static_cast<int>(currentColor.g) + colDeltaDist(rng)));
                    // Cast to int for arithmetic
                    int newB = std::max(0, std::min(255, static_cast<int>(currentColor.b) + colDeltaDist(rng)));
                    // Cast to int for arithmetic
                    // Keep alpha the same or also perturb
                    gene = Gene{
                        gene.getType(), gene.getPos(), gene.getSize(),
                        sf::Color{
                            static_cast<uint8_t>(newR), static_cast<uint8_t>(newG), static_cast<uint8_t>(newB),
                            currentColor.a
                        } // Cast back to uint8_t
                    };
                    break;
                }
                case 3: {
                    // Randomize shape type
                    gene = Gene{
                        static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng)),
                        gene.getPos(), gene.getSize(), gene.getColor()
                    };
                    break;
                }
                case 4: {
                    // Randomize position
                    gene = Gene{
                        gene.getType(), {posReplaceDistX(rng), posReplaceDistY(rng)}, gene.getSize(), gene.getColor()
                    };
                    break;
                }
                case 5: {
                    // Randomize size
                    gene = Gene{gene.getType(), gene.getPos(), sizeReplaceDist(rng), gene.getColor()};
                    break;
                }
                case 6: {
                    // Randomize color (keep alpha for now)
                    sf::Color currentColor = gene.getColor();
                    gene = Gene{
                        gene.getType(), gene.getPos(), gene.getSize(),
                        sf::Color{
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            currentColor.a // Keep alpha
                            // static_cast<uint8_t>(colReplaceDist(rng)) // Randomize alpha
                        }
                    };
                    break;
                }
            }
        };
    }
    // TODO: Add gene addition/removal mutation here with a separate low probability
    // This would require changing the Individual (vector<Gene>) size,
    // which complicates the fixed-size population and CUDA data structure.
    // For now, we stick to fixed gene count.

    return mutated; // Return true if any mutation occurred
}

