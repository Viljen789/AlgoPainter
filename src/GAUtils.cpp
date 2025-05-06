#include "GAUtils.h"
#include "Individual.h"
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iostream>

int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng, int tournamentSize) {
    int populationSize = static_cast<int>(fitnessValues.size());
    tournamentSize = std::min(tournamentSize, populationSize);
    if (tournamentSize <= 0) {
        if (populationSize > 0) return 0;
        return -1;
    }

    std::uniform_int_distribution<int> dist(0, populationSize - 1);
    int bestIndex = dist(rng);

    for (int i = 1; i < tournamentSize; i++) {
        int challengerIndex = dist(rng);
        if (fitnessValues[challengerIndex] > fitnessValues[bestIndex]) {
            bestIndex = challengerIndex;
        }
    }
    return bestIndex;
}

std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng) {
    int n = static_cast<int>(A.size());
    if (n != B.size() || n == 0) {
        std::cerr << "Crossover size mismatch or empty individuals!" << std::endl;
        return {A, B};
    }

    std::uniform_int_distribution<int> dist(1, n - 1);
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

void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng) {
    int n = static_cast<int>(A.size());
    int m = static_cast<int>(B.size());

    if (n != m || n == 0 || m == 0) {
        std::cerr << "Optimized crossover size mismatch or empty individuals! A:" << n << " B:" << m << std::endl;

        if (n > 0) {
            C = A;
            D = A;
        } else if (m > 0) {
            C = B;
            D = B;
        } else {
            C.clear();
            D.clear();
        }
        return;
    }

    C.reserve(n);
    D.reserve(n);

    std::uniform_int_distribution<int> dist(1, n - 1);
    int cx = n > 1 ? dist(rng) : 0;

    C.clear();
    D.clear();

    C.insert(C.end(), A.begin(), A.begin() + cx);
    D.insert(D.end(), B.begin(), B.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), A.begin() + cx, A.end());

    if (C.size() != n || D.size() != n) {
        std::cerr << "Warning: After crossover, child sizes don't match parent size! "
                << "A:" << n << " B:" << m << " C:" << C.size() << " D:" << D.size() << std::endl;

        if (C.size() != n) {
            C = A;
        }
        if (D.size() != n) {
            D = B;
        }
    }
}

bool mutateIndividual(Individual &I, std::mt19937 &rng, unsigned canvasW, unsigned canvasH, float mutationRate) {
    bool mutated = false;
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> posDeltaDist(-20.0f, 20.0f);
    std::uniform_real_distribution<float> sizeDeltaDist(-10.0f, 10.0f);

    std::uniform_real_distribution<float> posReplaceDistX(0.0f, static_cast<float>(canvasW));
    std::uniform_real_distribution<float> posReplaceDistY(0.0f, static_cast<float>(canvasH));
    std::uniform_real_distribution<float> sizeReplaceDist(5.0f, 50.0f);
    std::uniform_int_distribution<int> colDeltaDist(-20, 20);
    std::uniform_int_distribution<int> colReplaceDist(0, 255);

    std::uniform_int_distribution<int> mutationTypeDist(0, 6);

    for (auto &gene: I) {
        if (prob(rng) < mutationRate) {
            mutated = true;
            int mutationType = mutationTypeDist(rng);

            switch (mutationType) {
                case 0: {
                    sf::Vector2f currentPos = gene.getPos();
                    float newX = currentPos.x + posDeltaDist(rng);
                    float newY = currentPos.y + posDeltaDist(rng);
                    newX = std::max(0.0f, std::min(static_cast<float>(canvasW), newX));
                    newY = std::max(0.0f, std::min(static_cast<float>(canvasH), newY));
                    gene = Gene{gene.getType(), {newX, newY}, gene.getSize(), gene.getColor()};
                    break;
                }
                case 1: {
                    float currentSize = gene.getSize();
                    float newSize = currentSize + sizeDeltaDist(rng);
                    newSize = std::max(1.0f, std::min(100.0f, newSize));
                    gene = Gene{gene.getType(), gene.getPos(), newSize, gene.getColor()};
                    break;
                }
                case 2: {
                    sf::Color currentColor = gene.getColor();
                    int newR = std::max(0, std::min(255, static_cast<int>(currentColor.r) + colDeltaDist(rng)));
                    int newG = std::max(0, std::min(255, static_cast<int>(currentColor.g) + colDeltaDist(rng)));
                    int newB = std::max(0, std::min(255, static_cast<int>(currentColor.b) + colDeltaDist(rng)));
                    gene = Gene{
                        gene.getType(), gene.getPos(), gene.getSize(),
                        sf::Color{
                            static_cast<uint8_t>(newR), static_cast<uint8_t>(newG), static_cast<uint8_t>(newB),
                            currentColor.a
                        }
                    };
                    break;
                }
                case 3: {
                    gene = Gene{
                        static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng)),
                        gene.getPos(), gene.getSize(), gene.getColor()
                    };
                    break;
                }
                case 4: {
                    gene = Gene{
                        gene.getType(), {posReplaceDistX(rng), posReplaceDistY(rng)}, gene.getSize(), gene.getColor()
                    };
                    break;
                }
                case 5: {
                    gene = Gene{gene.getType(), gene.getPos(), sizeReplaceDist(rng), gene.getColor()};
                    break;
                }
                case 6: {
                    sf::Color currentColor = gene.getColor();
                    gene = Gene{
                        gene.getType(), gene.getPos(), gene.getSize(),
                        sf::Color{
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            static_cast<uint8_t>(colReplaceDist(rng)),
                            currentColor.a
                        }
                    };
                    break;
                }
            }
        };
    }

    return mutated;
}
