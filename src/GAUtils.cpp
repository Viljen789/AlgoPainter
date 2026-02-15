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
    if (n != static_cast<int>(B.size()) || n == 0) {
        return {A, B};
    }

    std::uniform_int_distribution<int> dist(1, n - 1);
    int cx = dist(rng);

    Individual C, D;
    C.reserve(n);
    D.reserve(n);

    C.insert(C.end(), A.begin(), A.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), B.begin(), B.begin() + cx);
    D.insert(D.end(), A.begin() + cx, A.end());

    return {C, D};
}

void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng) {
    int n = static_cast<int>(A.size());
    int m = static_cast<int>(B.size());

    if (n != m || n == 0 || m == 0) {
        if (n > 0) { C = A; D = A; }
        else if (m > 0) { C = B; D = B; }
        return;
    }

    std::uniform_int_distribution<int> dist(1, n - 1);
    int cx = dist(rng);

    C.clear();
    D.clear();
    C.reserve(n);
    D.reserve(n);

    C.insert(C.end(), A.begin(), A.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), B.begin(), B.begin() + cx);
    D.insert(D.end(), A.begin() + cx, A.end());
}

bool mutateIndividual(Individual &I, std::mt19937 &rng, unsigned canvasW, unsigned canvasH, float mutationRate) {
    bool mutated = false;
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    std::uniform_real_distribution<float> posDelta(-15.0f, 15.0f);
    std::uniform_real_distribution<float> sizeDelta(-5.0f, 5.0f);
    std::uniform_int_distribution<int> colorDelta(-20, 20);

    std::uniform_real_distribution<float> posReplaceX(0.0f, static_cast<float>(canvasW));
    std::uniform_real_distribution<float> posReplaceY(0.0f, static_cast<float>(canvasH));
    std::uniform_real_distribution<float> sizeReplace(5.0f, 50.0f);
    std::uniform_int_distribution<int> colorReplace(0, 255);
    std::uniform_int_distribution<int> alphaReplace(50, 200);

    // Mutation types: 0-3 = small tweaks, 4-6 = large changes
    std::uniform_int_distribution<int> mutationType(0, 6);

    for (auto &gene : I) {
        if (prob(rng) < mutationRate) {
            mutated = true;
            int type = mutationType(rng);

            switch (type) {
                case 0: { // Position tweak
                    sf::Vector2f pos = gene.getPos();
                    pos.x = std::clamp(pos.x + posDelta(rng), 0.0f, static_cast<float>(canvasW));
                    pos.y = std::clamp(pos.y + posDelta(rng), 0.0f, static_cast<float>(canvasH));
                    gene = Gene{gene.getType(), pos, gene.getSize(), gene.getColor()};
                    break;
                }
                case 1: { // Size tweak
                    float newSize = std::clamp(gene.getSize() + sizeDelta(rng), 2.0f, 60.0f);
                    gene = Gene{gene.getType(), gene.getPos(), newSize, gene.getColor()};
                    break;
                }
                case 2: { // Color tweak
                    sf::Color c = gene.getColor();
                    c.r = static_cast<uint8_t>(std::clamp(static_cast<int>(c.r) + colorDelta(rng), 0, 255));
                    c.g = static_cast<uint8_t>(std::clamp(static_cast<int>(c.g) + colorDelta(rng), 0, 255));
                    c.b = static_cast<uint8_t>(std::clamp(static_cast<int>(c.b) + colorDelta(rng), 0, 255));
                    gene = Gene{gene.getType(), gene.getPos(), gene.getSize(), c};
                    break;
                }
                case 3: { // Alpha tweak
                    sf::Color c = gene.getColor();
                    c.a = static_cast<uint8_t>(std::clamp(static_cast<int>(c.a) + colorDelta(rng), 30, 230));
                    gene = Gene{gene.getType(), gene.getPos(), gene.getSize(), c};
                    break;
                }
                case 4: { // Position replace
                    gene = Gene{gene.getType(), {posReplaceX(rng), posReplaceY(rng)}, gene.getSize(), gene.getColor()};
                    break;
                }
                case 5: { // Size replace
                    gene = Gene{gene.getType(), gene.getPos(), sizeReplace(rng), gene.getColor()};
                    break;
                }
                case 6: { // Color replace
                    gene = Gene{
                        gene.getType(), gene.getPos(), gene.getSize(),
                        sf::Color{
                            static_cast<uint8_t>(colorReplace(rng)),
                            static_cast<uint8_t>(colorReplace(rng)),
                            static_cast<uint8_t>(colorReplace(rng)),
                            static_cast<uint8_t>(alphaReplace(rng))
                        }
                    };
                    break;
                }
            }
        }
    }

    return mutated;
}
