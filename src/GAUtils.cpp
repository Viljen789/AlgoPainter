#include "GAUtils.h"
#include "Individual.h"
#include <vector>
#include <random>
#include <algorithm>

int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng, int tournamentSize) {
    int populationSize = static_cast<int>(fitnessValues.size());
    std::uniform_int_distribution<int> dist(0, populationSize - 1);
    int bestIndex = dist(rng);
    for (int i = 1; i < tournamentSize; i++) {
        int challengerIndex = dist(rng);
        if (fitnessValues[challengerIndex] > fitnessValues[bestIndex]) bestIndex = challengerIndex;
    }
    return bestIndex;
}

void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng) {
    int n = static_cast<int>(A.size());
    std::uniform_int_distribution<int> dist(1, n - 1);
    int cx = dist(rng);
    C.clear(); D.clear();
    C.insert(C.end(), A.begin(), A.begin() + cx);
    C.insert(C.end(), B.begin() + cx, B.end());
    D.insert(D.end(), B.begin(), B.begin() + cx);
    D.insert(D.end(), A.begin() + cx, A.end());
}

bool mutateIndividual(Individual &I, std::mt19937 &rng, unsigned canvasW, unsigned canvasH, float mutationRate) {
    bool mutated = false;
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> posDelta(-30.0f, 30.0f);
    std::uniform_real_distribution<float> sizeDelta(-15.0f, 15.0f);
    std::uniform_real_distribution<float> rotDelta(-20.0f, 20.0f);
    std::uniform_int_distribution<int> colDelta(-30, 30);
    std::uniform_int_distribution<int> mutType(0, 9);

    for (auto &gene : I) {
        if (prob(rng) < mutationRate) {
            mutated = true;
            int type = mutType(rng);
            sf::Vector2f pos = gene.getPos();
            sf::Vector2f size = gene.getSize();
            float rot = gene.getRotation();
            sf::Color col = gene.getColor();

            switch (type) {
                case 0: pos.x = std::clamp(pos.x + posDelta(rng), 0.0f, (float)canvasW); break;
                case 1: pos.y = std::clamp(pos.y + posDelta(rng), 0.0f, (float)canvasH); break;
                case 2: size.x = std::clamp(size.x + sizeDelta(rng), 2.0f, 200.0f); break;
                case 3: size.y = std::clamp(size.y + sizeDelta(rng), 2.0f, 200.0f); break;
                case 4: rot = std::fmod(rot + rotDelta(rng) + 360.0f, 360.0f); break;
                case 5: col.r = std::clamp((int)col.r + colDelta(rng), 0, 255); break;
                case 6: col.g = std::clamp((int)col.g + colDelta(rng), 0, 255); break;
                case 7: col.b = std::clamp((int)col.b + colDelta(rng), 0, 255); break;
                case 8: col.a = std::clamp((int)col.a + colDelta(rng), 10, 255); break;
                case 9: // Swap order (handled by application loop for better efficiency, or here)
                    break;
            }
            gene = Gene(gene.getType(), pos, size, rot, col);
        }
    }
    return mutated;
}
