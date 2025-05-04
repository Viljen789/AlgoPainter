//
// Created by vilje on 04/05/2025.
//
#include "Application.h"
#include "GAUtils.h"
#include "Individual.h"
#include <vector>
#include <cstdint>
#include <iosfwd>
#include <random>


int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng) {
  std::uniform_int_distribution<int> dist(0, fitnessValues.size() - 1);
  int best = dist(rng);
  for (int i = 0; i < TOURNAMENT_SIZE; i++) {
    int challenger = dist(rng);
    if (fitnessValues[challenger] > fitnessValues[best]) {
      best = challenger;
    }
  }
  return best;
}

std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng) {
  int n = A.size();
  std::uniform_int_distribution<int> dist(0, n - 1);
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

void mutateIndividual(Individual &I, std::mt19937 &rng, unsigned canvasW, unsigned canvasH) {
  float mutationRate = MUTATION_RATE;
  std::uniform_real_distribution<float> prob(0, 1);
  std::uniform_real_distribution<float> posDistX(0, float(canvasW)), posDistY(0, float(canvasH)), sizeDist(5, 50);
  std::uniform_int_distribution<int> colDist(0, 255);

  for (auto &gene: I) {
    if (prob(rng) < mutationRate) {
      switch (std::uniform_int_distribution<int>(0, 3)(rng)) {
        case 0: //position
          gene = Gene{
            gene.getType(), {posDistX(rng), posDistY(rng)}, static_cast<float>(gene.getSize()), gene.getColor()
          };
          break;
        case 1: // size
          gene = Gene{gene.getType(), gene.getPos(), sizeDist(rng), gene.getColor()};
          break;
        case 2: // color
          gene = Gene{
            gene.getType(), gene.getPos(), static_cast<float>(gene.getSize()), sf::Color{
              static_cast<std::uint8_t>(colDist(rng)), static_cast<std::uint8_t>(colDist(rng)),
              static_cast<std::uint8_t>(colDist(rng)), gene.getColor().a
            }
          };
          break;
        case 3: // shape type
          gene = Gene{
            static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng)),
            gene.getPos(), static_cast<float>(gene.getSize()), gene.getColor()
          };
      }
    };
  }
}
