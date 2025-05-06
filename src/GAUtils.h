#pragma once
#include "Individual.h"
#include <vector>
#include <random>
#include <SFML/Graphics.hpp>

#include "Application.h"

int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng, int tournamentSize = 5);

std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng);

void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng);

bool mutateIndividual(Individual &individual, std::mt19937 &rng, unsigned canvasW, unsigned canvasH,
                      float mutationRate);
