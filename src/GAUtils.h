//
// Created by vilje on 04/05/2025.
//

#pragma once
#include "Individual.h"
#include <vector>
#include <random>
#include <SFML/Graphics.hpp> // Need sf::Vector2f and sf::Color definitions

// Include Application.h for GA constants
#include "Application.h"

// TOURNAMENT_SIZE and MUTATION_RATE are now defined in Application.h

// Add tournamentSize parameter with default value
int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng, int tournamentSize = 5);

// Original crossover that creates new individuals (kept for reference, not used in Application)
std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng);

// Optimized crossover that reuses existing individuals
void onePointCrossover(const Individual &A, const Individual &B, Individual &C, Individual &D, std::mt19937 &rng);

// Mutation function - returns true if a gene was mutated (for future potential gene count changes)
// Added parameter for mutation rate to make it configurable
bool mutateIndividual(Individual &individual, std::mt19937 &rng, unsigned canvasW, unsigned canvasH,
                      float mutationRate);

