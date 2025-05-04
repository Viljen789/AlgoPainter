//
// Created by vilje on 04/05/2025.
//

#pragma once
#include "Individual.h"
#include "Application.h"
#include <vector>
#include <random>

int tournamentSelect(const std::vector<float> &fitnessValues, std::mt19937 &rng);

std::pair<Individual, Individual> onePointCrossover(const Individual &A, const Individual &B, std::mt19937 &rng);

void mutateIndividual(Individual &individual, std::mt19937 &rng, unsigned canvasW, unsigned canvasH);

static constexpr int TOURNAMENT_SIZE = 5;
static constexpr float MUTATION_RATE = 0.05f;
