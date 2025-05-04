//
// Created by vilje on 03/05/2025.
//

#ifndef APPLICATION_H
#define APPLICATION_H


#pragma once
#include <random>
#include <SFML/Graphics.hpp>

#include "Gene.h"
#include "Individual.h"

class Application {
public:
    Application();

    void run();

private:
    void processEvents();

    void update();

    void render();

    sf::RenderWindow window;
    sf::RenderTexture renderTexture;
    sf::Image targetImage;
    unsigned canvasW, canvasH;
    Gene testCircle{Gene::Shape::Circle, {100, 100}, 30.f, sf::Color::Red};
    Gene testTri{Gene::Shape::Triangle, {300, 100}, 60.f, sf::Color::Green};
    Gene testSquare{Gene::Shape::Square, {500, 100}, 50.f, sf::Color::Blue};

    static constexpr int POPULATION_SIZE = 100;
    static constexpr int GENES_PER_INDIVIDUAL = 150;
    static constexpr float MUTATION_RATE = 0.05f;
    static constexpr int TOURNAMENT_SIZE = 5;


    std::vector<Individual> population;
    std::vector<float> fitnessValues;
    Individual bestIndividual;


    std::mt19937 rng;
};


#endif //APPLICATION_H
