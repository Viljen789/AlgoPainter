//
// Created by vilje on 03/05/2025.
//

#include <iostream>
#include <cstdint>
#include <iosfwd>
#include <random>

#include "Application.h"
#include "Fitness.h"
#include "GAUtils.h"
#include "Gene.h"


Application::Application()
    : window(sf::VideoMode({800, 600}), "EvoArt") // width, height, title
{
    if (!targetImage.loadFromFile("../assets/Target.jpg")) {
        throw std::runtime_error("Failed to load target.png");
    }
    auto ts = targetImage.getSize();
    canvasW = ts.x;
    canvasH = ts.y;
    renderTexture = sf::RenderTexture{ts};
    std::cout << "TargetImage size: " << ts.x << ", " << ts.y << std::endl;;
    std::uniform_real_distribution<float> xDist(0.0f, canvasW), yDist(0.0f, canvasH), sizeDist(5, 50);
    std::uniform_int_distribution<int> colorDist(0, 255);

    population.reserve(POPULATION_SIZE);
    fitnessValues.resize(POPULATION_SIZE);
    population.push_back(bestIndividual);
    for (int k = 1; k < POPULATION_SIZE; k++) {
        Individual indiv;
        indiv.reserve(GENES_PER_INDIVIDUAL);
        for (int i = 0; i < GENES_PER_INDIVIDUAL; i++) {
            auto shape = static_cast<Gene::Shape>(std::uniform_int_distribution<int>(0, 2)(rng));
            sf::Vector2f pos{xDist(rng), yDist(rng)};
            float size = sizeDist(rng);
            sf::Color color{
                static_cast<std::uint8_t>(colorDist(rng)), static_cast<std::uint8_t>(colorDist(rng)),
                static_cast<std::uint8_t>(colorDist(rng)), 128
            };
            indiv.emplace_back(shape, pos, size, color);
        }
        population.push_back(std::move(indiv));
    }
    std::cout << population.size() << std::endl;
}


void Application::run() {
    while (window.isOpen()) {
        processEvents();
        update();
        render();
    }
}

void Application::processEvents() {
    while (auto event = window.pollEvent()) {
        if (event->is<sf::Event::Closed>()) {
            window.close();
        }
    }
}

void Application::update() {
    for (int k = 0; k < POPULATION_SIZE; k++) {
        renderTexture.clear();
        for (auto &gene: population[k])gene.draw(renderTexture);
        renderTexture.display();

        sf::Image img = renderTexture.getTexture().copyToImage();
        fitnessValues[k] = computeFitness(img, targetImage);
    }
    int bestIndex = 0;
    float bestFit = fitnessValues[bestIndex];
    for (int k = 1; k < POPULATION_SIZE; k++) {
        if (fitnessValues[k] > bestFit) {
            bestFit = fitnessValues[k];
            bestIndex = k;
        }
    }
    bestIndividual = population[bestIndex];
    std::cout << "Gen best fitness = " << bestFit << std::endl;
    std::vector<Individual> newPop;
    newPop.reserve(POPULATION_SIZE);
    newPop.push_back(bestIndividual);

    while (int(newPop.size()) < POPULATION_SIZE) {
        int i1 = tournamentSelect(fitnessValues, rng);
        int i2 = tournamentSelect(fitnessValues, rng);
        auto [child1, child2] = onePointCrossover(population[i1], population[i2], rng);
        mutateIndividual(child1, rng, canvasW, canvasH);
        mutateIndividual(child2, rng, canvasW, canvasH);
        newPop.push_back(std::move(child1));
        if (int(newPop.size()) < POPULATION_SIZE) {
            newPop.push_back(std::move(child2));
        }
    }
    population.swap(newPop);
}

void Application::render() {
    renderTexture.clear(sf::Color::Black);

    for (auto i: bestIndividual) {
        i.draw(renderTexture);
    }
    renderTexture.display();

    sf::Sprite spr(renderTexture.getTexture());

    float scaleX = static_cast<float>(window.getSize().x) / renderTexture.getSize().x;
    float scaleY = static_cast<float>(window.getSize().y) / renderTexture.getSize().y;

    float uniform = std::min(scaleX, scaleY);
    spr.setScale({uniform, uniform});
    sf::Vector2f offset{
        (window.getSize().x - renderTexture.getSize().x * uniform) / 2.f,
        (window.getSize().y - renderTexture.getSize().y * uniform) / 2.f
    };
    spr.setPosition(offset);
    window.clear(sf::Color::Black);
    window.draw(spr);
    window.display();
}

