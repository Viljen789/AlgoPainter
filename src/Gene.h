#pragma once
#include <SFML/Graphics.hpp>

#ifndef GENE_H
#define GENE_H

class Gene {
public:
    enum class Shape { Circle, Triangle, Square };

    Gene(Shape type, sf::Vector2f pos, float size, sf::Color color);

    Gene() : m_type(Shape::Circle), m_pos({0, 0}), m_size(0), m_color({0, 0, 0, 0}) {
    }

    void draw(sf::RenderTarget &target) const;

    Shape getType() const;

    float getSize() const;

    sf::Color getColor() const;

    sf::Vector2f getPos() const;

private:
    Shape m_type;
    sf::Vector2f m_pos;
    float m_size;
    sf::Color m_color;
};

#endif //GENE_H
