#pragma once
#include <SFML/Graphics.hpp>

#ifndef GENE_H
#define GENE_H

class Gene {
  public:
    enum class Shape { Circle, Rectangle, Triangle };

    Gene(Shape type, sf::Vector2f pos, sf::Vector2f size, float rotation, sf::Color color);

    Gene() : m_type(Shape::Circle), m_pos({0, 0}), m_size({0, 0}), m_rotation(0), m_color({0, 0, 0, 0}) {}

    void draw(sf::RenderTarget& target) const;

    Shape getType() const;
    sf::Vector2f getSize() const;
    float getRotation() const;
    sf::Color getColor() const;
    sf::Vector2f getPos() const;

  private:
    Shape m_type;
    sf::Vector2f m_pos;
    sf::Vector2f m_size; // width, height (or radius for circle)
    float m_rotation;    // in degrees
    sf::Color m_color;
};

#endif // GENE_H
