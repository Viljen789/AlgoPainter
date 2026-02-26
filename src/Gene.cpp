#include "Gene.h"

#include <cmath>

Gene::Gene(Shape type, sf::Vector2f pos, sf::Vector2f size, float rotation, sf::Color color)
    : m_type(type), m_pos(pos), m_size(size), m_rotation(rotation), m_color(color) {}

void Gene::draw(sf::RenderTarget& target) const {
    if (m_type == Shape::Circle) {
        sf::CircleShape shape(m_size.x);
        shape.setOrigin({m_size.x, m_size.x});
        shape.setPosition(m_pos);
        shape.setRotation(sf::degrees(m_rotation));
        shape.setFillColor(m_color);
        target.draw(shape);
    } else if (m_type == Shape::Rectangle) {
        sf::RectangleShape shape(m_size);
        shape.setOrigin(m_size / 2.0f);
        shape.setPosition(m_pos);
        shape.setRotation(sf::degrees(m_rotation));
        shape.setFillColor(m_color);
        target.draw(shape);
    } else if (m_type == Shape::Triangle) {
        sf::CircleShape shape(m_size.x, 3);
        shape.setOrigin({m_size.x, m_size.x});
        shape.setPosition(m_pos);
        shape.setRotation(sf::degrees(m_rotation));
        shape.setFillColor(m_color);
        target.draw(shape);
    }
}

Gene::Shape Gene::getType() const {
    return m_type;
}
sf::Vector2f Gene::getSize() const {
    return m_size;
}
float Gene::getRotation() const {
    return m_rotation;
}
sf::Color Gene::getColor() const {
    return m_color;
}
sf::Vector2f Gene::getPos() const {
    return m_pos;
}
