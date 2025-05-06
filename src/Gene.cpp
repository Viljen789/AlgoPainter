#include "Gene.h"
#include <iostream>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>


Gene::Gene(Shape type, sf::Vector2f pos, float size, sf::Color color): m_type(type),
                                                                       m_pos(pos),
                                                                       m_size(size),
                                                                       m_color(color) {
}

void Gene::draw(sf::RenderTarget &target) const {
    switch (m_type) {
        case Shape::Circle: {
            sf::CircleShape circle(m_size);
            circle.setPosition(m_pos);
            circle.setFillColor(m_color);
            target.draw(circle);
            break;
        }
        case Shape::Triangle: {
            sf::ConvexShape triangle;
            triangle.setPointCount(3);
            float s = m_size;
            triangle.setPoint(0, {0, 0});
            triangle.setPoint(1, {s, 0});
            triangle.setPoint(2, {s / 2.0f, s * 0.866025f});

            triangle.setPosition(m_pos);
            triangle.setFillColor(m_color);
            target.draw(triangle);
            break;
        }
        case Shape::Square: {
            sf::RectangleShape square({m_size, m_size});
            square.setPosition(m_pos);
            square.setFillColor(m_color);
            target.draw(square);
            break;
        }
        default: {
            std::cout << "Unknown shape" << std::endl;
        };
    }
}

Gene::Shape Gene::getType() const {
    return m_type;
}

float Gene::getSize() const {
    return m_size;
}

sf::Color Gene::getColor() const {
    return m_color;
}

sf::Vector2f Gene::getPos() const {
    return m_pos;
}
