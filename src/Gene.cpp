//
// Created by vilje on 03/05/2025.
//

#include "Gene.h"
#include <iostream>


Gene::Gene(Shape type, sf::Vector2f pos, float size, sf::Color color): m_type(type),
                                                                       m_pos(pos),
                                                                       m_size(size),
                                                                       m_color(color) {
    // std::cout << "Created gene of size: " << m_size << " at (" << m_pos.x << ", " << m_pos.y << ")" << std::endl;
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
            triangle.setPoint(0, {0, 0});
            triangle.setPoint(1, {m_size, 0});
            triangle.setPoint(2, {m_size / 2, m_size});
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

int Gene::getSize() const {
    return m_size;
}

sf::Color Gene::getColor() const {
    return m_color;
}

sf::Vector2f Gene::getPos() const {
    return m_pos;
}
