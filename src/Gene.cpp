//
// Created by vilje on 03/05/2025.
//

#include "Gene.h"
#include <iostream>
#include <SFML/Graphics/CircleShape.hpp> // Include specific shape headers for draw method
#include <SFML/Graphics/ConvexShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>


Gene::Gene(Shape type, sf::Vector2f pos, float size, sf::Color color): m_type(type),
                                                                       m_pos(pos),
                                                                       m_size(size),
                                                                       m_color(color) {
    // std::cout << "Created gene of size: " << m_size << " at (" << m_pos.x << ", " << m_pos.y << ")" << std::endl;
}

void Gene::draw(sf::RenderTarget &target) const {
    // SFML drawing happens here, used for the final best image render
    switch (m_type) {
        case Shape::Circle: {
            // SFML CircleShape uses radius, not diameter for size
            sf::CircleShape circle(m_size); // Assuming m_size is radius now based on SFML
            circle.setPosition(m_pos);
            circle.setFillColor(m_color);
            target.draw(circle);
            break;
        }
        case Shape::Triangle: {
            sf::ConvexShape triangle;
            triangle.setPointCount(3);
            // Assuming the size is the side length for an equilateral triangle for SFML display
            // Note: This might differ from how size is used in the GPU kernel's AABB/barycentric math
            float s = m_size; // Use m_size as side length
            // Equilateral triangle points relative to origin, then translated by m_pos
            triangle.setPoint(0, {0, 0}); // Top point
            triangle.setPoint(1, {s, 0}); // Bottom right
            triangle.setPoint(2, {s / 2.0f, s * 0.866025f}); // Bottom left (sqrt(3)/2 * s)

            triangle.setPosition(m_pos);
            triangle.setFillColor(m_color);
            target.draw(triangle);
            break;
        }
        case Shape::Square: {
            // Assuming size is the side length
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
    // Changed return type
    return m_size;
}

sf::Color Gene::getColor() const {
    return m_color;
}

sf::Vector2f Gene::getPos() const {
    return m_pos;
}
