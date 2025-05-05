#pragma once
#include <vector>
#include <cstdint>
#include "Gene.h"
#include "Pixel.h" // Include Pixel definition


class Rasterizer {
public:
    Rasterizer(unsigned width, unsigned height);

    Rasterizer() : Rasterizer(1, 1) {
    } // Default constructor

    void clear(const Pixel &clearColor);

    void draw(const Gene &gene);

    const std::vector<Pixel> &data() const; // Get const reference to buffer

    // Add resizing capability
    void resize(unsigned width, unsigned height);

    // Add these getter methods
    unsigned getWidth() const { return width_; }
    unsigned getHeight() const { return height_; }

private:
    unsigned width_, height_;
    std::vector<Pixel> buffer_; // Use std::vector<Pixel>

    // Helper function for drawing a single pixel with alpha blending
    inline void blendPixel(int x, int y, const Pixel &col);


    void drawCircle(int cx, int cy, float radius, const Pixel &col); // Size is float
    void drawRectangle(float x, float y, float w, float h, const Pixel &col); // Pos and size are float
    void drawTriangle(const sf::Vector2f pts[3], const Pixel &col);
};
