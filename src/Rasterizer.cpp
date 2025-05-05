//
// Created by vilje on 04/05/2025.
//
#include "Rasterizer.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream> // For debugging/errors

// Renaming variables to avoid conflict with C standard library functions
#define min_x_bound min_x_bound_rasterizer
#define max_x_bound max_x_bound_rasterizer
#define min_y_bound min_y_bound_rasterizer
#define max_y_bound max_y_bound_rasterizer


Rasterizer::Rasterizer(unsigned w, unsigned h)
    : width_(w), height_(h), buffer_(static_cast<size_t>(w) * h) {
    // Cast to size_t
}

void Rasterizer::resize(unsigned w, unsigned h) {
    width_ = w;
    height_ = h;
    buffer_.resize(static_cast<size_t>(w) * h); // Cast to size_t
}

void Rasterizer::clear(const Pixel &c) {
    std::fill(buffer_.begin(), buffer_.end(), c);
}

const std::vector<Pixel> &Rasterizer::data() const {
    return buffer_;
}

// Helper function for drawing a single pixel with alpha blending and bounds checking
inline void Rasterizer::blendPixel(int x, int y, const Pixel &col) {
    if (x >= 0 && static_cast<unsigned>(x) < width_ && y >= 0 && static_cast<unsigned>(y) < height_) {
        // Cast x, y to unsigned for comparison
        Pixel &p = buffer_[static_cast<size_t>(y) * width_ + x]; // Cast y to size_t for index calculation
        uint16_t alpha = col.a;
        if (alpha == 0) return;
        if (alpha == 255) {
            p = col;
            return;
        }
        uint16_t invAlpha = 255 - alpha;

        // Using unsigned int for intermediate calculations to avoid overflow before shifting
        unsigned int r = (unsigned int) p.r * invAlpha + (unsigned int) col.r * alpha;
        unsigned int g = (unsigned int) p.g * invAlpha + (unsigned int) col.g * alpha;
        unsigned int b = (unsigned int) p.b * invAlpha + (unsigned int) col.b * alpha;

        p.r = static_cast<uint8_t>(r >> 8);
        p.g = static_cast<uint8_t>(g >> 8);
        p.b = static_cast<uint8_t>(b >> 8);
        // Alpha channel blending on canvas is often skipped for simplicity
        // p.a = static_cast<uint8_t>((p.a * invAlpha + col.a * alpha) >> 8);
    }
}


void Rasterizer::draw(const Gene &g) {
    Pixel col{g.getColor().r, g.getColor().g, g.getColor().b, g.getColor().a};
    switch (g.getType()) {
        case Gene::Shape::Circle:
            drawCircle(static_cast<int>(g.getPos().x), static_cast<int>(g.getPos().y), g.getSize(), col);
        // Pass size as float
            break;
        case Gene::Shape::Square: {
            // SFML uses top-left corner, but GA genes use center pos.
            // Adjust to top-left for rectangle drawing.
            drawRectangle(g.getPos().x - g.getSize() / 2.0f, g.getPos().y - g.getSize() / 2.0f, g.getSize(),
                          g.getSize(), col); // Pass pos/size as float
            break;
        }
        case Gene::Shape::Triangle: {
            // Use gene.pos as the *top* point for consistency with SFML draw?
            // Or use gene.pos as center and calculate points?
            // The SFML draw method used gene.pos as the top point and size as side length.
            // Let's try to match that for CPU rasterizer drawing for visual consistency.
            sf::Vector2f p0 = g.getPos(); // Top point
            float s = g.getSize(); // Size is side length

            // Define points relative to the top point (p0), then add p0 later
            sf::Vector2f points_relative[3] = {
                {0.0f, 0.0f}, // Top point (relative to p0)
                {s, 0.0f}, // Bottom-right if 0,0 is top-left and size is side
                {s / 2.0f, s * 0.866025f} // Bottom-left point for equilateral triangle
            };
            // This creates a triangle pointing *down* with p0 as the top vertex.

            sf::Vector2f points_absolute[3];
            points_absolute[0] = p0 + points_relative[0];
            points_absolute[1] = p0 + points_relative[1];
            points_absolute[2] = p0 + points_relative[2];

            drawTriangle(points_absolute, col); // Pass the absolute points
            break;
        }
    }
}

void Rasterizer::drawCircle(int cx, int cy, float radius, const Pixel &col) {
    int r = static_cast<int>(radius); // Convert float radius to int
    int r2 = r * r;
    int y_start = std::max(0, cy - r);
    int y_end = std::min(static_cast<int>(height_), cy + r + 1); // Cast height_ to int

    for (int y = y_start; y < y_end; ++y) {
        int dy = y - cy;
        int dx_squared = r2 - dy * dy;
        if (dx_squared >= 0) {
            int dx = static_cast<int>(std::sqrt(static_cast<float>(dx_squared))); // Cast to float for sqrt
            int x_start = std::max(0, cx - dx);
            int x_end = std::min(static_cast<int>(width_), cx + dx + 1); // Cast width_ to int

            for (int x = x_start; x < x_end; ++x) {
                blendPixel(x, y, col);
            }
        }
    }
}

void Rasterizer::drawRectangle(float x_f, float y_f, float w_f, float h_f, const Pixel &col) {
    // Convert float bounds to integer bounds, flooring min and ceiling max
    int x0 = std::max(0, static_cast<int>(std::floor(x_f)));
    int y0 = std::max(0, static_cast<int>(std::floor(y_f)));
    int x1 = std::min(static_cast<int>(width_), static_cast<int>(std::ceil(x_f + w_f))); // Cast width_ to int
    int y1 = std::min(static_cast<int>(height_), static_cast<int>(std::ceil(y_f + h_f))); // Cast height_ to int

    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            blendPixel(xx, yy, col);
        }
    }
}

void Rasterizer::drawTriangle(const sf::Vector2f pts[3], const Pixel &col) {
    // Use temporary variables for min/max bounds to avoid conflict with y0/y1 functions
    float minXf = std::min({pts[0].x, pts[1].x, pts[2].x}, [](float a, float b) { return a < b; });
    // Use lambda for initializer list min/max with float
    float maxXf = std::max({pts[0].x, pts[1].x, pts[2].x}, [](float a, float b) { return a < b; }); // Use lambda
    float minYf = std::min({pts[0].y, pts[1].y, pts[2].y}, [](float a, float b) { return a < b; }); // Use lambda
    float maxYf = std::max({pts[0].y, pts[1].y, pts[2].y}, [](float a, float b) { return a < b; }); // Use lambda


    // Convert float bounds to integer bounds, flooring min and ceiling max
    int x_start = std::max(0, static_cast<int>(std::floor(minXf)));
    int x_end = std::min(static_cast<int>(width_), static_cast<int>(std::ceil(maxXf))); // Cast width_ to int
    int y_start = std::max(0, static_cast<int>(std::floor(minYf)));
    int y_end = std::min(static_cast<int>(height_), static_cast<int>(std::ceil(maxYf))); // Cast height_ to int


    // Pre-compute edge equations for faster point-in-triangle test
    // Use double for intermediate calculations to reduce precision issues if long long is problematic
    // auto edge = [&](const sf::Vector2f &a, const sf::Vector2f &b, double x, double y) {
    //     return (double)(b.x - a.x) * (y - a.y) - (double)(b.y - a.y) * (x - a.x);
    // };
    // Or stick with long long, ensuring casting:
    auto edge = [&](const sf::Vector2f &a, const sf::Vector2f &b, float x, float y) {
        return (long long) ((double) b.x - a.x) * ((double) y - a.y) - (long long) ((double) b.y - a.y) * (
                   (double) x - a.x); // Cast to double for intermediate, then long long
    };


    // Pre-compute constants for each edge
    // Use double for edge constants to maintain precision if intermediate double is used
    // Or stick with long long, being careful with casts
    long long edge_const[3][3]; // [edge][a,b,c] where ax + by + c = 0 is the edge equation
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        edge_const[i][0] = static_cast<long long>(pts[j].y - pts[i].y); // a = y2 - y1 (cast float to long long)
        edge_const[i][1] = static_cast<long long>(pts[i].x - pts[j].x); // b = x1 - x2 (cast float to long long)
        edge_const[i][2] = static_cast<long long>((double) pts[j].x * pts[i].y - (double) pts[i].x * pts[j].y);
        // c = x2*y1 - x1*y2 (use double for multiplication, then cast to long long)
    }

    long long area = edge(pts[0], pts[1], pts[2].x, pts[2].y);
    // Handle degenerate triangle (area is 0)
    if (area == 0) return;

    // Determine if all edges should have the same sign (clockwise or counterclockwise)
    // Use a small epsilon for floating point comparisons, or check sign of area directly
    bool clockwise = area > 0;


    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            // Test point is center of pixel for rasterization
            float px = x + 0.5f;
            float py = y + 0.5f;

            // Test if point is inside triangle using precomputed edge equations
            bool inside = true;
            for (int i = 0; i < 3; i++) {
                long long val = edge(pts[i], pts[(i + 1) % 3], px, py); // Calculate edge value using lambda

                // Check if point is on the 'correct' side or exactly on the edge (val == 0)
                if ((clockwise && val < 0) || (!clockwise && val > 0)) {
                    inside = false;
                    break;
                }
            }

            if (inside) {
                blendPixel(x, y, col);
            }
        }
    }
}

// Undefine the macros after use
#undef min_x_bound
#undef max_x_bound
#undef min_y_bound
#undef max_y_bound
