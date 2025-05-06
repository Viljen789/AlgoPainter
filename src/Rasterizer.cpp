#include "Rasterizer.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>

#define min_x_bound min_x_bound_rasterizer
#define max_x_bound max_x_bound_rasterizer
#define min_y_bound min_y_bound_rasterizer
#define max_y_bound max_y_bound_rasterizer

Rasterizer::Rasterizer(unsigned w, unsigned h)
    : width_(w), height_(h), buffer_(static_cast<size_t>(w) * h) {
}

void Rasterizer::resize(unsigned w, unsigned h) {
    width_ = w;
    height_ = h;
    buffer_.resize(static_cast<size_t>(w) * h);
}

void Rasterizer::clear(const Pixel &c) {
    std::fill(buffer_.begin(), buffer_.end(), c);
}

const std::vector<Pixel> &Rasterizer::data() const {
    return buffer_;
}

inline void Rasterizer::blendPixel(int x, int y, const Pixel &col) {
    if (x >= 0 && static_cast<unsigned>(x) < width_ && y >= 0 && static_cast<unsigned>(y) < height_) {
        Pixel &p = buffer_[static_cast<size_t>(y) * width_ + x];
        uint16_t alpha = col.a;
        if (alpha == 0) return;
        if (alpha == 255) {
            p = col;
            return;
        }
        uint16_t invAlpha = 255 - alpha;

        unsigned int r = (unsigned int) p.r * invAlpha + (unsigned int) col.r * alpha;
        unsigned int g = (unsigned int) p.g * invAlpha + (unsigned int) col.g * alpha;
        unsigned int b = (unsigned int) p.b * invAlpha + (unsigned int) col.b * alpha;

        p.r = static_cast<uint8_t>(r >> 8);
        p.g = static_cast<uint8_t>(g >> 8);
        p.b = static_cast<uint8_t>(b >> 8);
    }
}

void Rasterizer::draw(const Gene &g) {
    Pixel col{g.getColor().r, g.getColor().g, g.getColor().b, g.getColor().a};
    switch (g.getType()) {
        case Gene::Shape::Circle:
            drawCircle(static_cast<int>(g.getPos().x), static_cast<int>(g.getPos().y), g.getSize(), col);
            break;
        case Gene::Shape::Square: {
            drawRectangle(g.getPos().x - g.getSize() / 2.0f, g.getPos().y - g.getSize() / 2.0f, g.getSize(),
                          g.getSize(), col);
            break;
        }
        case Gene::Shape::Triangle: {
            sf::Vector2f p0 = g.getPos();
            float s = g.getSize();

            sf::Vector2f points_relative[3] = {
                {0.0f, 0.0f},
                {s, 0.0f},
                {s / 2.0f, s * 0.866025f}
            };

            sf::Vector2f points_absolute[3];
            points_absolute[0] = p0 + points_relative[0];
            points_absolute[1] = p0 + points_relative[1];
            points_absolute[2] = p0 + points_relative[2];

            drawTriangle(points_absolute, col);
            break;
        }
    }
}

void Rasterizer::drawCircle(int cx, int cy, float radius, const Pixel &col) {
    int r = static_cast<int>(radius);
    int r2 = r * r;
    int y_start = std::max(0, cy - r);
    int y_end = std::min(static_cast<int>(height_), cy + r + 1);

    for (int y = y_start; y < y_end; ++y) {
        int dy = y - cy;
        int dx_squared = r2 - dy * dy;
        if (dx_squared >= 0) {
            int dx = static_cast<int>(std::sqrt(static_cast<float>(dx_squared)));
            int x_start = std::max(0, cx - dx);
            int x_end = std::min(static_cast<int>(width_), cx + dx + 1);

            for (int x = x_start; x < x_end; ++x) {
                blendPixel(x, y, col);
            }
        }
    }
}

void Rasterizer::drawRectangle(float x_f, float y_f, float w_f, float h_f, const Pixel &col) {
    int x0 = std::max(0, static_cast<int>(std::floor(x_f)));
    int y0 = std::max(0, static_cast<int>(std::floor(y_f)));
    int x1 = std::min(static_cast<int>(width_), static_cast<int>(std::ceil(x_f + w_f)));
    int y1 = std::min(static_cast<int>(height_), static_cast<int>(std::ceil(y_f + h_f)));

    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            blendPixel(xx, yy, col);
        }
    }
}

void Rasterizer::drawTriangle(const sf::Vector2f pts[3], const Pixel &col) {
    float minXf = std::min({pts[0].x, pts[1].x, pts[2].x}, [](float a, float b) { return a < b; });
    float maxXf = std::max({pts[0].x, pts[1].x, pts[2].x}, [](float a, float b) { return a < b; });
    float minYf = std::min({pts[0].y, pts[1].y, pts[2].y}, [](float a, float b) { return a < b; });
    float maxYf = std::max({pts[0].y, pts[1].y, pts[2].y}, [](float a, float b) { return a < b; });

    int x_start = std::max(0, static_cast<int>(std::floor(minXf)));
    int x_end = std::min(static_cast<int>(width_), static_cast<int>(std::ceil(maxXf)));
    int y_start = std::max(0, static_cast<int>(std::floor(minYf)));
    int y_end = std::min(static_cast<int>(height_), static_cast<int>(std::ceil(maxYf)));

    auto edge = [&](const sf::Vector2f &a, const sf::Vector2f &b, float x, float y) {
        return (long long) ((double) b.x - a.x) * ((double) y - a.y) - (long long) ((double) b.y - a.y) * (
                   (double) x - a.x);
    };

    long long edge_const[3][3];
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        edge_const[i][0] = static_cast<long long>(pts[j].y - pts[i].y);
        edge_const[i][1] = static_cast<long long>(pts[i].x - pts[j].x);
        edge_const[i][2] = static_cast<long long>((double) pts[j].x * pts[i].y - (double) pts[i].x * pts[j].y);
    }

    long long area = edge(pts[0], pts[1], pts[2].x, pts[2].y);
    if (area == 0) return;

    bool clockwise = area > 0;

    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            float px = x + 0.5f;
            float py = y + 0.5f;

            bool inside = true;
            for (int i = 0; i < 3; i++) {
                long long val = edge(pts[i], pts[(i + 1) % 3], px, py);

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

#undef min_x_bound
#undef max_x_bound
#undef min_y_bound
#undef max_y_bound
