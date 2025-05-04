//
// Created by vilje on 04/05/2025.
//
#include "Rasterizer.h"
#include <algorithm>
#include <cmath>

Rasterizer::Rasterizer(unsigned w, unsigned h)
    : width_(w), height_(h), buffer_(w * h) {
}

void Rasterizer::clear(const Pixel &c) {
    std::fill(buffer_.begin(), buffer_.end(), c);
}

const std::vector<Pixel> &Rasterizer::data() const {
    return buffer_;
}

void Rasterizer::draw(const Gene &g) {
    Pixel col{g.getColor().r, g.getColor().g, g.getColor().b, g.getColor().a};
    switch (g.getType()) {
        case Gene::Shape::Circle:
            drawCircle(int(g.getPos().x), int(g.getPos().y), int(g.getSize()), col);
            break;
        case Gene::Shape::Square: {
            int x = int(g.getPos().x - g.getSize() / 2);
            int y = int(g.getPos().y - g.getSize() / 2);
            drawRectangle(x, y, int(g.getSize()), int(g.getSize()), col);
            break;
        }
        case Gene::Shape::Triangle: {
            sf::Vector2f p0 = g.getPos();
            float s = g.getSize();
            sf::Vector2f pts[3] = {
                p0,
                {p0.x + s, p0.y},
                {p0.x + s / 2, p0.y + s * 0.866f}
            };
            drawTriangle(pts, col);
            break;
        }
    }
}

void Rasterizer::drawCircle(int cx, int cy, int r, const Pixel &col) {
    int r2 = r * r;
    int y0 = std::max(0, cy - r);
    int y1 = std::min(int(height_), cy + r);


    if (col.a >= 250) {
        for (int y = y0; y < y1; ++y) {
            int dy = y - cy;
            int dx = int(std::sqrt(r2 - dy * dy));
            int x0 = std::max(0, cx - dx);
            int x1 = std::min(int(width_), cx + dx);
            for (int x = x0; x < x1; ++x) {
                // Direct assignment for opaque colors
                Pixel &p = buffer_[y * width_ + x];
                p.r = col.r;
                p.g = col.g;
                p.b = col.b;
            }
        }
    } else {
        float a = col.a / 255.f;
        float invA = 1.0f - a;
        for (int y = y0; y < y1; ++y) {
            int dy = y - cy;
            int dx = int(std::sqrt(r2 - dy * dy));
            int x0 = std::max(0, cx - dx);
            int x1 = std::min(int(width_), cx + dx);
            for (int x = x0; x < x1; ++x) {
                Pixel &p = buffer_[y * width_ + x];
                p.r = uint8_t(p.r * invA + col.r * a);
                p.g = uint8_t(p.g * invA + col.g * a);
                p.b = uint8_t(p.b * invA + col.b * a);
            }
        }
    }
}

void Rasterizer::drawRectangle(int x, int y, int w, int h, const Pixel &col) {
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(int(width_), x + w);
    int y1 = std::min(int(height_), y + h);
    for (int yy = y0; yy < y1; ++yy) {
        for (int xx = x0; xx < x1; ++xx) {
            Pixel &p = buffer_[yy * width_ + xx];
            float a = col.a / 255.f;
            p.r = uint8_t(p.r * (1 - a) + col.r * a);
            p.g = uint8_t(p.g * (1 - a) + col.g * a);
            p.b = uint8_t(p.b * (1 - a) + col.b * a);
        }
    }
}

void Rasterizer::drawTriangle(const sf::Vector2f pts[3], const Pixel &col) {
    float minXf = std::min({pts[0].x, pts[1].x, pts[2].x});
    float maxXf = std::max({pts[0].x, pts[1].x, pts[2].x});
    float minYf = std::min({pts[0].y, pts[1].y, pts[2].y});
    float maxYf = std::max({pts[0].y, pts[1].y, pts[2].y});
    int x0 = std::max(0, int(std::floor(minXf)));
    int x1 = std::min(int(width_), int(std::ceil(maxXf)));
    int y0 = std::max(0, int(std::floor(minYf)));
    int y1 = std::min(int(height_), int(std::ceil(maxYf)));


    auto edge = [&](const sf::Vector2f &a, const sf::Vector2f &b, float x, float y) {
        return (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    };
    float area = edge(pts[0], pts[1], pts[2].x, pts[2].y);
    if (area == 0) return;

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float w0 = edge(pts[1], pts[2], x + 0.5f, y + 0.5f);
            float w1 = edge(pts[2], pts[0], x + 0.5f, y + 0.5f);
            float w2 = edge(pts[0], pts[1], x + 0.5f, y + 0.5f);
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                Pixel &p = buffer_[y * width_ + x];
                float a = col.a / 255.f;
                p.r = uint8_t(p.r * (1 - a) + col.r * a);
                p.g = uint8_t(p.g * (1 - a) + col.g * a);
                p.b = uint8_t(p.b * (1 - a) + col.b * a);
            }
        }
    }
}

