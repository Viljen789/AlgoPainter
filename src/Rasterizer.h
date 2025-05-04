#pragma once
#include <vector>
#include <cstdint>
#include "Gene.h"


#pragma pack(push,1)
struct Pixel {
    uint8_t r, g, b, a;
};
#pragma pack(pop)

class Rasterizer {
public:
    Rasterizer(unsigned width, unsigned height);

    void clear(const Pixel &clearColor);

    void draw(const Gene &gene);

    const std::vector<Pixel> &data() const;

private:
    unsigned width_, height_;
    std::vector<Pixel> buffer_;

    void drawCircle(int cx, int cy, int radius, const Pixel &col);

    void drawRectangle(int x, int y, int w, int h, const Pixel &col);

    void drawTriangle(const sf::Vector2f pts[3], const Pixel &col);
};
