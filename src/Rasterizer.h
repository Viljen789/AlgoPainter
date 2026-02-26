#pragma once
#include "Gene.h"
#include "IRasterizer.h"
#include "Individual.h"
#include "Pixel.h"

#include <cstdint>
#include <vector>

class Rasterizer : public IRasterizer {
  public:
    Rasterizer(unsigned width, unsigned height);

    Rasterizer() : Rasterizer(1, 1) {}

    ~Rasterizer() override = default;

    // IRasterizer interface implementation
    void uploadPopulation(const std::vector<Individual>& population, const std::vector<Pixel>& targetImage) override;
    void renderAndEvaluate(const std::vector<Pixel>& targetImage, std::vector<float>& fitnessResults) override;
    std::vector<Pixel> getRenderedImage(int index) override;
    void resize(unsigned width, unsigned height, const std::vector<Pixel>& targetImage) override;

    unsigned int getWidth() const override { return width_; }
    unsigned int getHeight() const override { return height_; }

    // Additional CPU rasterizer methods
    void clear(const Pixel& clearColor);
    void draw(const Gene& gene);
    const std::vector<Pixel>& data() const;

  private:
    unsigned width_, height_;
    std::vector<Pixel> buffer_;
    std::vector<Individual> population_;
    std::vector<Pixel> targetImage_;

    inline void blendPixel(int x, int y, const Pixel& col);

    void drawCircle(int cx, int cy, float radius, const Pixel& col);
    void drawRectangle(float x, float y, float w, float h, const Pixel& col);
    void drawTriangle(const sf::Vector2f pts[3], const Pixel& col);

    // Helper to compute fitness for a single rendered image vs target
    float computeFitness(const std::vector<Pixel>& rendered, const std::vector<Pixel>& target) const;
};
