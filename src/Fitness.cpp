//
// Created by vilje on 03/05/2025.
//

#include "Fitness.h"
float computeFitness(const sf::Image& rendered, const sf::Image& target) {
  auto size = target.getSize();
  float sum = 0;
  for (unsigned int y = 0; y < size.y; y++) {
    for (unsigned int x = 0; x < size.x; x++) {
      auto c1 = rendered.getPixel({x, y});
      auto c2 = target.getPixel({x, y});
      float dr = float(c1.r) - c2.r;
      float dg = float(c1.g) - c2.g;
      float db = float(c1.b) - c2.b;
      sum -= dr * dr + dg * dg + db * db;
    }
  }
  return sum;
}