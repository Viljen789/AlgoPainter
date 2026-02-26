//
// Created by Viljen Apalset Vassbø on 17/02/2026.
//

#ifndef ALGOPAINTER_FONTANALYZER_H
#define ALGOPAINTER_FONTANALYZER_H

#pragma once
#include "FontAnalyzer.h"

#include <string>
#include <unordered_map>

struct GlyphData {
    char symbol;
    float weightTopLeft;
    float weightTopRight;
    float weightBottomLeft;
    float weightBottomRight;
};

class FontAnalyzer {
  public:
    std::unordered_map<char, GlyphData> analyzeFont(const std::string& fontPath, const std::string& charset);
};

#endif // ALGOPAINTER_FONTANALYZER_H