//
// Created by Viljen Apalset Vassbø on 17/02/2026.
//
#include "FontAnalyzer.h"
// #include <stb_truetype.h>  // TODO: Add stb_truetype when implementing actual font analysis
#include <iostream>

std::unordered_map<char, GlyphData> FontAnalyzer::analyzeFont(const std::string& fontPath, const std::string& charset) {
    std::unordered_map<char, GlyphData> glyphMap;
    for (char c : charset) {
        GlyphData data;
        data.symbol = c;
        data.weightTopLeft = 0.25f;
        data.weightTopRight = 0.25f;
        data.weightBottomLeft = 0.25f;
        data.weightBottomRight = 0.25f;
        glyphMap[c] = data;
    }
    return glyphMap;
}
