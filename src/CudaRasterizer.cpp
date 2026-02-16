#include "CudaRasterizer.h"
#include "Fitness.h"
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
    }
}

CudaRasterizer::CudaRasterizer(unsigned width, unsigned height)
    : width_(width), height_(height), cpuRasterizer_(width, height) {
    int deviceCount;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        cudaAvailable_ = true;
        cudaSetDevice(0);
    }
}

CudaRasterizer::~CudaRasterizer() {
    if (initialized_) {
        cudaFree(d_population_);
        cudaFree(d_targetImage_);
        cudaFree(d_renderedBuffers_);
        cudaFree(d_fitnessResults_);
    }
}

void CudaRasterizer::resize(unsigned width, unsigned height, const std::vector<Pixel> &targetImage) {
    width_ = width; height_ = height;
    if (initialized_) {
        cudaFree(d_targetImage_);
        cudaFree(d_renderedBuffers_);
        d_targetImage_ = nullptr; d_renderedBuffers_ = nullptr;
        initialized_ = false;
    }
    uploadTargetImage(targetImage);
}

void CudaRasterizer::uploadPopulation(const std::vector<Individual> &population, const std::vector<Pixel> &targetImage) {
    populationSize_ = population.size();
    genesPerIndividual_ = population[0].size();
    
    if (cudaAvailable_ && !initialized_) {
        cudaMalloc(&d_population_, populationSize_ * genesPerIndividual_ * sizeof(CudaGene));
        cudaMalloc(&d_renderedBuffers_, populationSize_ * width_ * height_ * sizeof(Pixel));
        cudaMalloc(&d_fitnessResults_, populationSize_ * sizeof(float));
        if (!d_targetImage_) uploadTargetImage(targetImage);
        initialized_ = true;
    }

    if (initialized_) {
        std::vector<CudaGene> h_cudaPop(populationSize_ * genesPerIndividual_);
        for (size_t i = 0; i < populationSize_; ++i) {
            for (size_t j = 0; j < genesPerIndividual_; ++j) {
                const Gene &g = population[i][j];
                CudaGene &cg = h_cudaPop[i * genesPerIndividual_ + j];
                cg.type = (int)g.getType();
                cg.posX = g.getPos().x;
                cg.posY = g.getPos().y;
                cg.sizeX = g.getSize().x;
                cg.sizeY = g.getSize().y;
                cg.rotation = g.getRotation();
                cg.r = g.getColor().r;
                cg.g = g.getColor().g;
                cg.b = g.getColor().b;
                cg.a = g.getColor().a;
            }
        }
        cudaMemcpy(d_population_, h_cudaPop.data(), h_cudaPop.size() * sizeof(CudaGene), cudaMemcpyHostToDevice);
    }
}

void CudaRasterizer::uploadTargetImage(const std::vector<Pixel> &targetImage) {
    if (!cudaAvailable_) return;
    if (d_targetImage_) cudaFree(d_targetImage_);
    cudaMalloc(&d_targetImage_, width_ * height_ * sizeof(Pixel));
    cudaMemcpy(d_targetImage_, targetImage.data(), width_ * height_ * sizeof(Pixel), cudaMemcpyHostToDevice);
}

void CudaRasterizer::renderAndEvaluate(const std::vector<Pixel> &targetImage, std::vector<float> &fitnessResults) {
    if (!initialized_) return;
    
    launchClearBuffersKernel(d_renderedBuffers_, (unsigned)populationSize_, width_, height_, {0, 0, 0, 255});
    launchRenderKernel(d_population_, d_renderedBuffers_, (unsigned)populationSize_, (unsigned)genesPerIndividual_, width_, height_);
    launchFitnessKernel(d_renderedBuffers_, d_targetImage_, d_fitnessResults_, (unsigned)populationSize_, width_, height_);
    
    fitnessResults.resize(populationSize_);
    cudaMemcpy(fitnessResults.data(), d_fitnessResults_, populationSize_ * sizeof(float), cudaMemcpyDeviceToHost);
}

std::vector<Pixel> CudaRasterizer::getRenderedImage(int index) {
    std::vector<Pixel> res(width_ * height_);
    if (initialized_) {
        cudaMemcpy(res.data(), d_renderedBuffers_ + (size_t)index * width_ * height_, width_ * height_ * sizeof(Pixel), cudaMemcpyDeviceToHost);
    }
    return res;
}
