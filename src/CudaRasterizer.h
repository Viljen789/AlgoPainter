//
// Created by vilje on 05/05/2025.
//

#ifndef CUDARASTERIZER_H
#define CUDARASTERIZER_H

#include <vector>
#include "Gene.h"
#include "Individual.h"
#include "Pixel.h"
#include "Rasterizer.h" // Still needed for CPU fallback

// Include CUDA headers here *before* any usage of cudaError_t or other CUDA types
#include <cuda_runtime.h>
// #include <cuda_profiler_api.h> // Included for profiling if needed

// Ensure the checkCudaError function prototype is visible to the CHECK macro
// Placed after cuda_runtime.h include
void checkCudaError(cudaError_t err, const char *file, int line);

// Define the CHECK macro using the non-member function
#define CHECK(err) checkCudaError(err, __FILE__, __LINE__)


// Struct for transferring Gene data to CUDA
struct CudaGene {
    int type; // Corresponds to Gene::Shape enum
    float posX, posY;
    float size;
    unsigned char r, g, b, a;
};


class CudaRasterizer {
public:
    CudaRasterizer(unsigned width, unsigned height);

    ~CudaRasterizer();

    // Initialize CUDA and allocate device memory
    bool initialize();

    // Check if CUDA is actually available and initialized
    bool isInitialized() const { return initialized_ && cudaAvailable_; } // Check both flags

    // Upload individuals to GPU (and potentially target image if resolution changes)
    void uploadPopulation(const std::vector<Individual> &population, const std::vector<Pixel> &targetImage);

    // Render entire population and calculate fitness in one GPU operation
    void renderAndEvaluate(const std::vector<Pixel> &targetImage, std::vector<float> &fitnessResults);

    // Get the rendered image for a specific individual (from GPU if possible)
    std::vector<Pixel> getRenderedImage(int individualIndex);

    // Add resize capability, reallocating GPU memory
    void resize(unsigned width, unsigned height, const std::vector<Pixel> &targetImage);

    // Add getters for current dimensions
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }

private:
    unsigned int width_, height_; // Use unsigned int consistently
    size_t pixelBufferSize_; // Size of one image buffer (width * height * sizeof(Pixel))
    size_t populationSize_ = 0;
    unsigned int genesPerIndividual_ = 0; // Needed for GPU kernel indexing

    // Host memory for results
    std::vector<float> h_fitnessResults_;
    // std::vector<Pixel> h_renderedBuffer_; // Not strictly needed if we copy one by one

    // Device memory pointers
    CudaGene *d_population_ = nullptr;
    Pixel *d_targetImage_ = nullptr;
    Pixel *d_renderedBuffers_ = nullptr; // Array of buffers, one per individual
    float *d_fitnessResults_ = nullptr;

    // Store host copies for fallback or getRenderedImage
    std::vector<Individual> population_;
    // We don't need a host copies of all rendered buffers

    // CUDA status flags
    bool cudaAvailable_ = false; // True if CUDA runtime is found and device exists
    bool initialized_ = false; // True if GPU memory is allocated and ready

    // CPU fallback rasterizer (still useful if CUDA is not available or for getRenderedImage)
    Rasterizer cpuRasterizer_;

    // Helper function to copy target image to GPU
    void uploadTargetImage(const std::vector<Pixel> &targetImage);

    // The CHECK_CUDA_ERROR definition is now a free function below
    // Add these declarations to CudaRasterizer.h
    // In CudaRasterizer.h, replace the three extern "C" declarations with:

    // Function declarations for CUDA kernel wrappers
    void launchRenderKernel(CudaGene *d_population,
                            Pixel *d_renderedBuffers,
                            unsigned int populationSize,
                            unsigned int genesPerIndividual,
                            unsigned int width,
                            unsigned int height);

    void launchFitnessKernel(Pixel *d_renderedBuffers,
                             Pixel *d_targetImage,
                             float *d_fitnessResults,
                             unsigned int populationSize,
                             unsigned int width,
                             unsigned int height);

    void launchClearBuffersKernel(Pixel *d_renderedBuffers,
                                  unsigned int populationSize,
                                  unsigned int width,
                                  unsigned int height,
                                  Pixel clearColor);
};

#endif //CUDARASTERIZER_H
