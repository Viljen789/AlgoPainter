#ifndef CUDARASTERIZER_H
#define CUDARASTERIZER_H

#include "Gene.h"
#include "IRasterizer.h"
#include "Individual.h"
#include "Pixel.h"
#include "Rasterizer.h"

#include <cuda_runtime.h>

#include <vector>

void checkCudaError(cudaError_t err, const char* file, int line);

#define CHECK(err) checkCudaError(err, __FILE__, __LINE__)

struct CudaGene {
    int type;
    float posX, posY;
    float sizeX, sizeY;
    float rotation;
    unsigned char r, g, b, a;
};

class CudaRasterizer : public IRasterizer {
  public:
    CudaRasterizer(unsigned width, unsigned height);

    ~CudaRasterizer() override;

    bool initialize();

    bool isInitialized() const { return initialized_ && cudaAvailable_; }

    // IRasterizer interface implementation
    void uploadPopulation(const std::vector<Individual>& population, const std::vector<Pixel>& targetImage) override;
    void renderAndEvaluate(const std::vector<Pixel>& targetImage, std::vector<float>& fitnessResults) override;
    std::vector<Pixel> getRenderedImage(int individualIndex) override;
    void resize(unsigned width, unsigned height, const std::vector<Pixel>& targetImage) override;

    unsigned int getWidth() const override { return width_; }
    unsigned int getHeight() const override { return height_; }

  private:
    unsigned int width_, height_;
    size_t pixelBufferSize_;
    size_t populationSize_ = 0;
    unsigned int genesPerIndividual_ = 0;

    std::vector<float> h_fitnessResults_;

    CudaGene* d_population_ = nullptr;
    Pixel* d_targetImage_ = nullptr;
    Pixel* d_renderedBuffers_ = nullptr;
    float* d_fitnessResults_ = nullptr;

    std::vector<Individual> population_;

    bool cudaAvailable_ = false;
    bool initialized_ = false;

    Rasterizer cpuRasterizer_;

    void uploadTargetImage(const std::vector<Pixel>& targetImage);

    void launchRenderKernel(CudaGene* d_population, Pixel* d_renderedBuffers, unsigned int populationSize,
                            unsigned int genesPerIndividual, unsigned int width, unsigned int height);

    void launchFitnessKernel(Pixel* d_renderedBuffers, Pixel* d_targetImage, float* d_fitnessResults,
                             unsigned int populationSize, unsigned int width, unsigned int height);

    void launchClearBuffersKernel(Pixel* d_renderedBuffers, unsigned int populationSize, unsigned int width,
                                  unsigned int height, Pixel clearColor);

    void fitnessKernel_optimized(const Pixel* renderedBuffers, const Pixel* targetImage, float* fitnessResults,
                                 unsigned int numIndividuals, unsigned int imgWidth, unsigned int imgHeight);
};

#endif // CUDARASTERIZER_H
