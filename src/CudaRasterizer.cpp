#include "CudaRasterizer.h"
#include "Fitness.h"
#include <iostream>
#include <cuda_runtime.h>

#ifndef NO_OPENMP
#include <omp.h>
#endif

extern "C" __global__ void renderKernel_simple(const CudaGene *population, Pixel *renderedBuffers,
                                               unsigned int numIndividuals, unsigned int genesPerIndividual,
                                               unsigned int imgWidth, unsigned int imgHeight);

extern "C" __global__ void fitnessKernel(const Pixel *renderedBuffers, const Pixel *targetImage,
                                         float *fitnessResults, unsigned int numIndividuals,
                                         unsigned int imgWidth, unsigned int imgHeight);

extern "C" __global__ void clearBuffersKernel(Pixel *renderedBuffers, unsigned int numIndividuals,
                                              unsigned int imgWidth, unsigned int imgHeight, Pixel clearColor);

void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
    }
}

CudaRasterizer::CudaRasterizer(unsigned width, unsigned height)
    : width_(width), height_(height),
      pixelBufferSize_(static_cast<size_t>(width) * height * sizeof(Pixel)),
      cpuRasterizer_(width, height) {
    initialize();
}

CudaRasterizer::~CudaRasterizer() {
    if (initialized_ && cudaAvailable_) {
        if (d_population_)
            CHECK(cudaFree(d_population_));
        if (d_targetImage_)
            CHECK(cudaFree(d_targetImage_));
        if (d_renderedBuffers_)
            CHECK(cudaFree(d_renderedBuffers_));
        if (d_fitnessResults_)
            CHECK(cudaFree(d_fitnessResults_));
    }
}

bool CudaRasterizer::initialize() {
    if (cudaAvailable_) return true;

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        cudaAvailable_ = false;
        initialized_ = false;
        std::cerr << "CUDA not available or no devices found. Using CPU fallback implementation." << std::endl;
        cudaGetLastError();
        return false;
    }

    CHECK(cudaSetDevice(0));
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, 0));
    std::cout << "CUDA device found: " << devProp.name << std::endl;

    cudaAvailable_ = true;
    initialized_ = false;

    return true;
}

void CudaRasterizer::resize(unsigned width, unsigned height, const std::vector<Pixel> &targetImage) {
    width_ = width;
    height_ = height;
    pixelBufferSize_ = static_cast<size_t>(width) * height * sizeof(Pixel);

    cpuRasterizer_.resize(width_, height_);

    if (cudaAvailable_) {
        if (d_population_)
            CHECK(cudaFree(d_population_));
        if (d_targetImage_)
            CHECK(cudaFree(d_targetImage_));
        if (d_renderedBuffers_)
            CHECK(cudaFree(d_renderedBuffers_));
        if (d_fitnessResults_)
            CHECK(cudaFree(d_fitnessResults_));

        d_population_ = nullptr;
        d_targetImage_ = nullptr;
        d_renderedBuffers_ = nullptr;
        d_fitnessResults_ = nullptr;

        initialized_ = false;

        uploadTargetImage(targetImage);
    }
}


void CudaRasterizer::uploadPopulation(const std::vector<Individual> &population,
                                      const std::vector<Pixel> &targetImage) {
    populationSize_ = population.size();
    if (populationSize_ == 0) {
        std::cerr << "Warning: Attempted to upload empty population" << std::endl;
        genesPerIndividual_ = 0;
        return;
    }

    genesPerIndividual_ = population[0].size();
    if (genesPerIndividual_ == 0) {
        std::cerr << "Warning: First individual has 0 genes" << std::endl;
        return;
    }

    bool populationValid = true;
    for (const auto &individual: population) {
        if (individual.size() != genesPerIndividual_) {
            std::cerr << "Warning: Individual has " << individual.size()
                    << " genes, expected " << genesPerIndividual_ << std::endl;
            populationValid = false;
            break;
        }
    }

    if (!populationValid) {
        std::cerr << "Warning: Population has inconsistent gene counts. Using only valid individuals." << std::endl;
    }

    population_ = population;
    h_fitnessResults_.resize(populationSize_);

    if (cudaAvailable_) {
        if (!initialized_) {
            size_t geneDataSize = static_cast<size_t>(populationSize_) * genesPerIndividual_ * sizeof(CudaGene);
            CHECK(cudaMalloc(&d_population_, geneDataSize));
            CHECK(cudaMalloc(&d_renderedBuffers_, static_cast<size_t>(populationSize_) * pixelBufferSize_));
            CHECK(cudaMalloc(&d_fitnessResults_, static_cast<size_t>(populationSize_) * sizeof(float)));

            if (!d_targetImage_) {
                uploadTargetImage(targetImage);
            }

            initialized_ = true;
            std::cout << "CUDA device memory allocated." << std::endl;
        }

        std::vector<CudaGene> h_cudaPopulation(static_cast<size_t>(populationSize_) * genesPerIndividual_);
        for (size_t i = 0; i < populationSize_; ++i) {
            for (size_t j = 0; j < genesPerIndividual_; ++j) {
                const Gene &gene = population[i][j];
                CudaGene &cudaGene = h_cudaPopulation[static_cast<size_t>(i) * genesPerIndividual_ + j];
                cudaGene.type = static_cast<int>(gene.getType());
                cudaGene.posX = gene.getPos().x;
                cudaGene.posY = gene.getPos().y;
                cudaGene.size = gene.getSize();
                cudaGene.r = gene.getColor().r;
                cudaGene.g = gene.getColor().g;
                cudaGene.b = gene.getColor().b;
                cudaGene.a = gene.getColor().a;
            }
        }

        if (d_population_ != nullptr && !h_cudaPopulation.empty()) {
            size_t geneDataSize = static_cast<size_t>(populationSize_) * genesPerIndividual_ * sizeof(CudaGene);
            CHECK(cudaMemcpy(d_population_, h_cudaPopulation.data(), geneDataSize, cudaMemcpyHostToDevice));
        } else {
            std::cerr << "Error: Cannot copy population data - device buffer or host data is NULL" << std::endl;
            initialized_ = false;
        }
    }
}

void CudaRasterizer::uploadTargetImage(const std::vector<Pixel> &targetImage) {
    if (!cudaAvailable_) return;

    if (d_targetImage_) {
        CHECK(cudaFree(d_targetImage_));
        d_targetImage_ = nullptr;
    }

    size_t targetBufferSize = static_cast<size_t>(width_) * height_ * sizeof(Pixel);
    CHECK(cudaMalloc(&d_targetImage_, targetBufferSize));
    CHECK(cudaMemcpy(d_targetImage_, targetImage.data(), targetBufferSize, cudaMemcpyHostToDevice));
    std::cout << "Target image uploaded to GPU (" << width_ << "x" << height_ << ")" << std::endl;
}


void CudaRasterizer::renderAndEvaluate(const std::vector<Pixel> &targetImage, std::vector<float> &fitnessResults) {
    if (populationSize_ == 0 || genesPerIndividual_ == 0) {
        std::cerr << "Population not uploaded or empty before evaluation" << std::endl;
        fitnessResults.assign(populationSize_, -1e12f);
        return;
    }

    if (initialized_ && cudaAvailable_) {
        Pixel clearColor = {0, 0, 0, 255};
        int numThreadsPerClearBlock = 256;
        size_t totalPixelsAcrossBuffers = static_cast<size_t>(populationSize_) * width_ * height_;
        int numClearBlocks = static_cast<int>((totalPixelsAcrossBuffers + numThreadsPerClearBlock - 1) /
                                              numThreadsPerClearBlock);

        launchClearBuffersKernel(d_renderedBuffers_, static_cast<unsigned int>(populationSize_),
                                 static_cast<unsigned int>(width_), static_cast<unsigned int>(height_),
                                 clearColor);
        CHECK(cudaGetLastError());

        int threadsPerRenderBlock = 256;
        size_t totalRenderThreads = static_cast<size_t>(populationSize_) * genesPerIndividual_;
        int numRenderBlocks = static_cast<int>((totalRenderThreads + threadsPerRenderBlock - 1) /
                                               threadsPerRenderBlock);

        if (genesPerIndividual_ > 1024) {
            std::cerr << "Warning: genesPerIndividual (" << genesPerIndividual_ <<
                    ") exceeds typical CUDA block size limit (1024) for the simple render kernel launch config." <<
                    std::endl;
            std::cerr << "Using flattened thread distribution instead." << std::endl;

            launchRenderKernel(d_population_, d_renderedBuffers_,
                               static_cast<unsigned int>(populationSize_),
                               genesPerIndividual_, static_cast<unsigned int>(width_),
                               static_cast<unsigned int>(height_));
        } else {
            launchRenderKernel(d_population_, d_renderedBuffers_,
                               static_cast<unsigned int>(populationSize_),
                               genesPerIndividual_, static_cast<unsigned int>(width_),
                               static_cast<unsigned int>(height_));
        }

        CHECK(cudaGetLastError());

        int threadsPerFitnessBlock = 256;

        launchFitnessKernel(d_renderedBuffers_, d_targetImage_, d_fitnessResults_,
                            static_cast<unsigned int>(populationSize_),
                            static_cast<unsigned int>(width_), static_cast<unsigned int>(height_));

        CHECK(cudaGetLastError());

        if (d_fitnessResults_ != nullptr) {
            CHECK(cudaMemcpy(h_fitnessResults_.data(), d_fitnessResults_,
                static_cast<size_t>(populationSize_) * sizeof(float),
                cudaMemcpyDeviceToHost));
        } else {
            std::cerr << "Error: Cannot copy fitness results - device buffer is NULL" << std::endl;
            goto cpu_fallback;
        }

        CHECK(cudaDeviceSynchronize());

        fitnessResults = h_fitnessResults_;
    } else {
    cpu_fallback:
        if (!initialized_ && cudaAvailable_) {
            std::cerr << "CUDA initialization failed, using CPU fallback." << std::endl;
        }

        unsigned int currentW = width_;
        unsigned int currentH = height_;

#ifndef NO_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int i = 0; i < static_cast<int>(populationSize_); i++) {
            cpuRasterizer_.clear({0, 0, 0, 255});

            if (i < population_.size()) {
                for (const auto &gene: population_[i]) {
                    cpuRasterizer_.draw(gene);
                }
                h_fitnessResults_[i] = computeFitness(cpuRasterizer_.data(), targetImage, currentW, currentH);
            } else {
                h_fitnessResults_[i] = -1e9;
            }
        }

        fitnessResults = h_fitnessResults_;
    }
}

std::vector<Pixel> CudaRasterizer::getRenderedImage(int individualIndex) {
    if (individualIndex < 0 || individualIndex >= static_cast<int>(populationSize_)) {
        throw std::runtime_error("Individual index out of bounds");
    }

    std::vector<Pixel> imageBuffer(static_cast<size_t>(width_) * height_);

    if (initialized_ && cudaAvailable_) {
        if (!d_renderedBuffers_) {
            std::cerr << "GPU buffers not allocated, cannot get rendered image from GPU." << std::endl;
            goto cpu_render_for_display;
        }

        size_t offset = static_cast<size_t>(individualIndex) * width_ * height_ * sizeof(Pixel);

        CHECK(cudaMemcpy(imageBuffer.data(), d_renderedBuffers_ + offset / sizeof(Pixel), static_cast<size_t>(width_) *
            height_ * sizeof(Pixel), cudaMemcpyDeviceToHost));

        CHECK(cudaDeviceSynchronize());

        return imageBuffer;
    } else {
    cpu_render_for_display:
        cpuRasterizer_.clear({0, 0, 0, 255});
        if (individualIndex < population_.size()) {
            for (const auto &gene: population_[individualIndex]) {
                cpuRasterizer_.draw(gene);
            }
            imageBuffer = cpuRasterizer_.data();
        } else {
            std::cerr << "CPU population data mismatch for index " << individualIndex << std::endl;
            std::fill(imageBuffer.begin(), imageBuffer.end(), Pixel{0, 0, 0, 255});
        }
        return imageBuffer;
    }
}
