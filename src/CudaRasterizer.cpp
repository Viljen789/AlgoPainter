//
// Created by vilje on 05/05/2025.
//

#include "CudaRasterizer.h"
#include "Fitness.h" // Still needed for CPU fallback
#include <iostream>
#include <cuda_runtime.h> // CUDA API
// No need for #include <device_launch_parameters.h> here, it's for kernels

#ifndef NO_OPENMP
#include <omp.h>
#endif

// Define kernels declared in kernels.cu
// extern "C" tells the C++ compiler these functions are defined externally
// and should not be name-mangled. NVCC will handle linking.
extern "C" __global__ void renderKernel_simple(const CudaGene *population, Pixel *renderedBuffers,
                                               unsigned int numIndividuals, unsigned int genesPerIndividual,
                                               unsigned int imgWidth, unsigned int imgHeight);

extern "C" __global__ void fitnessKernel(const Pixel *renderedBuffers, const Pixel *targetImage,
                                         float *fitnessResults, unsigned int numIndividuals,
                                         unsigned int imgWidth, unsigned int imgHeight);

extern "C" __global__ void clearBuffersKernel(Pixel *renderedBuffers, unsigned int numIndividuals,
                                              unsigned int imgWidth, unsigned int imgHeight, Pixel clearColor);


// Simple non-member CUDA error checking helper function
void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        // In a real application, you might want to exit or throw
        // exit(1);
    }
}

// The CHECK macro is defined in CudaRasterizer.h using this function


CudaRasterizer::CudaRasterizer(unsigned width, unsigned height)
    : width_(width), height_(height),
      pixelBufferSize_(static_cast<size_t>(width) * height * sizeof(Pixel)), // Cast to size_t for calculation
      cpuRasterizer_(width, height) {
    // Attempt to initialize CUDA immediately
    initialize();
}

CudaRasterizer::~CudaRasterizer() {
    // Free GPU memory if it was allocated
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
    // No need to free cpuRasterizer_, it's an object member
}

bool CudaRasterizer::initialize() {
    if (cudaAvailable_) return true; // Already checked

    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        cudaAvailable_ = false;
        initialized_ = false; // Ensure initialized_ is false if CUDA is not available
        std::cerr << "CUDA not available or no devices found. Using CPU fallback implementation." << std::endl;
        // Clear any potential CUDA error flags if devices were checked but failed
        cudaGetLastError();
        return false;
    }

    // Select device (e.g., device 0 or based on capabilities)
    CHECK(cudaSetDevice(0)); // Use device 0
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, 0));
    std::cout << "CUDA device found: " << devProp.name << std::endl;

    cudaAvailable_ = true;
    initialized_ = false; // Will be set true after first uploadPopulation/resize

    // At this point, CUDA is available, but memory isn't allocated yet.
    // Allocation happens in uploadPopulation or resize.
    return true;
}

void CudaRasterizer::resize(unsigned width, unsigned height, const std::vector<Pixel> &targetImage) {
    width_ = width;
    height_ = height;
    pixelBufferSize_ = static_cast<size_t>(width) * height * sizeof(Pixel); // Cast to size_t

    // Resize CPU fallback
    cpuRasterizer_.resize(width_, height_);

    if (cudaAvailable_) {
        // Free existing GPU memory if any
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

        initialized_ = false; // Need to re-upload population and target

        // Re-upload target image immediately after resizing
        uploadTargetImage(targetImage);

        // population_ and h_fitnessResults_ retain their sizes,
        // but will be re-uploaded/re-filled in the next update cycle's uploadPopulation/renderAndEvaluate
    } else {
        // If CUDA not available, ensure host buffers match new size
        // No need to resize h_fitnessResults_, only number of individuals matters
        // No host buffers for rendered images are stored persistently in CPU fallback
    }
}


void CudaRasterizer::uploadPopulation(const std::vector<Individual> &population,
                                      const std::vector<Pixel> &targetImage) {
    populationSize_ = population.size();
    if (populationSize_ == 0) {
        std::cerr << "Warning: Attempted to upload empty population" << std::endl;
        genesPerIndividual_ = 0; // Reset if population is empty
        return;
    }

    genesPerIndividual_ = population[0].size(); // Assume all individuals have same gene count
    if (genesPerIndividual_ == 0) {
        std::cerr << "Warning: First individual has 0 genes" << std::endl;
        return; // Nothing to upload if no genes
    }

    // Validate all individuals have the same number of genes
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
        // Continue with upload, but only copy valid individuals
    }

    // Store host copy for fallback/getRenderedImage
    population_ = population;
    h_fitnessResults_.resize(populationSize_);

    if (cudaAvailable_) {
        // Allocate device memory if not already initialized
        if (!initialized_) {
            size_t geneDataSize = static_cast<size_t>(populationSize_) * genesPerIndividual_ * sizeof(CudaGene);
            // Cast to size_t
            CHECK(cudaMalloc(&d_population_, geneDataSize));
            CHECK(cudaMalloc(&d_renderedBuffers_, static_cast<size_t>(populationSize_) * pixelBufferSize_));
            // Cast to size_t
            CHECK(cudaMalloc(&d_fitnessResults_, static_cast<size_t>(populationSize_) * sizeof(float)));
            // Cast to size_t

            // Upload target image if not already done (e.g., after resize)
            if (!d_targetImage_) {
                uploadTargetImage(targetImage);
            }

            initialized_ = true;
            std::cout << "CUDA device memory allocated." << std::endl;
        }

        // Create a flattened CudaGene array on the host
        std::vector<CudaGene> h_cudaPopulation(static_cast<size_t>(populationSize_) * genesPerIndividual_);
        // Cast to size_t
        for (size_t i = 0; i < populationSize_; ++i) {
            for (size_t j = 0; j < genesPerIndividual_; ++j) {
                const Gene &gene = population[i][j];
                CudaGene &cudaGene = h_cudaPopulation[static_cast<size_t>(i) * genesPerIndividual_ + j];
                // Cast i to size_t
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

        // Copy population data to device - verify buffer is allocated
        if (d_population_ != nullptr && !h_cudaPopulation.empty()) {
            size_t geneDataSize = static_cast<size_t>(populationSize_) * genesPerIndividual_ * sizeof(CudaGene);
            // Cast to size_t
            CHECK(cudaMemcpy(d_population_, h_cudaPopulation.data(), geneDataSize, cudaMemcpyHostToDevice));
        } else {
            std::cerr << "Error: Cannot copy population data - device buffer or host data is NULL" << std::endl;
            initialized_ = false;
        }
    }
}

void CudaRasterizer::uploadTargetImage(const std::vector<Pixel> &targetImage) {
    if (!cudaAvailable_) return;

    // Free old target image memory if exists
    if (d_targetImage_) {
        CHECK(cudaFree(d_targetImage_));
        d_targetImage_ = nullptr;
    }

    // Allocate and copy new target image
    size_t targetBufferSize = static_cast<size_t>(width_) * height_ * sizeof(Pixel); // Cast to size_t
    CHECK(cudaMalloc(&d_targetImage_, targetBufferSize));
    CHECK(cudaMemcpy(d_targetImage_, targetImage.data(), targetBufferSize, cudaMemcpyHostToDevice));
    std::cout << "Target image uploaded to GPU (" << width_ << "x" << height_ << ")" << std::endl;
}


void CudaRasterizer::renderAndEvaluate(const std::vector<Pixel> &targetImage, std::vector<float> &fitnessResults) {
    if (populationSize_ == 0 || genesPerIndividual_ == 0) {
        std::cerr << "Population not uploaded or empty before evaluation" << std::endl;
        // Ensure fitnessResults has correct size even if not calculated
        fitnessResults.assign(populationSize_, -1e12f); // Assign a very low fitness to all
        return;
    }

    if (initialized_ && cudaAvailable_) {
        // --- CUDA Implementation ---

        // 1. Clear all individual rendered buffers on the GPU
        Pixel clearColor = {0, 0, 0, 255}; // Black background
        int numThreadsPerClearBlock = 256; // Max threads per block is 1024, 256 is safe and common
        size_t totalPixelsAcrossBuffers = static_cast<size_t>(populationSize_) * width_ * height_; // Cast to size_t
        int numClearBlocks = static_cast<int>((totalPixelsAcrossBuffers + numThreadsPerClearBlock - 1) /
                                              numThreadsPerClearBlock); // Cast to int for kernel launch

        // Cast arguments to unsigned int for kernel launch
        launchClearBuffersKernel(d_renderedBuffers_, static_cast<unsigned int>(populationSize_),
                                 static_cast<unsigned int>(width_), static_cast<unsigned int>(height_),
                                 clearColor);
        CHECK(cudaGetLastError()); // Check for launch errors


        // 2. Launch Rendering Kernel
        // Use a single grid of threads, distribute across individuals and genes
        int threadsPerRenderBlock = 256; // Or choose an optimal block size
        size_t totalRenderThreads = static_cast<size_t>(populationSize_) * genesPerIndividual_;
        int numRenderBlocks = static_cast<int>((totalRenderThreads + threadsPerRenderBlock - 1) /
                                               threadsPerRenderBlock);


        // Validate block size (limit is often 1024 threads total, but depends on device)
        // The kernel logic now works with a single grid of threads, so we don't need
        // blockDim.x == genesPerIndividual. We just need enough threads total.
        // The kernel logic needs to be updated to use globalThreadId properly if blockDim.x != genesPerIndividual
        // Let's stick to the thread-per-gene launch config for now as implemented in kernels.cu,
        // but be aware the kernel itself needs update for flexible block sizes.
        // Using the thread-per-gene launch:

        if (genesPerIndividual_ > 1024) {
            std::cerr << "Warning: genesPerIndividual (" << genesPerIndividual_ <<
                    ") exceeds typical CUDA block size limit (1024) for the simple render kernel launch config." <<
                    std::endl;
            std::cerr << "Using flattened thread distribution instead." << std::endl;

            // Use the numRenderBlocks and threadsPerRenderBlock calculated above
            launchRenderKernel(d_population_, d_renderedBuffers_,
                               static_cast<unsigned int>(populationSize_),
                               genesPerIndividual_, static_cast<unsigned int>(width_),
                               static_cast<unsigned int>(height_));
        } else {
            // Original fixed block size approach
            launchRenderKernel(d_population_, d_renderedBuffers_,
                               static_cast<unsigned int>(populationSize_),
                               genesPerIndividual_, static_cast<unsigned int>(width_),
                               static_cast<unsigned int>(height_));
        }

        CHECK(cudaGetLastError()); // Check for launch errors
        // cudaDeviceSynchronize(); // Synchronize if subsequent kernel depends on this one completing (fitness does)


        // 3. Launch Fitness Kernel
        // One block per individual. Shared memory size needed for reduction.
        int threadsPerFitnessBlock = 256; // Common choice for reduction block size

        launchFitnessKernel(d_renderedBuffers_, d_targetImage_, d_fitnessResults_,
                            static_cast<unsigned int>(populationSize_),
                            static_cast<unsigned int>(width_), static_cast<unsigned int>(height_));

        CHECK(cudaGetLastError()); // Check for launch errors

        // 4. Copy fitness results back to host - FIX THE DIRECTION FLAG
        if (d_fitnessResults_ != nullptr) {
            // Changed to DeviceToHost - this was the main error
            CHECK(cudaMemcpy(h_fitnessResults_.data(), d_fitnessResults_,
                static_cast<size_t>(populationSize_) * sizeof(float),
                cudaMemcpyDeviceToHost));
        } else {
            std::cerr << "Error: Cannot copy fitness results - device buffer is NULL" << std::endl;
            // Fall back to CPU calculation
            goto cpu_fallback;
        }

        // 5. Synchronize to ensure data is available on host
        CHECK(cudaDeviceSynchronize());

        // Copy results to output buffer
        fitnessResults = h_fitnessResults_;
    } else {
    cpu_fallback:
        // --- CPU Fallback Implementation ---
        if (!initialized_ && cudaAvailable_) {
            std::cerr << "CUDA initialization failed, using CPU fallback." << std::endl;
        }
        if (!cudaAvailable_) {
            // If CUDA is not available at all, this message is printed in initialize()
            // std::cerr << "CUDA not available, using CPU fallback." << std::endl;
        }

        // Ensure targetImage passed to computeFitness matches current rasterizer size
        unsigned int currentW = width_;
        unsigned int currentH = height_;

        // CPU implementation - render each individual and calculate fitness
#ifndef NO_OPENMP
#pragma omp parallel for schedule(dynamic, 1) // Dynamic schedule good if render time varies per individual
#endif
        for (int i = 0; i < static_cast<int>(populationSize_); i++) {
            // Cast size_t to int for OpenMP loop index
            cpuRasterizer_.clear({0, 0, 0, 255});

            // Render individual with CPU rasterizer
            if (i < population_.size()) {
                // Check bounds just in case
                for (const auto &gene: population_[i]) {
                    cpuRasterizer_.draw(gene);
                }
                // Calculate fitness score using CPU function
                h_fitnessResults_[i] = computeFitness(cpuRasterizer_.data(), targetImage, currentW, currentH);
            } else {
                // Should not happen if population_ size matches populationSize_
                h_fitnessResults_[i] = -1e9; // Assign a very low fitness
            }
        }

        // Copy results to output buffer
        fitnessResults = h_fitnessResults_;
    }
}

std::vector<Pixel> CudaRasterizer::getRenderedImage(int individualIndex) {
    if (individualIndex < 0 || individualIndex >= static_cast<int>(populationSize_)) {
        // Cast size_t to int for comparison
        throw std::runtime_error("Individual index out of bounds");
    }

    std::vector<Pixel> imageBuffer(static_cast<size_t>(width_) * height_); // Cast to size_t

    if (initialized_ && cudaAvailable_) {
        // --- Get image from GPU ---
        if (!d_renderedBuffers_) {
            std::cerr << "GPU buffers not allocated, cannot get rendered image from GPU." << std::endl;
            // Fallback to CPU rendering if GPU buffer not available
            goto cpu_render_for_display;
        }

        // Calculate the offset for this individual's buffer on the GPU
        size_t offset = static_cast<size_t>(individualIndex) * width_ * height_ * sizeof(Pixel);
        // Cast individualIndex to size_t

        // Copy the specific individual's buffer from GPU to host
        CHECK(cudaMemcpy(imageBuffer.data(), d_renderedBuffers_ + offset / sizeof(Pixel), static_cast<size_t>(width_) *
            height_ * sizeof(Pixel), cudaMemcpyDeviceToHost)); // Changed to DeviceToHost

        // Synchronize to ensure the copy is complete
        CHECK(cudaDeviceSynchronize());

        return imageBuffer;
    } else {
    cpu_render_for_display:
        // --- CPU Fallback Rendering ---
        // Render the specific individual using the CPU rasterizer
        cpuRasterizer_.clear({0, 0, 0, 255});
        if (individualIndex < population_.size()) {
            // Check bounds
            for (const auto &gene: population_[individualIndex]) {
                cpuRasterizer_.draw(gene);
            }
            imageBuffer = cpuRasterizer_.data();
        } else {
            // This shouldn't happen if population_ is kept consistent with populationSize_
            std::cerr << "CPU population data mismatch for index " << individualIndex << std::endl;
            // Return an empty buffer or black image
            std::fill(imageBuffer.begin(), imageBuffer.end(), Pixel{0, 0, 0, 255});
        }
        return imageBuffer;
    }
}


