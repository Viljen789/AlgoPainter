//
// Created by vilje on 05/05/2025.
//

// This file must be compiled with NVCC (NVIDIA CUDA Compiler)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h> // If using half precision (optional)

#include "Pixel.h" // Include the shared Pixel struct
#include "CudaRasterizer.h" // Include CudaGene struct definition

// Make sure Gene::Shape enum values are consistent
#define SHAPE_CIRCLE 0
#define SHAPE_TRIANGLE 1
#define SHAPE_SQUARE 2

// Alpha blending function optimized for integer math on device
// Renamed to avoid potential conflicts if other libraries define alphaBlend
__device__ inline void blendPixels_device(Pixel &background, const Pixel &foreground) {
    // Ensure foreground alpha is not 0 or 255 for the blend calculation
    if (foreground.a == 0) return;
    if (foreground.a == 255) {
        background = foreground;
        return;
    }

    // Use integer arithmetic for blending
    unsigned int alpha = foreground.a;
    unsigned int invAlpha = 255 - alpha;

    // Using unsigned int for intermediate calculations to avoid overflow before shifting
    unsigned int r = (unsigned int) background.r * invAlpha + (unsigned int) foreground.r * alpha;
    unsigned int g = (unsigned int) background.g * invAlpha + (unsigned int) foreground.g * alpha;
    unsigned int b = (unsigned int) background.b * invAlpha + (unsigned int) foreground.b * alpha;


    background.r = static_cast<uint8_t>(r >> 8);
    background.g = static_cast<uint8_t>(g >> 8);
    background.b = static_cast<uint8_t>(b >> 8);
    // Alpha channel of the canvas might not be strictly necessary to blend,
    // as we only care about the final blended color against the target.
    // For simplicity, we can ignore updating the canvas alpha here.
    // background.a = static_cast<uint8_t>((background.a * invAlpha + foreground.a * alpha) >> 8);
}

// --- Rendering Kernel ---
// Renders all genes for ONE individual.
// Each block is responsible for one individual's image buffer.
// Threads within the block collaborate (or each thread draws one gene).
// Simple approach: Each thread draws one gene into its individual's buffer.
// This is inefficient as multiple threads draw into the same buffer,
// but avoids complex coordination for geometric rasterization.
// A more advanced kernel would parallelize the rasterization of *one* shape
// across threads within a block, or assign regions to thread blocks.
__global__ void renderKernel_simple(const CudaGene *population, Pixel *renderedBuffers,
                                    unsigned int numIndividuals, unsigned int genesPerIndividual,
                                    unsigned int imgWidth, unsigned int imgHeight) {
    // Use gridDim.x and blockDim.x to calculate global thread index across all individuals and genes
    // Assuming gridDim.x = numIndividuals, blockDim.x = genesPerIndividual
    // unsigned int individualIdx = blockIdx.x;
    // unsigned int geneIdx = threadIdx.x;

    // Let's use a single grid of threads and calculate individual and gene index from global thread ID
    // This is more flexible than assuming specific block/grid dimensions in the kernel itself
    // Requires launching the kernel with enough total threads (numIndividuals * genesPerIndividual)
    unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int individualIdx = globalThreadId / genesPerIndividual;
    unsigned int geneIdx = globalThreadId % genesPerIndividual;


    if (individualIdx >= numIndividuals || geneIdx >= genesPerIndividual) {
        return; // Out of bounds
    }

    // Get the gene this thread is responsible for
    const CudaGene &gene = population[static_cast<size_t>(individualIdx) * genesPerIndividual + geneIdx];
    // Use size_t for index calculation
    const Pixel geneColor = {gene.r, gene.g, gene.b, gene.a};

    // Get the base address for this individual's output buffer
    Pixel *outputBuffer = renderedBuffers + static_cast<size_t>(individualIdx) * imgWidth * imgHeight;
    // Use size_t for index calculation

    // Simplified Rasterization Logic on the GPU
    // This is a direct porting attempt and might be slow or have artifacts.
    // Optimized GPU rasterization is significantly more complex.

    switch (gene.type) {
        case SHAPE_CIRCLE: {
            int cx = static_cast<int>(gene.posX);
            int cy = static_cast<int>(gene.posY);
            int r = static_cast<int>(gene.size);
            int r2 = r * r;

            int y0 = max(0, cy - r);
            int y1 = min((int) imgHeight, cy + r + 1); // +1 for exclusive end

            for (int y = y0; y < y1; ++y) {
                int dy = y - cy;
                int dx_squared = r2 - dy * dy;
                if (dx_squared >= 0) {
                    // Check if there's a valid x span
                    int dx = static_cast<int>(sqrtf(static_cast<float>(dx_squared))); // Cast dx_squared to float
                    int x0 = max(0, cx - dx);
                    int x1 = min((int) imgWidth, cx + dx + 1); // +1 for exclusive end

                    for (int x = x0; x < x1; ++x) {
                        // Calculate linear index
                        int pixelIdx = y * imgWidth + x;
                        // Perform alpha blending (requires atomics if multiple genes overlap heavily and threads write to same pixel)
                        // For simplicity here, we just write. Atomicity would be needed for true correctness if threads overlap.
                        blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor); // Cast to size_t
                    }
                }
            }
            break;
        }
        case SHAPE_SQUARE: {
            int x = static_cast<int>(gene.posX - gene.size / 2.0f);
            int y = static_cast<int>(gene.posY - gene.size / 2.0f);
            int w = static_cast<int>(gene.size);
            int h = static_cast<int>(gene.size);

            int x0 = max(0, x);
            int y0 = max(0, y);
            int x1 = min((int) imgWidth, x + w);
            int y1 = min((int) imgHeight, y + h);

            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    int pixelIdx = yy * imgWidth + xx;
                    // Simple write (potential race if genes overlap)
                    blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor); // Cast to size_t
                }
            }
            break;
        }
        case SHAPE_TRIANGLE: {
            // AABB of the triangle
            float minXf = fminf(fminf(gene.posX, gene.posX + gene.size), gene.posX + gene.size / 2.0f);
            float maxXf = fmaxf(fmaxf(gene.posX, gene.posX + gene.size), gene.posX + gene.size / 2.0f);
            float minYf = fminf(fminf(gene.posY, gene.posY), gene.posY + gene.size * 0.866f);
            float maxYf = fmaxf(fmaxf(gene.posY, gene.posY), gene.posY + gene.size * 0.866f);

            int x0 = max(0, (int) floorf(minXf));
            int x1 = min((int) imgWidth, (int) ceilf(maxXf));
            int y0 = max(0, (int) floorf(minYf));
            int y1 = min((int) imgHeight, (int) ceilf(maxYf));

            // Triangle vertices relative to gene.pos
            float p0x = 0, p0y = 0;
            float p1x = gene.size, p1y = 0;
            float p2x = gene.size / 2.0f, p2y = gene.size * 0.866f;

            // Barycentric coordinates setup
            float area = (p1x - p0x) * (p2y - p0y) - (p2x - p0x) * (p1y - p0y);
            // Handle degenerate triangle (area is 0)
            if (fabs(area) < 1e-6f) break; // Use fabs for float, use float literal for epsilon
            float invArea = 1.0f / area;

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    // Point to test (center of pixel)
                    float px = (float) x - gene.posX + 0.5f;
                    float py = (float) y - gene.posY + 0.5f;

                    // Calculate barycentric coordinates
                    float w2 = ((p0x - p2x) * (py - p2y) - (p0y - p2y) * (px - p2x)) * invArea;
                    float w0 = ((p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x)) * invArea;
                    float w1 = 1.0f - w0 - w2;

                    // Check if point is inside triangle (with small epsilon for floating point inaccuracies)
                    // Check against 0.0f and use f >= 0.0f
                    if (w0 >= -1e-3f && w1 >= -1e-3f && w2 >= -1e-3f) {
                        // Use float literals for epsilon
                        int pixelIdx = y * imgWidth + x;
                        blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor); // Cast to size_t
                    }
                }
            }
            break;
        }
    }
}

// --- Fitness Kernel ---
// Computes fitness for all individuals.
// Each thread block computes the fitness for one individual.
// Threads within the block perform a parallel reduction (sum).
__global__ void fitnessKernel(const Pixel *renderedBuffers, const Pixel *targetImage,
                              float *fitnessResults, unsigned int numIndividuals,
                              unsigned int imgWidth, unsigned int imgHeight) {
    unsigned int individualIdx = blockIdx.x; // Each block handles one individual

    if (individualIdx >= numIndividuals) {
        return; // Out of bounds
    }

    // Base addresses for this individual's data
    const Pixel *renderedBuffer = renderedBuffers + static_cast<size_t>(individualIdx) * imgWidth * imgHeight;
    // Cast to size_t
    // Target image is the same for all individuals: const Pixel* targetImage

    unsigned int totalPixels = imgWidth * imgHeight;

    // Shared memory for reduction within the block
    extern __shared__ float sdata[]; // Declared dynamically

    // Each thread loads a pixel difference sum into shared memory
    float mySum = 0.0f;
    // Loop through pixels using global thread index within the block's assigned buffer
    for (unsigned int i = threadIdx.x; i < totalPixels; i += blockDim.x) {
        const Pixel &c1 = renderedBuffer[i];
        const Pixel &c2 = targetImage[i]; // Assuming targetImage is contiguous

        int dr = static_cast<int>(c1.r) - c2.r;
        int dg = static_cast<int>(c1.g) - c2.g;
        int db = static_cast<int>(c1.b) - c2.b;

        mySum -= static_cast<float>(dr * dr + dg * dg + db * db);
        // Negative sum for higher fitness = lower difference, explicit cast
    }

    sdata[threadIdx.x] = mySum;

    __syncthreads();

    // Perform reduction in shared memory
    // This loop is unrolled by the compiler for common block sizes
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The result for this block is in sdata[0]
    if (threadIdx.x == 0) {
        fitnessResults[individualIdx] = sdata[0];
    }
}

// --- Kernel for Clearing Render Buffers ---
__global__ void clearBuffersKernel(Pixel *renderedBuffers, unsigned int numIndividuals, unsigned int imgWidth,
                                   unsigned int imgHeight, Pixel clearColor) {
    // Calculate global pixel index across all buffers
    size_t totalPixelsPerBuffer = static_cast<size_t>(imgWidth) * imgHeight;
    size_t globalPixelIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; // Cast blockIdx.x to size_t

    size_t totalPixelsAcrossBuffers = totalPixelsPerBuffer * numIndividuals;
    // Multiplication order shouldn't matter if total fits size_t

    if (globalPixelIdx < totalPixelsAcrossBuffers) {
        renderedBuffers[globalPixelIdx] = clearColor;
    }
}

// --- Kernel Launcher Functions ---
// Direct implementations - no need for external declarations since kernels are in this file

void CudaRasterizer::launchRenderKernel(CudaGene *d_population,
                                        Pixel *d_renderedBuffers,
                                        unsigned int populationSize,
                                        unsigned int genesPerIndividual,
                                        unsigned int width,
                                        unsigned int height) {
    // Calculate grid and block dimensions
    const int threadsPerBlock = 256;
    int totalThreadsNeeded = populationSize * genesPerIndividual;
    int blocksNeeded = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    renderKernel_simple<<<blocksNeeded, threadsPerBlock>>>(
        d_population, d_renderedBuffers, populationSize,
        genesPerIndividual, width, height
    );
}

void CudaRasterizer::launchFitnessKernel(Pixel *d_renderedBuffers,
                                         Pixel *d_targetImage,
                                         float *d_fitnessResults,
                                         unsigned int populationSize,
                                         unsigned int width,
                                         unsigned int height) {
    // Each block handles one individual, threads within block perform reduction
    const int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    fitnessKernel<<<populationSize, threadsPerBlock, sharedMemSize>>>(
        d_renderedBuffers, d_targetImage, d_fitnessResults,
        populationSize, width, height
    );
}

void CudaRasterizer::launchClearBuffersKernel(Pixel *d_renderedBuffers,
                                              unsigned int populationSize,
                                              unsigned int width,
                                              unsigned int height,
                                              Pixel clearColor) {
    // Calculate grid and block dimensions
    const int threadsPerBlock = 256;
    size_t totalPixels = static_cast<size_t>(width) * height * populationSize;
    int blocksNeeded = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    clearBuffersKernel<<<blocksNeeded, threadsPerBlock>>>(
        d_renderedBuffers, populationSize, width, height, clearColor
    );
}
