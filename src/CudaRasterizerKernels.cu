#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "Pixel.h"
#include "CudaRasterizer.h"

#define SHAPE_CIRCLE 0
#define SHAPE_TRIANGLE 1
#define SHAPE_SQUARE 2

__device__ inline void blendPixels_device(Pixel &background, const Pixel &foreground) {
    if (foreground.a == 0) return;
    if (foreground.a == 255) {
        background = foreground;
        return;
    }

    unsigned int alpha = foreground.a;
    unsigned int invAlpha = 255 - alpha;

    unsigned int r = (unsigned int) background.r * invAlpha + (unsigned int) foreground.r * alpha;
    unsigned int g = (unsigned int) background.g * invAlpha + (unsigned int) foreground.g * alpha;
    unsigned int b = (unsigned int) background.b * invAlpha + (unsigned int) foreground.b * alpha;

    background.r = static_cast<uint8_t>(r >> 8);
    background.g = static_cast<uint8_t>(g >> 8);
    background.b = static_cast<uint8_t>(b >> 8);
}

__global__ void renderKernel_simple(const CudaGene *population, Pixel *renderedBuffers,
                                    unsigned int numIndividuals, unsigned int genesPerIndividual,
                                    unsigned int imgWidth, unsigned int imgHeight) {
    unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int individualIdx = globalThreadId / genesPerIndividual;
    unsigned int geneIdx = globalThreadId % genesPerIndividual;

    if (individualIdx >= numIndividuals || geneIdx >= genesPerIndividual) {
        return;
    }

    const CudaGene &gene = population[static_cast<size_t>(individualIdx) * genesPerIndividual + geneIdx];
    const Pixel geneColor = {gene.r, gene.g, gene.b, gene.a};

    Pixel *outputBuffer = renderedBuffers + static_cast<size_t>(individualIdx) * imgWidth * imgHeight;

    switch (gene.type) {
        case SHAPE_CIRCLE: {
            int cx = static_cast<int>(gene.posX);
            int cy = static_cast<int>(gene.posY);
            int r = static_cast<int>(gene.size);
            int r2 = r * r;

            int y0 = max(0, cy - r);
            int y1 = min((int) imgHeight, cy + r + 1);

            for (int y = y0; y < y1; ++y) {
                int dy = y - cy;
                int dx_squared = r2 - dy * dy;
                if (dx_squared >= 0) {
                    int dx = static_cast<int>(sqrtf(static_cast<float>(dx_squared)));
                    int x0 = max(0, cx - dx);
                    int x1 = min((int) imgWidth, cx + dx + 1);

                    for (int x = x0; x < x1; ++x) {
                        int pixelIdx = y * imgWidth + x;
                        blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor);
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
                    blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor);
                }
            }
            break;
        }
        case SHAPE_TRIANGLE: {
            float minXf = fminf(fminf(gene.posX, gene.posX + gene.size), gene.posX + gene.size / 2.0f);
            float maxXf = fmaxf(fmaxf(gene.posX, gene.posX + gene.size), gene.posX + gene.size / 2.0f);
            float minYf = fminf(fminf(gene.posY, gene.posY), gene.posY + gene.size * 0.866f);
            float maxYf = fmaxf(fmaxf(gene.posY, gene.posY), gene.posY + gene.size * 0.866f);

            int x0 = max(0, (int) floorf(minXf));
            int x1 = min((int) imgWidth, (int) ceilf(maxXf));
            int y0 = max(0, (int) floorf(minYf));
            int y1 = min((int) imgHeight, (int) ceilf(maxYf));

            float p0x = 0, p0y = 0;
            float p1x = gene.size, p1y = 0;
            float p2x = gene.size / 2.0f, p2y = gene.size * 0.866f;

            float area = (p1x - p0x) * (p2y - p0y) - (p2x - p0x) * (p1y - p0y);
            if (fabs(area) < 1e-6f) break;
            float invArea = 1.0f / area;

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    float px = (float) x - gene.posX + 0.5f;
                    float py = (float) y - gene.posY + 0.5f;

                    float w2 = ((p0x - p2x) * (py - p2y) - (p0y - p2y) * (px - p2x)) * invArea;
                    float w0 = ((p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x)) * invArea;
                    float w1 = 1.0f - w0 - w2;

                    if (w0 >= -1e-3f && w1 >= -1e-3f && w2 >= -1e-3f) {
                        int pixelIdx = y * imgWidth + x;
                        blendPixels_device(outputBuffer[static_cast<size_t>(pixelIdx)], geneColor);
                    }
                }
            }
            break;
        }
    }
}

__global__ void fitnessKernel(const Pixel *renderedBuffers, const Pixel *targetImage,
                              float *fitnessResults, unsigned int numIndividuals,
                              unsigned int imgWidth, unsigned int imgHeight) {
    unsigned int individualIdx = blockIdx.x;

    if (individualIdx >= numIndividuals) {
        return;
    }

    const Pixel *renderedBuffer = renderedBuffers + static_cast<size_t>(individualIdx) * imgWidth * imgHeight;

    unsigned int totalPixels = imgWidth * imgHeight;

    extern __shared__ float sdata[];

    float mySum = 0.0f;
    for (unsigned int i = threadIdx.x; i < totalPixels; i += blockDim.x) {
        const Pixel &c1 = renderedBuffer[i];
        const Pixel &c2 = targetImage[i];

        int dr = static_cast<int>(c1.r) - c2.r;
        int dg = static_cast<int>(c1.g) - c2.g;
        int db = static_cast<int>(c1.b) - c2.b;

        mySum -= static_cast<float>(dr * dr + dg * dg + db * db);
    }

    sdata[threadIdx.x] = mySum;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        fitnessResults[individualIdx] = sdata[0];
    }
}

__global__ void clearBuffersKernel(Pixel *renderedBuffers, unsigned int numIndividuals, unsigned int imgWidth,
                                   unsigned int imgHeight, Pixel clearColor) {
    size_t totalPixelsPerBuffer = static_cast<size_t>(imgWidth) * imgHeight;
    size_t globalPixelIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t totalPixelsAcrossBuffers = totalPixelsPerBuffer * numIndividuals;

    if (globalPixelIdx < totalPixelsAcrossBuffers) {
        renderedBuffers[globalPixelIdx] = clearColor;
    }
}

void CudaRasterizer::launchRenderKernel(CudaGene *d_population,
                                        Pixel *d_renderedBuffers,
                                        unsigned int populationSize,
                                        unsigned int genesPerIndividual,
                                        unsigned int width,
                                        unsigned int height) {
    const int threadsPerBlock = 256;
    int totalThreadsNeeded = populationSize * genesPerIndividual;
    int blocksNeeded = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

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
    const int threadsPerBlock = 256;
    size_t totalPixels = static_cast<size_t>(width) * height * populationSize;
    int blocksNeeded = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    clearBuffersKernel<<<blocksNeeded, threadsPerBlock>>>(
        d_renderedBuffers, populationSize, width, height, clearColor
    );
}
