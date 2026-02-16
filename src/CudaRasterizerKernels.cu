#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "Pixel.h"
#include "CudaRasterizer.h"

#define SHAPE_CIRCLE 0
#define SHAPE_RECTANGLE 1
#define SHAPE_TRIANGLE 2

__device__ inline void blendPixels_device(Pixel &background, const Pixel &foreground) {
    if (foreground.a == 0) return;
    if (foreground.a == 255) {
        background = foreground;
        return;
    }
    unsigned int alpha = foreground.a;
    unsigned int invAlpha = 255 - alpha;
    background.r = static_cast<uint8_t>((static_cast<unsigned int>(background.r) * invAlpha + static_cast<unsigned int>(foreground.r) * alpha) >> 8);
    background.g = static_cast<uint8_t>((static_cast<unsigned int>(background.g) * invAlpha + static_cast<unsigned int>(foreground.g) * alpha) >> 8);
    background.b = static_cast<uint8_t>((static_cast<unsigned int>(background.b) * invAlpha + static_cast<unsigned int>(foreground.b) * alpha) >> 8);
}

__device__ inline float cross_product(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__global__ void renderKernel_perPixel(const CudaGene *population, Pixel *renderedBuffers,
                                      unsigned int numIndividuals, unsigned int genesPerIndividual,
                                      unsigned int imgWidth, unsigned int imgHeight) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int individualIdx = blockIdx.z;

    if (x >= imgWidth || y >= imgHeight || individualIdx >= numIndividuals) return;

    const CudaGene *individualGenes = population + (static_cast<size_t>(individualIdx) * genesPerIndividual);
    
    float sampleOffsets[4][2] = {{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}};
    Pixel samples[4];
    for(int s=0; s<4; ++s) samples[s] = {0, 0, 0, 255};

    float px = static_cast<float>(x);
    float py = static_cast<float>(y);

    for (unsigned int i = 0; i < genesPerIndividual; ++i) {
        const CudaGene &gene = individualGenes[i];
        const Pixel geneColor = {gene.r, gene.g, gene.b, gene.a};
        
        float rad = gene.rotation * (3.14159265f / 180.0f);
        float cosA = cosf(rad);
        float sinA = sinf(rad);

        for (int s = 0; s < 4; ++s) {
            float fx = px + sampleOffsets[s][0];
            float fy = py + sampleOffsets[s][1];
            
            // Transform to local coordinates
            float dx = fx - gene.posX;
            float dy = fy - gene.posY;
            float lx = dx * cosA + dy * sinA;
            float ly = -dx * sinA + dy * cosA;
            
            bool inside = false;
            if (gene.type == SHAPE_CIRCLE) {
                if (lx * lx + ly * ly <= gene.sizeX * gene.sizeX) inside = true;
            } else if (gene.type == SHAPE_RECTANGLE) {
                if (fabsf(lx) <= gene.sizeX * 0.5f && fabsf(ly) <= gene.sizeY * 0.5f) inside = true;
            } else if (gene.type == SHAPE_TRIANGLE) {
                // Approximate equilateral triangle in local space
                float s = gene.sizeX;
                float x0 = -s*0.5f, y0 = -s*0.288f;
                float x1 = s*0.5f,  y1 = -s*0.288f;
                float x2 = 0,       y2 = s*0.577f;
                if (is_inside_triangle(lx, ly, x0, y0, x1, y1, x2, y2)) inside = true;
            }

            if (inside) blendPixels_device(samples[s], geneColor);
        }
    }

    size_t pixelIdx = static_cast<size_t>(individualIdx) * imgWidth * imgHeight + (y * imgWidth + x);
    renderedBuffers[pixelIdx] = {
        static_cast<uint8_t>((static_cast<int>(samples[0].r) + samples[1].r + samples[2].r + samples[3].r) >> 2),
        static_cast<uint8_t>((static_cast<int>(samples[0].g) + samples[1].g + samples[2].g + samples[3].g) >> 2),
        static_cast<uint8_t>((static_cast<int>(samples[0].b) + samples[1].b + samples[2].b + samples[3].b) >> 2),
        255
    };
}

// ... Rest of the file (fitnessKernel, clearBuffersKernel, launch functions) remain identical to the previous version
// I will rewrite them to ensure completeness.

__global__ void fitnessKernel(const Pixel *renderedBuffers, const Pixel *targetImage,
                              float *fitnessResults, unsigned int numIndividuals,
                              unsigned int imgWidth, unsigned int imgHeight) {
    unsigned int individualIdx = blockIdx.x;
    if (individualIdx >= numIndividuals) return;
    const Pixel *renderedBuffer = renderedBuffers + static_cast<size_t>(individualIdx) * imgWidth * imgHeight;
    unsigned int totalPixels = imgWidth * imgHeight;
    extern __shared__ float sdata[];
    float mySum = 0.0f;
    for (unsigned int i = threadIdx.x; i < totalPixels; i += blockDim.x) {
        const Pixel &c1 = renderedBuffer[i];
        const Pixel &c2 = targetImage[i];
        int dr = (int)c1.r - c2.r; int dg = (int)c1.g - c2.g; int db = (int)c1.b - c2.b;
        mySum -= (float)(dr * dr + dg * dg + db * db);
    }
    sdata[threadIdx.x] = mySum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) fitnessResults[individualIdx] = sdata[0];
}

__global__ void clearBuffersKernel(Pixel *renderedBuffers, unsigned int numIndividuals, unsigned int imgWidth,
                                   unsigned int imgHeight, Pixel clearColor) {
    size_t totalPixelsAcrossBuffers = (size_t)imgWidth * imgHeight * numIndividuals;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixelsAcrossBuffers) renderedBuffers[idx] = clearColor;
}

void CudaRasterizer::launchRenderKernel(CudaGene *d_population, Pixel *d_renderedBuffers, unsigned int populationSize, unsigned int genesPerIndividual, unsigned int width, unsigned int height) {
    dim3 threads(16, 16, 1);
    dim3 blocks((width + 15) / 16, (height + 15) / 16, populationSize);
    renderKernel_perPixel<<<blocks, threads>>>(d_population, d_renderedBuffers, populationSize, genesPerIndividual, width, height);
}

void CudaRasterizer::launchFitnessKernel(Pixel *d_renderedBuffers, Pixel *d_targetImage, float *d_fitnessResults, unsigned int populationSize, unsigned int width, unsigned int height) {
    fitnessKernel<<<populationSize, 256, 256 * sizeof(float)>>>(d_renderedBuffers, d_targetImage, d_fitnessResults, populationSize, width, height);
}

void CudaRasterizer::launchClearBuffersKernel(Pixel *d_renderedBuffers, unsigned int populationSize, unsigned int width, unsigned int height, Pixel clearColor) {
    size_t total = (size_t)width * height * populationSize;
    clearBuffersKernel<<<(total + 255) / 256, 256>>>(d_renderedBuffers, populationSize, width, height, clearColor);
}
