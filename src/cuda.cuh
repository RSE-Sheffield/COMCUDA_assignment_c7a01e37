#ifndef __cuda_cuh__
#define __cuda_cuh__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "internal/common.h"

/**
 * @brief Calculate and return the sum of the absolute difference of all ordered pairs within the array x
 *
 * @param x An array of random values
 * @param n The length of the array x
 * @return The sum of absolute differences of all ordered pairs within the array x
 */
uint64_t cuda_productofdifferences(const unsigned char *x, size_t n);
/**
 * @brief Remove all factors of divisor from input
 *
 * Remove all factors of divisor from the array input
 * whilst retaining it's order and store the result in output
 * and return the new length.
 *
 * @param input An array of integer values
 * @param n The length of the array input
 * @param divisor The divisor to test whether values in input are a factor of
 * @return The length of the array stored in output
 */
size_t cuda_removefactors(const int *input, size_t n, int divisor, int *output);
/**
 * @brief Calculate the chromatic aberration of the image pixels_in
 *
 * @param input An array of pixels, where each char represents a single colour channel of a single pixel contrasts of an image with width x height pixels
 * @param output A preallocated array to store the resulting chromatic aberration image with the same dimensions
 * @param width The width of image `input`
 * @param height The height of image `input`
 */
void cuda_chromaticaberration(const unsigned char *input, unsigned char *output, size_t width, size_t height);


/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on failure
 * e.g. CUDA_CALL(cudaFree(myPtr));
 * CUDA_CHECk() can also be used to perform error checking after kernel launches and async methods
 * e.g. CUDA_CHECK()
 */
#if defined(_DEBUG) || defined(D_DEBUG)
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); }
#else
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif // __cuda_cuh__
