#ifndef __openmp_h__
#define __openmp_h__

#include <stdint.h>

#include "internal/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate and return the sum of the absolute difference of all ordered pairs within the array x
 *
 * @param x An array of random values
 * @param n The length of the array x
 * @return The sum of absolute differences of all ordered pairs within the array x
 */
uint64_t openmp_productofdifferences(const unsigned char *x, size_t n);
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
size_t openmp_removefactors(const int *input, size_t n, int divisor, int *output);
/**
 * @brief Calculate the chromatic aberration of the image pixels_in
 *
 * @param input An array of pixels, where each char represents a single colour channel of a single pixel contrasts of an image with width x height pixels
 * @param output A preallocated array to store the resulting chromatic aberration image with the same dimensions
 * @param width The width of image `input`
 * @param height The height of image `input`
 */
void openmp_chromaticaberration(const unsigned char *input, unsigned char *output, size_t width, size_t height);

#ifdef __cplusplus
}
#endif

#endif // __openmp_h__
