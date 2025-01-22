#include "cpu.h"
#include <stdlib.h>

uint64_t cpu_productofdifferences(const unsigned char *x, const size_t n) {
    // Initialise the result
    uint64_t result = 0;
    // Iterate the array x, skipping the first element
    for (size_t i = 1; i < n; ++i) {
        // Add the absolute difference of the current and previous element to result
        result += abs((int)x[i - 1] - (int)x[i]);
    }
    return result;
}

size_t cpu_removefactors(const int *input, const size_t n, const int divisor, int *output) {
    // Initialise the length of the array output
    size_t return_len = 0;
    // Iterate the array input
    for (size_t i = 0; i < n; ++i) {
        // Test whether the element should be kept
        if (input[i] % divisor != 0) {
            // Add the absolute difference of the current and previous element to result
            output[return_len] = input[i];
            // Increment the length
            ++return_len;
        }
    }
    return return_len;
}

void cpu_chromaticaberration(const unsigned char *input, unsigned char *output, const size_t width, const size_t height) {
    // Iterate the image's pixels
    for (unsigned int x = 0; x < width; ++x) {
        for (unsigned int y = 0; y < height; ++x) {
            // Calculate normalised distance of pixel from center of image
            const float distance_x = x - (width / 2.0f);
            const float distance_y = y - (height / 2.0f);
            const float distance_xy = sqrtf(powf(distance_x, 2) + powf(distance_y, 2));
            const float norm_distance = distance_xy /max(width, height);
            // Process each channel individually
            for (unsigned int channel = 0; channel < 3; ++channel) {
                // Scale displacement for each channel
                const float displacement_x = distance_x * FACTOR * norm_distance * CHANNEL_FACTORS[channel];
                const float displacement_y = distance_y * FACTOR * norm_distance * CHANNEL_FACTORS[channel];
                // Bilinear sample the image, with the floating point coordinates
                unsigned int x0 = (unsigned int)floorf(displacement_x);
                unsigned int x1 = x0 + 1;
                unsigned int y0 = (unsigned int)floorf(displacement_y);
                unsigned int y1 = y0 + 1;
                const float wx = displacement_x - x0;
                const float wy = displacement_y - y0;
                // Clamp offset pixels into bounds
                x0 = min(max(x0, 0), width);
                x1 = min(max(x1, 0), width);
                y0 = min(max(y0, 0), height);
                y1 = min(max(y1, 0), height);
                // Read/Write weighted pixel data
                output[CHANNELS * width * y + CHANNELS * x + channel] = (unsigned char)
                   (input[CHANNELS * width * y0 + CHANNELS * x0 + channel] * (1 - wx) * (1 - wy) +
                    input[CHANNELS * width * y0 + CHANNELS * x1 + channel] * wx * (1 - wy) +
                    input[CHANNELS * width * y1 + CHANNELS * x0 + channel] * (1 - wx) * wy +
                    input[CHANNELS * width * y1 + CHANNELS * x1 + channel] * wx * wy);
            }
        }
    }
}
