#include "cpu.h"

#include <math.h>
#include <stdio.h>
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
            // Store the value to be kept at the next index in the return buffer
            output[return_len] = input[i];
            // Increment the counter of how many items are in the return buffer
            ++return_len;
        }
    }
    return return_len;
}

void cpu_chromaticaberration(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height) {
    // Iterate the image's pixels
    for (unsigned int x = 0; x < width; ++x) {
        for (unsigned int y = 0; y < height; ++y) {
            // Calculate normalised distance of pixel from center of image
            const float distance_x = (x - (width / 2.0f));
            const float distance_y = y - (height / 2.0f);
            const float distance_xy = sqrtf(powf(distance_x, 2) + powf(distance_y, 2));
            const float norm_distance = distance_xy / fmaxf((float)width, (float)height);
            // Process each channel individually
            for (unsigned int channel = 0; channel < 3; ++channel) {
                // Scale displacement for each channel
                const float displacement_x = distance_x * FACTOR * norm_distance * CHANNEL_FACTORS[channel];
                const float displacement_y = distance_y * FACTOR * norm_distance * CHANNEL_FACTORS[channel];
                // Displaced sample coordinates
                float sample_x = x + displacement_x;
                float sample_y = y + displacement_y;
                sample_x = fminf(fmaxf(sample_x, 0), (float)width - 1);
                sample_y = fminf(fmaxf(sample_y, 0), (float)height - 1);
                // Bilinear sample the image, with the floating point coordinates
                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                // Clamp offset pixels into bounds (int max does not exist within c stdlib, avoid cast)
                x1 = (int)fminf(fmaxf((float)x1, 0), (float)width - 1);  // this is now implicit casting to/from floats
                y1 = (int)fminf(fmaxf((float)y1, 0), (float)height - 1);  // this is now implicit casting to/from floats
                // Read/Write weighted pixel data
                const float wx = sample_x - x0;
                const float wy = sample_y - y0;
                output[CHANNELS * width * y + CHANNELS * x + channel] = (unsigned char)
                    (input[CHANNELS * width * y0 + CHANNELS * x0 + channel] * (1 - wx) * (1 - wy) +
                     input[CHANNELS * width * y0 + CHANNELS * x1 + channel] * wx * (1 - wy) +
                     input[CHANNELS * width * y1 + CHANNELS * x0 + channel] * (1 - wx) * wy +
                     input[CHANNELS * width * y1 + CHANNELS * x1 + channel] * wx * wy);
            }
        }
    }
}
