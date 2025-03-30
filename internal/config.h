#ifndef __config_h__
#define __config_h__

#include <math.h>

/**
 * The number of channels in valid images
 */
#define CHANNELS 3
/**
 * A factor by which displacement scales based on distance from center
 */
#define FACTOR 0.1f
/**
 * Unique per channel factors
 * @note C does not support modern C++ constexpr
 * so we define it two different ways that won't upset the linker
 * Host code sees an ugly macro
 * CUDA device code sees a modern c++ constant expression
 */
#ifndef __CUDACC__
#define CHANNEL_FACTORS (float[]){ 1.2f, 0.8f, 1.0f }
#else
__device__ constexpr float CHANNEL_FACTORS[3] = { 1.2f, 0.8f, 1.0f };
#endif

/**
 * Number of runs to complete for benchmarking
 */
#define BENCHMARK_RUNS 100

/**
 * ANSI colour codes used for console output
 */
#define CONSOLE_RED "\x1b[91m"
#define CONSOLE_GREEN "\x1b[92m"
#define CONSOLE_YELLOW "\x1b[93m"
#define CONSOLE_BLUE "\x1b[94m"
#define CONSOLE_RESET "\x1b[39m"

#endif  // __config_h__
