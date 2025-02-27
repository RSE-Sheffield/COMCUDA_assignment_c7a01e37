#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <ctime>

#include "config.h"
#include "common.h"
#include "cpu.h"
#include "openmp.h"
#include "cuda.cuh"
#include "main.h"


namespace {
/**
 * Structure containing the options provided by runtime arguments
 */
struct CConfig {
    /**
      * Path to input file
      */
    char *input_file = nullptr;
    /**
      * Path to output file
      */
    char *output_file = nullptr;
    /**
      * Treated as boolean, program will operate in benchmark mode
      * This repeats the algorithm multiple times and returns an average time
      */
    bool benchmark;
}; typedef struct CConfig CConfig;
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char** argv, CConfig* config) {
    if (argc < 4 || argc > 7) {
        fprintf(stderr, CONSOLE_RED "Chromatic Aberration expects 3-6 arguments, %d provided.\n" CONSOLE_RESET, argc - 1);
        print_help(argv[0]);
    }
    // Clear config struct
    *config = {};
    // Iterate over remaining args    
    int i = 3;
    char* t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len = strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg)
            free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for (; argv[i][j]; ++j) {
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        // Benchmark
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg) || !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        // Input/Output file
        if (!strcmp(t_arg + arg_len - 5, ".png")) {
            if (config->input_file) {
                if (config->output_file) {
                    fprintf(stderr, "Multiple inputs/outputs were provided, this is not supported!\n");
                    print_help(argv[0]);
                } else {
                    // Allocate memory and copy
                    config->output_file = (char*)malloc(arg_len);
                    memcpy(config->output_file, argv[i], arg_len);
                    continue;
                }
            } else {
                // Allocate memory and copy
                config->input_file = (char*)malloc(arg_len);
                memcpy(config->input_file, argv[i], arg_len);
                continue;
            }
        } else {
            fprintf(stderr, ".png is the only supported input/output file format for the Chromatic Aberration algorithm!\n");
            print_help(argv[0]);
        }
        fprintf(stderr, "Unexpected chromatic aberration argument: %s\n", argv[i]);
        print_help(argv[0]);
    }
    if (!config->input_file) {
        fprintf(stderr, "Input file required\n");
        print_help(argv[0]);
    }
    if (t_arg)
        free(t_arg);
}
}

void runChromaticAberration(int argc, char** argv, const Implementation implementation) {
    CConfig config;
    parse_args(argc, argv, &config);

    // Inputs
    Image input_image;
    memset(&input_image, 0, sizeof(Image));
    
    // Load RGB input image
    printf("Using input file: %s%s%s\n", CONSOLE_YELLOW, config.input_file, CONSOLE_RESET);
    loadImage(config.input_file, &input_image);
    if (input_image.width < 3 || input_image.height < 3) {
        fprintf(stderr, "Input image dimensions too small!\n");
        exit(EXIT_FAILURE);
    }
    printf("Input has dimensions: %s%d%sx%s%d%s\n", CONSOLE_YELLOW, input_image.width, CONSOLE_RESET, CONSOLE_YELLOW, input_image.height, CONSOLE_RESET);

    
    // Create result for validation
    Image validation_image;
    validation_image.width = input_image.width;
    validation_image.height = input_image.height;
    validation_image.data = (unsigned char*)malloc(validation_image.width * validation_image.height * sizeof(unsigned char) * CHANNELS);
    cpu_chromaticaberration(input_image.data, validation_image.data, input_image.width, input_image.height);

    // Run student implementation
    float timing_log;
    Image result_image;
    result_image.width = input_image.width;
    result_image.height = input_image.height;
    result_image.data = (unsigned char*)malloc(result_image.width * result_image.height * sizeof(unsigned char) * CHANNELS);
    const int TOTAL_RUNS = config.benchmark ? BENCHMARK_RUNS : 1;
    {
        //Init for run  
        cudaEvent_t startT, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&stopT));
        // Run 1 or many times
        timing_log = 0.0f;
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1)
                printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            // Reset result image
            memset(result_image.data, 0, result_image.width * result_image.height * 3 * sizeof(unsigned char));
            // Run Chromatic Aberration algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (implementation) {
            case CPU:
                cpu_chromaticaberration(input_image.data, result_image.data, input_image.width, input_image.height);
                break;
            case OPENMP:
                openmp_chromaticaberration(input_image.data, result_image.data, input_image.width, input_image.height);
                break;
            case CUDA:
                cuda_chromaticaberration(input_image.data, result_image.data, input_image.width, input_image.height);
                break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log += milliseconds;
        }
        if (TOTAL_RUNS > 1)
            printf("\n");
        // Convert timing info to average
        timing_log /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);
    }

    // Validate and report
    {
        const unsigned int total_pixels = validation_image.width * validation_image.height;
        unsigned int errors = 0;
        for (unsigned int i = 0; i < total_pixels; ++i) {
            if (validation_image.data[i] != result_image.data[i]) {
                ++errors;
            }
        }
        printf("Chromatic Aberration Result: %s" CONSOLE_RESET "\n", errors ? CONSOLE_RED "Fail" : CONSOLE_GREEN "Pass");
        if (errors) {
            printf("\t%u/%u pixels wrong!" CONSOLE_RESET "\n", errors, total_pixels);
            printf("\t(Consider comparing output images)\n");
        }
    }

    // Export output
    if (config.output_file) {
        saveImage(config.output_file, result_image);
    }

    // Report timing information    
    printf("%s average execution timing from %d runs\n", implementation_to_string(implementation), TOTAL_RUNS);
    if (implementation == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf(CONSOLE_YELLOW "Code built as DEBUG, timing results are invalid!\n" CONSOLE_RESET);
#endif
    printf("Time: %.3fms\n", timing_log);

    // Cleanup
    free(result_image.data);
    free(validation_image.data);
    free(input_image.data);
    if (config.output_file)
        free(config.output_file);
    free(config.input_file);
}