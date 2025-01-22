#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#include <cuda_runtime.h>

#ifdef _MSC_VER
#include <windows.h>
#include <WinCon.h>
#endif

#include "main.h"

#include "config.h"

namespace {
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char** argv, MainConfig* config) {
    // Clear config struct
    memset(config, 0, sizeof(MainConfig));
    if (argc < 3 || argc > 7) {
        fprintf(stderr, CONSOLE_RED "Program expects 2-6 arguments, %d provided.\n" CONSOLE_RESET, argc - 1);
        print_help(argv[0]);
    }
    // Parse first arg as implementation
    {
        char lower_arg[7];  // We only care about first 6 characters
        // Convert to lower case
        int i = 0;
        for (; argv[1][i] && i < 6; i++) {
            lower_arg[i] = tolower(argv[1][i]);
        }
        lower_arg[i] = '\0';
        // Check for a match
        if (!strcmp(lower_arg, "cpu")) {
            config->implementation = CPU;
        }
        else if (!strcmp(lower_arg, "openmp")) {
            config->implementation = OPENMP;
        }
        else if (!strcmp(lower_arg, "cuda") || !strcmp(lower_arg, "gpu")) {
            config->implementation = CUDA;
        }
        else {
            fprintf(stderr, CONSOLE_RED "Unexpected string provided as first argument: '%s' .\n" CONSOLE_RESET, argv[1]);
            fprintf(stderr, CONSOLE_RED "First argument expects a single implementation as string: CPU, OPENMP, CUDA.\n" CONSOLE_RESET);
            print_help(argv[0]);
        }
    }
    // Parse second arg as algorithm
    {
        char lower_arg;  // We only care about the first character
        // Convert to lower case
        lower_arg = tolower(argv[2][0]);
        // Check for a match
        if (lower_arg == 'p') {
            config->algorithm = ProductOfDifferences;
        } else if (lower_arg == 'r') {
            config->algorithm = RemoveFactors;
        } else if (lower_arg == 'c') {
            config->algorithm = ChromaticAberration;
        } else {
            fprintf(stderr, CONSOLE_RED "Unexpected string provided as second argument: '%s' .\n" CONSOLE_RESET, argv[2]);
            fprintf(stderr, CONSOLE_RED "Second argument expects a single algorithm as string: SD, C, DS.\n" CONSOLE_RESET);
            print_help(argv[0]);
        }
    }
}
}

void runProductOfDifferences(int argc, char** argv, Implementation implementation);
void runRemoveFactors(int argc, char** argv, Implementation implementation);
void runChromaticAberration(int argc, char** argv, Implementation implementation);

int main(int argc, char **argv) {
#ifdef _MSC_VER
    {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD consoleMode;
        GetConsoleMode(hConsole, &consoleMode);
        consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;  // Enable support for ANSI colours (Windows 10+)
        SetConsoleMode(hConsole, consoleMode);
    }
#endif
    // Parse args
    MainConfig config;
    parse_args(argc, argv, &config);

    switch (config.algorithm) {
    case ProductOfDifferences:
        runProductOfDifferences(argc, argv, config.implementation);
        break;
    case RemoveFactors:
        runRemoveFactors(argc, argv, config.implementation);
        break;
    case ChromaticAberration:
        runChromaticAberration(argc, argv, config.implementation);
        break;
    }
    
    // Cleanup
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
void print_help(const char *program_name) {
    fprintf(stderr, "%s " CONSOLE_BLUE "<mode> <algorithm> <algorithm_args>" CONSOLE_RESET " (" CONSOLE_BLUE "--bench" CONSOLE_RESET ")\n", program_name);
    
    const char* line_fmt = "%-28s %s\n";
    const char* line_fmt_0 = "%-18s %s\n";  // ANSI codes affect format offsets
    const char *line_fmt_1 = "%-38s %s\n";  // ANSI codes affect format offsets
    fprintf(stderr, CONSOLE_YELLOW "Common Arguments" CONSOLE_RESET ":\n");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<mode>" CONSOLE_RESET, "The implementation to use: CPU, OPENMP, CUDA");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<algorithm>" CONSOLE_RESET, "The algorithm to run: PD, RF, CA,");
    fprintf(stderr, line_fmt_0, "", "(only the first character is checked)");
    fprintf(stderr, line_fmt_1, CONSOLE_BLUE "-b" CONSOLE_RESET ", " CONSOLE_BLUE "--bench" CONSOLE_RESET, "Enable benchmark mode");
    fprintf(stderr, CONSOLE_YELLOW "Product of Differences Arguments" CONSOLE_RESET ": " CONSOLE_BLUE "<input file>" CONSOLE_RESET " | " CONSOLE_BLUE "<random seed> <length>" CONSOLE_RESET "\n");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<input file>" CONSOLE_RESET, "Path to .csv of uchar to initialise input buffer");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<random seed>" CONSOLE_RESET, "Seed for random generation of input buffer");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<length>" CONSOLE_RESET, "Length of the input buffer to be generated");
    fprintf(stderr, CONSOLE_YELLOW "Remove Factors Arguments" CONSOLE_RESET ": " CONSOLE_BLUE "<input file> <divisor>" CONSOLE_RESET " (" CONSOLE_BLUE "<output file>" CONSOLE_RESET ") | " CONSOLE_BLUE "<random seed> <length> <divisor>" CONSOLE_RESET " (" CONSOLE_BLUE "<output file>" CONSOLE_RESET ")" CONSOLE_RESET "\n");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<input file>" CONSOLE_RESET, "Path to .csv of ints to initialise input buffer");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<output file>" CONSOLE_RESET, "Optional, path to .csv output file.");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<random seed>" CONSOLE_RESET, "Seed for random generation of input buffer");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<length>" CONSOLE_RESET, "Length of the input buffer to be generated");
    fprintf(stderr, CONSOLE_YELLOW "Chromatic Aberration Arguments" CONSOLE_RESET ": " CONSOLE_BLUE "<input image>" CONSOLE_RESET " (" CONSOLE_BLUE "<output image>" CONSOLE_RESET ")\n");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<input image>" CONSOLE_RESET, "Path to .png input image");
    fprintf(stderr, line_fmt_0, "", "Image must have 3 channels.");
    fprintf(stderr, line_fmt, CONSOLE_BLUE "<output image>" CONSOLE_RESET, "Optional, path to .png output image.");

    exit(EXIT_FAILURE);
}

const char* implementation_to_string(Implementation i) {
    switch (i)
    {
    case CPU:
        return "CPU";
    case OPENMP:
        return "OpenMP";
    case CUDA:
        return "CUDA";
    }
    return "?";
}
const char* algorithm_to_string(Algorithm a) {
    switch (a)
    {
    case ProductOfDifferences:
        return "ProductOfDifferences";
    case RemoveFactors:
        return "RemoveFactors";
    case ChromaticAberration:
        return "ChromaticAberration";
    }
    return "?";
}
