#ifndef __main_h__
#define __main_h__

enum Implementation{CPU, OPENMP, CUDA};
typedef enum Implementation Implementation;
enum Algorithm{ProductOfDifferences, RemoveFactors, ChromaticAberration};
typedef enum Algorithm Algorithm;
/**
 * Structure containing the options provided by runtime arguments
 */
struct MainConfig {
    /**
     * Which implementation to use CPU, OpenMP, CUDA
     */
    Implementation implementation;
    /**
    * Which algorithm to run StandardDeviation, Convolution, DataStructure
    */
    Algorithm algorithm;
    /**
     * Treated as boolean, program will operate in benchmark mode
     * This repeats the algorithm multiple times and returns an average time
     * It may also warn about incorrect settings
     */
    unsigned char benchmark;
}; typedef struct MainConfig MainConfig;
/**
 * Print runtime args and exit
 * @param program_name argv[0] should always be passed to this parameter
 */
void print_help(const char *program_name);

const char* implementation_to_string(Implementation i);
const char* algorithm_to_string(Algorithm a);

#endif  // __main_h__
