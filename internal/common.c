#include "common.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

/**
 * This structure represents a variable channel image
 */
struct CImage {
    /**
     * Array of pixel data of the image, 1 unsigned char per pixel channel
     * Pixels ordered left to right, top to bottom
     * There is no stride, this is a compact storage
     */
    unsigned char* data;
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
typedef struct CImage CImage;
/**
 * This structure represents a multi-channel image
 * It is used internally to create single channel images to pass to the algorithm
 */
struct HSVImage {
    /**
     * Array of pixel data of the image, 1 unsigned char per pixel channel
     * Pixels ordered left to right, top to bottom
     * There is no stride, this is a compact storage
     */
    float* h; // Angle in degrees [0-360]
    float* s; //Fractional value [0-1]
    unsigned char* v; //Fractional value [0-255], divide by 255 for the real value
    // Optional alpha channel
    unsigned char* a; // Unchanged from input image
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
typedef struct HSVImage HSVImage;

void loadCSV(const char *input_file, void **buffer, size_t *buf_elements, const char* format) {
    // Open file
    FILE* file = fopen(input_file, "r");
    if (file == NULL) {
        fprintf(stderr, CONSOLE_RED "Unable to open file '%s' for reading.\n" CONSOLE_RESET, input_file);
        exit(EXIT_FAILURE);
    }
    // Read file to string
    fseek(file, 0L, SEEK_END);
    const size_t file_sz = ftell(file);
    fseek(file, 0L, SEEK_SET);
    char* file_buf = (char*)malloc(file_sz);
    const size_t read_sz = fread(file_buf, 1, file_sz, file);
    if (read_sz != file_sz && ferror(file)) {
        fprintf(stderr, CONSOLE_RED "An error occurred whilst reading '%s'.\n" CONSOLE_RESET, input_file);
        exit(EXIT_FAILURE);
    }
    fclose(file);
    // Skip utf8-BOM header if present (excel export junk)
    size_t sz = file_sz;
    if (file_sz > 3 && 
        file_buf[0] == (char)0xEF &&
        file_buf[1] == (char)0xBB &&
        file_buf[2] == (char)0xBF) {
        sz -= 3;
    }
    // strtok modifies buf, and we run it twice.
    char* buf1 = malloc(sz);
    char* buf2 = malloc(sz);
    memcpy(buf1, file_buf + (file_sz - sz), sz);
    memcpy(buf2, file_buf + (file_sz - sz), sz);
    // Find the delimiter
    char delimiter[2];
    for (size_t i = 0; i < sz; ++i) {
        if (!isdigit(buf1[i]) && buf1[i] != '.') {  // '.' required for float, can't be delimiter
            delimiter[0] = buf1[i];
            break;
        }
    }
    delimiter[1] = '\0';
    // Count items
    char* token = strtok(buf1, delimiter);
    size_t elements = 0;
    while (token != NULL) {
        float value;
        if (sscanf(token, format, &value) == 1) {
            ++elements;
        }
        token = strtok(NULL, delimiter);
    }
    *buf_elements = elements;
    float *t_buffer = malloc(elements * sizeof(float));  // sizeof(float) == sizeof(unsigned int)
    // Read items (sd:float, ds:uint)
    token = strtok(buf2, delimiter);
    elements = 0;
    while (token != NULL) {
        sscanf(token, format, &t_buffer[elements++]);  // should return 1
        token = strtok(NULL, delimiter);
    }
    // Cleanup
    free(buf2);
    free(buf1);
    free(file_buf);
    *buffer = t_buffer;
}
void saveCSV(const char* output_file, int* buf, size_t buf_elements) {
    FILE *outf = fopen(output_file, "w");
    if (outf == NULL) {
        fprintf(stderr, CONSOLE_RED "Unable to open file '%s' for writing.\n" CONSOLE_RESET, output_file);
        exit(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < buf_elements; ++i) {
        fprintf(outf, "%d\n", buf[i]);
    }
    fclose(outf);
}

void loadImage(const char* input_file, Image* out_image) {
    // Load Image
    CImage user_cimage;
    {
        user_cimage.data = stbi_load(input_file, &user_cimage.width, &user_cimage.height, &user_cimage.channels, 0);
        if (!user_cimage.data) {
            fprintf(stderr, CONSOLE_RED "Unable to load image '%s', please try a different file.\n" CONSOLE_RESET, input_file);
            exit(EXIT_FAILURE);
        }
        if (user_cimage.channels != 3) {
            fprintf(stderr, CONSOLE_RED "Only 3 channel images are supported, please try a different file.\n" CONSOLE_RESET);
            exit(EXIT_FAILURE);
        }
        // Transfer image to our output buffers
        out_image->width = user_cimage.width;
        out_image->height = user_cimage.height;
        out_image->data = (unsigned char*)malloc(out_image->width * out_image->height * user_cimage.channels * sizeof(unsigned char));
        memcpy(out_image->data, user_cimage.data, out_image->width * out_image->height * user_cimage.channels * sizeof(unsigned char));
        // Cleanup
        stbi_image_free(user_cimage.data);
    }
}
void saveImage(const char *output_file, Image out_image) {
    if (!stbi_write_png(output_file, out_image.width, out_image.height, CHANNELS, out_image.data, out_image.width * CHANNELS)) {
        printf(CONSOLE_RED "Unable to save image output to %s.\n" CONSOLE_RESET, output_file);
    }
}
