/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Codec - libpng wrapper
 * PNG encoding/decoding with full alpha support
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NEUROVA_HAVE_PNG

#include <png.h>

namespace neurova {
namespace codecs {
namespace png {

// Memory read structure
struct MemoryReader {
    const unsigned char* data;
    size_t size;
    size_t offset;
};

static void memory_read_fn(png_structp png_ptr, png_bytep data, size_t length) {
    MemoryReader* reader = (MemoryReader*)png_get_io_ptr(png_ptr);
    if (reader->offset + length > reader->size) {
        png_error(png_ptr, "Read past end of data");
        return;
    }
    memcpy(data, reader->data + reader->offset, length);
    reader->offset += length;
}

// Memory write structure
struct MemoryWriter {
    unsigned char* data;
    size_t size;
    size_t capacity;
};

static void memory_write_fn(png_structp png_ptr, png_bytep data, size_t length) {
    MemoryWriter* writer = (MemoryWriter*)png_get_io_ptr(png_ptr);
    
    while (writer->size + length > writer->capacity) {
        writer->capacity *= 2;
        writer->data = (unsigned char*)realloc(writer->data, writer->capacity);
        if (!writer->data) {
            png_error(png_ptr, "Memory allocation failed");
            return;
        }
    }
    
    memcpy(writer->data + writer->size, data, length);
    writer->size += length;
}

static void memory_flush_fn(png_structp png_ptr) {
    // No-op for memory
}

// Decode PNG from memory
bool decode(
    const unsigned char* data,
    size_t size,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    // Check PNG signature
    if (size < 8 || png_sig_cmp(data, 0, 8)) {
        return false;
    }
    
    png_structp png_ptr = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return false;
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }
    
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        return false;
    }
    
    MemoryReader reader = {data, size, 8};
    png_set_read_fn(png_ptr, &reader, memory_read_fn);
    png_set_sig_bytes(png_ptr, 8);
    
    png_read_info(png_ptr, info_ptr);
    
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    
    // Transform to 8-bit RGB/RGBA
    if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }
    
    png_read_update_info(png_ptr, info_ptr);
    
    *channels = png_get_channels(png_ptr, info_ptr);
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    
    *output = (unsigned char*)malloc(row_bytes * *height);
    if (!*output) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        return false;
    }
    
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * *height);
    for (int y = 0; y < *height; y++) {
        row_pointers[y] = *output + y * row_bytes;
    }
    
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, nullptr);
    
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    
    return true;
}

// Decode PNG from file
bool decode_file(
    const char* filename,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    FILE* file = fopen(filename, "rb");
    if (!file) return false;
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    unsigned char* data = (unsigned char*)malloc(size);
    if (!data) {
        fclose(file);
        return false;
    }
    
    if (fread(data, 1, size, file) != (size_t)size) {
        free(data);
        fclose(file);
        return false;
    }
    fclose(file);
    
    bool result = decode(data, size, output, width, height, channels);
    free(data);
    
    return result;
}

// Encode to PNG
bool encode(
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int compression_level,
    unsigned char** output,
    size_t* output_size
) {
    if (channels < 1 || channels > 4) return false;
    
    png_structp png_ptr = png_create_write_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return false;
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        return false;
    }
    
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    MemoryWriter writer = {nullptr, 0, 65536};
    writer.data = (unsigned char*)malloc(writer.capacity);
    if (!writer.data) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    png_set_write_fn(png_ptr, &writer, memory_write_fn, memory_flush_fn);
    
    int color_type;
    switch (channels) {
        case 1: color_type = PNG_COLOR_TYPE_GRAY; break;
        case 2: color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
        case 3: color_type = PNG_COLOR_TYPE_RGB; break;
        case 4: color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
        default: 
            free(writer.data);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            return false;
    }
    
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    
    if (compression_level >= 0 && compression_level <= 9) {
        png_set_compression_level(png_ptr, compression_level);
    }
    
    png_write_info(png_ptr, info_ptr);
    
    size_t row_bytes = width * channels;
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)(input + y * row_bytes);
    }
    
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, nullptr);
    
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    
    *output = writer.data;
    *output_size = writer.size;
    
    return true;
}

// Encode to file
bool encode_file(
    const char* filename,
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int compression_level
) {
    unsigned char* buffer = nullptr;
    size_t size = 0;
    
    if (!encode(input, width, height, channels, compression_level, &buffer, &size)) {
        return false;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        free(buffer);
        return false;
    }
    
    bool success = fwrite(buffer, 1, size, file) == size;
    fclose(file);
    free(buffer);
    
    return success;
}

// Get PNG info without decoding
bool get_info(
    const unsigned char* data,
    size_t size,
    int* width,
    int* height,
    int* channels,
    int* bit_depth
) {
    if (size < 8 || png_sig_cmp(data, 0, 8)) {
        return false;
    }
    
    png_structp png_ptr = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) return false;
    
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }
    
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        return false;
    }
    
    MemoryReader reader = {data, size, 8};
    png_set_read_fn(png_ptr, &reader, memory_read_fn);
    png_set_sig_bytes(png_ptr, 8);
    
    png_read_info(png_ptr, info_ptr);
    
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);
    *bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return true;
}

} // namespace png
} // namespace codecs
} // namespace neurova

#endif // NEUROVA_HAVE_PNG
