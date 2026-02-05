/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Codec - libjpeg-turbo wrapper
 * High-performance JPEG encoding/decoding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NEUROVA_HAVE_JPEG

#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

namespace neurova {
namespace codecs {
namespace jpeg {

// Error handling structure
struct ErrorManager {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
    char message[JMSG_LENGTH_MAX];
};

static void error_exit(j_common_ptr cinfo) {
    ErrorManager* err = (ErrorManager*)cinfo->err;
    (*cinfo->err->format_message)(cinfo, err->message);
    longjmp(err->setjmp_buffer, 1);
}

static void output_message(j_common_ptr cinfo) {
    // Suppress output
}

// Decode JPEG from memory
bool decode(
    const unsigned char* data,
    size_t size,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    jpeg_decompress_struct cinfo;
    ErrorManager jerr;
    
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;
    jerr.pub.output_message = output_message;
    
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, size);
    
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    cinfo.out_color_space = JCS_RGB;
    
    jpeg_start_decompress(&cinfo);
    
    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *channels = cinfo.output_components;
    
    size_t row_stride = cinfo.output_width * cinfo.output_components;
    size_t buffer_size = row_stride * cinfo.output_height;
    
    *output = (unsigned char*)malloc(buffer_size);
    if (!*output) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    JSAMPARRAY row_pointer = (*cinfo.mem->alloc_sarray)(
        (j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    unsigned char* dst = *output;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
        memcpy(dst, row_pointer[0], row_stride);
        dst += row_stride;
    }
    
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    
    return true;
}

// Decode JPEG from file
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

// Encode to JPEG
bool encode(
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int quality,
    unsigned char** output,
    size_t* output_size
) {
    if (channels != 1 && channels != 3) return false;
    
    jpeg_compress_struct cinfo;
    ErrorManager jerr;
    
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;
    jerr.pub.output_message = output_message;
    
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        return false;
    }
    
    jpeg_create_compress(&cinfo);
    
    unsigned long out_size = 0;
    unsigned char* out_buffer = nullptr;
    jpeg_mem_dest(&cinfo, &out_buffer, &out_size);
    
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = channels;
    cinfo.in_color_space = (channels == 1) ? JCS_GRAYSCALE : JCS_RGB;
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    
    // Enable optimized Huffman tables
    cinfo.optimize_coding = TRUE;
    
    jpeg_start_compress(&cinfo, TRUE);
    
    size_t row_stride = width * channels;
    JSAMPROW row_pointer[1];
    
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPROW)&input[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    
    *output = out_buffer;
    *output_size = out_size;
    
    return true;
}

// Encode to file
bool encode_file(
    const char* filename,
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int quality
) {
    unsigned char* buffer = nullptr;
    size_t size = 0;
    
    if (!encode(input, width, height, channels, quality, &buffer, &size)) {
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

// Get JPEG info without decoding
bool get_info(
    const unsigned char* data,
    size_t size,
    int* width,
    int* height,
    int* channels
) {
    jpeg_decompress_struct cinfo;
    ErrorManager jerr;
    
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;
    
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, size);
    
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    *width = cinfo.image_width;
    *height = cinfo.image_height;
    *channels = cinfo.num_components;
    
    jpeg_destroy_decompress(&cinfo);
    return true;
}

} // namespace jpeg
} // namespace codecs
} // namespace neurova

#endif // NEUROVA_HAVE_JPEG
