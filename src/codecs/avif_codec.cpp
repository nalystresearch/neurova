/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/**
 * Neurova Codec - libavif wrapper
 * AVIF image encoding/decoding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NEUROVA_HAVE_AVIF

#include <avif/avif.h>

namespace neurova {
namespace codecs {
namespace avif {

// Decode AVIF from memory
bool decode(
    const unsigned char* data,
    size_t size,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    avifDecoder* decoder = avifDecoderCreate();
    if (!decoder) return false;
    
    avifResult result = avifDecoderSetIOMemory(decoder, data, size);
    if (result != AVIF_RESULT_OK) {
        avifDecoderDestroy(decoder);
        return false;
    }
    
    result = avifDecoderParse(decoder);
    if (result != AVIF_RESULT_OK) {
        avifDecoderDestroy(decoder);
        return false;
    }
    
    result = avifDecoderNextImage(decoder);
    if (result != AVIF_RESULT_OK) {
        avifDecoderDestroy(decoder);
        return false;
    }
    
    avifImage* image = decoder->image;
    
    *width = image->width;
    *height = image->height;
    *channels = image->alphaPlane ? 4 : 3;
    
    // Convert to RGB(A)
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, image);
    rgb.depth = 8;
    rgb.format = image->alphaPlane ? AVIF_RGB_FORMAT_RGBA : AVIF_RGB_FORMAT_RGB;
    
    avifRGBImageAllocatePixels(&rgb);
    
    result = avifImageYUVToRGB(image, &rgb);
    if (result != AVIF_RESULT_OK) {
        avifRGBImageFreePixels(&rgb);
        avifDecoderDestroy(decoder);
        return false;
    }
    
    size_t output_size = *width * *height * *channels;
    *output = (unsigned char*)malloc(output_size);
    if (!*output) {
        avifRGBImageFreePixels(&rgb);
        avifDecoderDestroy(decoder);
        return false;
    }
    
    memcpy(*output, rgb.pixels, output_size);
    
    avifRGBImageFreePixels(&rgb);
    avifDecoderDestroy(decoder);
    
    return true;
}

// Decode from file
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

// Encode to AVIF
bool encode(
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int quality,  // 0-100
    int speed,    // 0-10 (0=slowest/best, 10=fastest)
    unsigned char** output,
    size_t* output_size
) {
    avifImage* image = avifImageCreate(width, height, 8, AVIF_PIXEL_FORMAT_YUV444);
    if (!image) return false;
    
    avifRGBImage rgb;
    avifRGBImageSetDefaults(&rgb, image);
    rgb.depth = 8;
    rgb.format = (channels == 4) ? AVIF_RGB_FORMAT_RGBA : AVIF_RGB_FORMAT_RGB;
    rgb.pixels = (uint8_t*)input;
    rgb.rowBytes = width * channels;
    
    avifResult result = avifImageRGBToYUV(image, &rgb);
    if (result != AVIF_RESULT_OK) {
        avifImageDestroy(image);
        return false;
    }
    
    avifEncoder* encoder = avifEncoderCreate();
    if (!encoder) {
        avifImageDestroy(image);
        return false;
    }
    
    // Quality: 0=lossless, 63=worst
    encoder->quality = 63 - (quality * 63 / 100);
    encoder->qualityAlpha = encoder->quality;
    encoder->speed = speed;
    
    avifRWData output_data = AVIF_DATA_EMPTY;
    result = avifEncoderWrite(encoder, image, &output_data);
    
    if (result != AVIF_RESULT_OK) {
        avifEncoderDestroy(encoder);
        avifImageDestroy(image);
        return false;
    }
    
    *output = (unsigned char*)malloc(output_data.size);
    if (!*output) {
        avifRWDataFree(&output_data);
        avifEncoderDestroy(encoder);
        avifImageDestroy(image);
        return false;
    }
    
    memcpy(*output, output_data.data, output_data.size);
    *output_size = output_data.size;
    
    avifRWDataFree(&output_data);
    avifEncoderDestroy(encoder);
    avifImageDestroy(image);
    
    return true;
}

// Encode to file
bool encode_file(
    const char* filename,
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int quality,
    int speed
) {
    unsigned char* buffer = nullptr;
    size_t size = 0;
    
    if (!encode(input, width, height, channels, quality, speed, &buffer, &size)) {
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

// Get AVIF info
bool get_info(
    const unsigned char* data,
    size_t size,
    int* width,
    int* height,
    int* channels,
    int* bit_depth,
    bool* has_alpha
) {
    avifDecoder* decoder = avifDecoderCreate();
    if (!decoder) return false;
    
    avifResult result = avifDecoderSetIOMemory(decoder, data, size);
    if (result != AVIF_RESULT_OK) {
        avifDecoderDestroy(decoder);
        return false;
    }
    
    result = avifDecoderParse(decoder);
    if (result != AVIF_RESULT_OK) {
        avifDecoderDestroy(decoder);
        return false;
    }
    
    *width = decoder->image->width;
    *height = decoder->image->height;
    *bit_depth = decoder->image->depth;
    *has_alpha = (decoder->image->alphaPlane != nullptr);
    *channels = *has_alpha ? 4 : 3;
    
    avifDecoderDestroy(decoder);
    return true;
}

} // namespace avif
} // namespace codecs
} // namespace neurova

#endif // NEUROVA_HAVE_AVIF
