/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Codec - libtiff wrapper
 * TIFF encoding/decoding with multi-page support
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NEUROVA_HAVE_TIFF

#include <tiffio.h>

namespace neurova {
namespace codecs {
namespace tiff {

// Decode TIFF from file
bool decode_file(
    const char* filename,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) return false;
    
    uint32_t w, h;
    uint16_t samples_per_pixel, bits_per_sample;
    
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    
    *width = w;
    *height = h;
    *channels = samples_per_pixel;
    
    // Use RGBA read for maximum compatibility
    size_t npixels = w * h;
    uint32_t* raster = (uint32_t*)_TIFFmalloc(npixels * sizeof(uint32_t));
    if (!raster) {
        TIFFClose(tif);
        return false;
    }
    
    if (!TIFFReadRGBAImageOriented(tif, w, h, raster, ORIENTATION_TOPLEFT, 0)) {
        _TIFFfree(raster);
        TIFFClose(tif);
        return false;
    }
    
    // Convert to RGB/RGBA
    *channels = 4;  // RGBA from TIFFReadRGBAImage
    *output = (unsigned char*)malloc(npixels * 4);
    if (!*output) {
        _TIFFfree(raster);
        TIFFClose(tif);
        return false;
    }
    
    // TIFFReadRGBAImage returns ABGR, convert to RGBA
    unsigned char* dst = *output;
    for (size_t i = 0; i < npixels; i++) {
        uint32_t pixel = raster[i];
        dst[i * 4 + 0] = TIFFGetR(pixel);
        dst[i * 4 + 1] = TIFFGetG(pixel);
        dst[i * 4 + 2] = TIFFGetB(pixel);
        dst[i * 4 + 3] = TIFFGetA(pixel);
    }
    
    _TIFFfree(raster);
    TIFFClose(tif);
    
    return true;
}

// Encode to TIFF file
bool encode_file(
    const char* filename,
    const unsigned char* input,
    int width,
    int height,
    int channels,
    int compression  // 0=none, 1=LZW, 2=ZIP, 3=JPEG
) {
    TIFF* tif = TIFFOpen(filename, "w");
    if (!tif) return false;
    
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    
    if (channels == 1) {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    } else if (channels == 3 || channels == 4) {
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    }
    
    // Set compression
    switch (compression) {
        case 1:
            TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
            break;
        case 2:
            TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
            break;
        case 3:
            TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
            TIFFSetField(tif, TIFFTAG_JPEGQUALITY, 90);
            break;
        default:
            TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            break;
    }
    
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width * channels));
    
    size_t row_bytes = width * channels;
    for (int row = 0; row < height; row++) {
        if (TIFFWriteScanline(tif, (void*)(input + row * row_bytes), row, 0) < 0) {
            TIFFClose(tif);
            return false;
        }
    }
    
    TIFFClose(tif);
    return true;
}

// Get TIFF info
bool get_info(
    const char* filename,
    int* width,
    int* height,
    int* channels,
    int* bit_depth,
    int* num_pages
) {
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) return false;
    
    uint32_t w, h;
    uint16_t samples, bits;
    
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    
    *width = w;
    *height = h;
    *channels = samples;
    *bit_depth = bits;
    
    // Count pages
    int pages = 1;
    while (TIFFReadDirectory(tif)) {
        pages++;
    }
    *num_pages = pages;
    
    TIFFClose(tif);
    return true;
}

// Decode specific page
bool decode_page(
    const char* filename,
    int page,
    unsigned char** output,
    int* width,
    int* height,
    int* channels
) {
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) return false;
    
    // Navigate to page
    for (int i = 0; i < page; i++) {
        if (!TIFFReadDirectory(tif)) {
            TIFFClose(tif);
            return false;
        }
    }
    
    uint32_t w, h;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    
    *width = w;
    *height = h;
    *channels = 4;
    
    size_t npixels = w * h;
    uint32_t* raster = (uint32_t*)_TIFFmalloc(npixels * sizeof(uint32_t));
    if (!raster) {
        TIFFClose(tif);
        return false;
    }
    
    if (!TIFFReadRGBAImageOriented(tif, w, h, raster, ORIENTATION_TOPLEFT, 0)) {
        _TIFFfree(raster);
        TIFFClose(tif);
        return false;
    }
    
    *output = (unsigned char*)malloc(npixels * 4);
    if (!*output) {
        _TIFFfree(raster);
        TIFFClose(tif);
        return false;
    }
    
    unsigned char* dst = *output;
    for (size_t i = 0; i < npixels; i++) {
        uint32_t pixel = raster[i];
        dst[i * 4 + 0] = TIFFGetR(pixel);
        dst[i * 4 + 1] = TIFFGetG(pixel);
        dst[i * 4 + 2] = TIFFGetB(pixel);
        dst[i * 4 + 3] = TIFFGetA(pixel);
    }
    
    _TIFFfree(raster);
    TIFFClose(tif);
    
    return true;
}

} // namespace tiff
} // namespace codecs
} // namespace neurova

#endif // NEUROVA_HAVE_TIFF
