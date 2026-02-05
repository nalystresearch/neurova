/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Codec - OpenEXR wrapper
 * HDR image encoding/decoding
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef NEUROVA_HAVE_OPENEXR

#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>

using namespace Imf;
using namespace Imath;

namespace neurova {
namespace codecs {
namespace exr {

// Decode EXR to float array (RGBA)
bool decode_file(
    const char* filename,
    float** output,
    int* width,
    int* height,
    int* channels
) {
    try {
        RgbaInputFile file(filename);
        Box2i dw = file.dataWindow();
        
        int w = dw.max.x - dw.min.x + 1;
        int h = dw.max.y - dw.min.y + 1;
        
        *width = w;
        *height = h;
        *channels = 4;  // RGBA
        
        Array2D<Rgba> pixels;
        pixels.resizeErase(h, w);
        
        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * w, 1, w);
        file.readPixels(dw.min.y, dw.max.y);
        
        *output = (float*)malloc(w * h * 4 * sizeof(float));
        if (!*output) return false;
        
        float* dst = *output;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const Rgba& pixel = pixels[y][x];
                dst[(y * w + x) * 4 + 0] = pixel.r;
                dst[(y * w + x) * 4 + 1] = pixel.g;
                dst[(y * w + x) * 4 + 2] = pixel.b;
                dst[(y * w + x) * 4 + 3] = pixel.a;
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

// Encode float array to EXR
bool encode_file(
    const char* filename,
    const float* input,
    int width,
    int height,
    int channels,
    int compression  // 0=none, 1=RLE, 2=ZIPS, 3=ZIP, 4=PIZ
) {
    try {
        Compression comp = NO_COMPRESSION;
        switch (compression) {
            case 1: comp = RLE_COMPRESSION; break;
            case 2: comp = ZIPS_COMPRESSION; break;
            case 3: comp = ZIP_COMPRESSION; break;
            case 4: comp = PIZ_COMPRESSION; break;
        }
        
        Header header(width, height);
        header.compression() = comp;
        
        Array2D<Rgba> pixels;
        pixels.resizeErase(height, width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Rgba& pixel = pixels[y][x];
                const float* src = input + (y * width + x) * channels;
                
                pixel.r = src[0];
                pixel.g = (channels > 1) ? src[1] : src[0];
                pixel.b = (channels > 2) ? src[2] : src[0];
                pixel.a = (channels > 3) ? src[3] : 1.0f;
            }
        }
        
        RgbaOutputFile file(filename, header, WRITE_RGBA);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
        
        return true;
    } catch (...) {
        return false;
    }
}

// Get EXR info
bool get_info(
    const char* filename,
    int* width,
    int* height,
    int* channels,
    bool* is_tiled,
    std::vector<std::string>* channel_names
) {
    try {
        InputFile file(filename);
        Box2i dw = file.header().dataWindow();
        
        *width = dw.max.x - dw.min.x + 1;
        *height = dw.max.y - dw.min.y + 1;
        
        const ChannelList& channels_list = file.header().channels();
        *channels = 0;
        
        if (channel_names) {
            channel_names->clear();
        }
        
        for (ChannelList::ConstIterator it = channels_list.begin();
             it != channels_list.end(); ++it) {
            (*channels)++;
            if (channel_names) {
                channel_names->push_back(it.name());
            }
        }
        
        *is_tiled = file.header().hasTileDescription();
        
        return true;
    } catch (...) {
        return false;
    }
}

// Decode specific channels
bool decode_channels(
    const char* filename,
    const std::vector<std::string>& channel_names,
    float** output,
    int* width,
    int* height
) {
    try {
        InputFile file(filename);
        Box2i dw = file.header().dataWindow();
        
        int w = dw.max.x - dw.min.x + 1;
        int h = dw.max.y - dw.min.y + 1;
        int num_channels = channel_names.size();
        
        *width = w;
        *height = h;
        
        *output = (float*)malloc(w * h * num_channels * sizeof(float));
        if (!*output) return false;
        
        FrameBuffer frameBuffer;
        
        for (int c = 0; c < num_channels; c++) {
            frameBuffer.insert(
                channel_names[c],
                Slice(
                    FLOAT,
                    (char*)(*output + c) - dw.min.x * num_channels * sizeof(float)
                                         - dw.min.y * w * num_channels * sizeof(float),
                    num_channels * sizeof(float),
                    w * num_channels * sizeof(float)
                )
            );
        }
        
        file.setFrameBuffer(frameBuffer);
        file.readPixels(dw.min.y, dw.max.y);
        
        return true;
    } catch (...) {
        return false;
    }
}

// Convert float EXR data to 8-bit with tone mapping
void tonemap_reinhard(
    const float* hdr_input,
    unsigned char* ldr_output,
    int width,
    int height,
    int channels,
    float gamma,
    float exposure
) {
    for (int i = 0; i < width * height * channels; i++) {
        // Apply exposure
        float val = hdr_input[i] * exposure;
        
        // Reinhard tone mapping
        val = val / (1.0f + val);
        
        // Gamma correction
        val = powf(val, 1.0f / gamma);
        
        // Clamp and convert
        ldr_output[i] = (unsigned char)(fminf(fmaxf(val * 255.0f, 0.0f), 255.0f));
    }
}

} // namespace exr
} // namespace codecs
} // namespace neurova

#endif // NEUROVA_HAVE_OPENEXR
