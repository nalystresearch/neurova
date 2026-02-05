// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file io.hpp
 * @brief Image I/O operations
 * 
 * Provides image reading and writing functionality
 */

#ifndef NEUROVA_IO_HPP
#define NEUROVA_IO_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace neurova {
namespace io {

// ============================================================================
// Image Format Constants
// ============================================================================

enum class ImageFormat {
    UNKNOWN,
    BMP,
    PPM,
    PGM,
    PNG,
    JPEG,
    TIFF
};

// Read flags
constexpr int IMREAD_UNCHANGED = -1;
constexpr int IMREAD_GRAYSCALE = 0;
constexpr int IMREAD_COLOR = 1;
constexpr int IMREAD_ANYDEPTH = 2;
constexpr int IMREAD_ANYCOLOR = 4;

// Write flags (JPEG quality, PNG compression)
constexpr int IMWRITE_JPEG_QUALITY = 1;
constexpr int IMWRITE_PNG_COMPRESSION = 16;

// ============================================================================
// Image Structure
// ============================================================================

/**
 * @brief Simple image container
 */
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;
    
    Image() = default;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(static_cast<size_t>(w) * h * c);
    }
    
    bool empty() const { return data.empty(); }
    
    size_t step() const { return static_cast<size_t>(width) * channels; }
    
    uint8_t* ptr(int row) { return data.data() + row * step(); }
    const uint8_t* ptr(int row) const { return data.data() + row * step(); }
    
    uint8_t& at(int row, int col, int ch = 0) {
        return data[(row * width + col) * channels + ch];
    }
    
    const uint8_t& at(int row, int col, int ch = 0) const {
        return data[(row * width + col) * channels + ch];
    }
};

// ============================================================================
// Format Detection
// ============================================================================

/**
 * @brief Detect image format from file signature
 */
inline ImageFormat detectFormat(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return ImageFormat::UNKNOWN;
    
    uint8_t header[8] = {0};
    file.read(reinterpret_cast<char*>(header), 8);
    
    // BMP: "BM"
    if (header[0] == 'B' && header[1] == 'M') {
        return ImageFormat::BMP;
    }
    
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (header[0] == 0x89 && header[1] == 'P' && header[2] == 'N' && header[3] == 'G') {
        return ImageFormat::PNG;
    }
    
    // JPEG: FF D8 FF
    if (header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF) {
        return ImageFormat::JPEG;
    }
    
    // PPM/PGM: P5, P6
    if (header[0] == 'P' && (header[1] == '5' || header[1] == '6')) {
        return (header[1] == '5') ? ImageFormat::PGM : ImageFormat::PPM;
    }
    
    // TIFF: II 42 or MM 42
    if ((header[0] == 'I' && header[1] == 'I' && header[2] == 42 && header[3] == 0) ||
        (header[0] == 'M' && header[1] == 'M' && header[2] == 0 && header[3] == 42)) {
        return ImageFormat::TIFF;
    }
    
    return ImageFormat::UNKNOWN;
}

/**
 * @brief Detect format from filename extension
 */
inline ImageFormat detectFormatFromExtension(const std::string& filename) {
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos) return ImageFormat::UNKNOWN;
    
    std::string ext = filename.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "bmp") return ImageFormat::BMP;
    if (ext == "ppm") return ImageFormat::PPM;
    if (ext == "pgm") return ImageFormat::PGM;
    if (ext == "png") return ImageFormat::PNG;
    if (ext == "jpg" || ext == "jpeg") return ImageFormat::JPEG;
    if (ext == "tif" || ext == "tiff") return ImageFormat::TIFF;
    
    return ImageFormat::UNKNOWN;
}

// ============================================================================
// BMP Reader/Writer
// ============================================================================

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
};

struct BMPInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression;
    uint32_t sizeImage;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

/**
 * @brief Read BMP image
 */
inline Image readBMP(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return Image();
    
    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    
    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    
    if (fileHeader.type != 0x4D42) return Image(); // "BM"
    
    int width = infoHeader.width;
    int height = std::abs(infoHeader.height);
    bool topDown = (infoHeader.height < 0);
    int bpp = infoHeader.bitCount;
    
    int channels = (bpp == 24 || bpp == 32) ? 3 : 1;
    Image img(width, height, channels);
    
    file.seekg(fileHeader.offset);
    
    int rowSize = ((bpp * width + 31) / 32) * 4;
    std::vector<uint8_t> rowBuffer(rowSize);
    
    for (int y = 0; y < height; ++y) {
        int destRow = topDown ? y : (height - 1 - y);
        file.read(reinterpret_cast<char*>(rowBuffer.data()), rowSize);
        
        if (bpp == 24) {
            for (int x = 0; x < width; ++x) {
                img.at(destRow, x, 0) = rowBuffer[x * 3 + 2]; // R
                img.at(destRow, x, 1) = rowBuffer[x * 3 + 1]; // G
                img.at(destRow, x, 2) = rowBuffer[x * 3 + 0]; // B
            }
        } else if (bpp == 32) {
            for (int x = 0; x < width; ++x) {
                img.at(destRow, x, 0) = rowBuffer[x * 4 + 2]; // R
                img.at(destRow, x, 1) = rowBuffer[x * 4 + 1]; // G
                img.at(destRow, x, 2) = rowBuffer[x * 4 + 0]; // B
            }
        } else if (bpp == 8) {
            for (int x = 0; x < width; ++x) {
                img.at(destRow, x, 0) = rowBuffer[x];
            }
        }
    }
    
    return img;
}

/**
 * @brief Write BMP image
 */
inline bool writeBMP(const std::string& filename, const Image& img) {
    if (img.empty()) return false;
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    int width = img.width;
    int height = img.height;
    int channels = img.channels;
    int bpp = (channels == 1) ? 8 : 24;
    int rowSize = ((bpp * width + 31) / 32) * 4;
    int imageSize = rowSize * height;
    int paletteSize = (channels == 1) ? 256 * 4 : 0;
    
    BMPFileHeader fileHeader;
    fileHeader.type = 0x4D42; // "BM"
    fileHeader.size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize + imageSize;
    fileHeader.reserved1 = 0;
    fileHeader.reserved2 = 0;
    fileHeader.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + paletteSize;
    
    BMPInfoHeader infoHeader;
    infoHeader.size = sizeof(BMPInfoHeader);
    infoHeader.width = width;
    infoHeader.height = height;
    infoHeader.planes = 1;
    infoHeader.bitCount = bpp;
    infoHeader.compression = 0;
    infoHeader.sizeImage = imageSize;
    infoHeader.xPixelsPerMeter = 2835;
    infoHeader.yPixelsPerMeter = 2835;
    infoHeader.colorsUsed = 0;
    infoHeader.colorsImportant = 0;
    
    file.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    
    // Write palette for grayscale
    if (channels == 1) {
        for (int i = 0; i < 256; ++i) {
            uint8_t palette[4] = {static_cast<uint8_t>(i), static_cast<uint8_t>(i), 
                                  static_cast<uint8_t>(i), 0};
            file.write(reinterpret_cast<char*>(palette), 4);
        }
    }
    
    // Write pixel data (bottom-up)
    std::vector<uint8_t> rowBuffer(rowSize, 0);
    
    for (int y = height - 1; y >= 0; --y) {
        std::fill(rowBuffer.begin(), rowBuffer.end(), 0);
        
        if (channels == 1) {
            for (int x = 0; x < width; ++x) {
                rowBuffer[x] = img.at(y, x, 0);
            }
        } else {
            for (int x = 0; x < width; ++x) {
                rowBuffer[x * 3 + 0] = (channels >= 3) ? img.at(y, x, 2) : img.at(y, x, 0); // B
                rowBuffer[x * 3 + 1] = (channels >= 3) ? img.at(y, x, 1) : img.at(y, x, 0); // G
                rowBuffer[x * 3 + 2] = img.at(y, x, 0); // R
            }
        }
        
        file.write(reinterpret_cast<char*>(rowBuffer.data()), rowSize);
    }
    
    return true;
}

// ============================================================================
// PPM/PGM Reader/Writer
// ============================================================================

/**
 * @brief Read PPM/PGM image
 */
inline Image readPPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return Image();
    
    std::string magic;
    file >> magic;
    
    if (magic != "P5" && magic != "P6") return Image();
    
    int channels = (magic == "P5") ? 1 : 3;
    
    // Skip comments
    char c;
    file >> std::ws;
    while (file.peek() == '#') {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    
    int width, height, maxVal;
    file >> width >> height >> maxVal;
    file.get(); // Skip single whitespace
    
    Image img(width, height, channels);
    
    if (maxVal <= 255) {
        file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    } else {
        // 16-bit, convert to 8-bit
        for (size_t i = 0; i < img.data.size(); ++i) {
            uint8_t hi, lo;
            file.read(reinterpret_cast<char*>(&hi), 1);
            file.read(reinterpret_cast<char*>(&lo), 1);
            img.data[i] = static_cast<uint8_t>((hi << 8 | lo) * 255 / maxVal);
        }
    }
    
    return img;
}

/**
 * @brief Write PPM/PGM image
 */
inline bool writePPM(const std::string& filename, const Image& img) {
    if (img.empty()) return false;
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    std::string magic = (img.channels == 1) ? "P5" : "P6";
    file << magic << "\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";
    
    if (img.channels == 1 || img.channels == 3) {
        file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    } else {
        // Convert to RGB
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                uint8_t r = img.at(y, x, 0);
                uint8_t g = (img.channels > 1) ? img.at(y, x, 1) : r;
                uint8_t b = (img.channels > 2) ? img.at(y, x, 2) : g;
                file.write(reinterpret_cast<const char*>(&r), 1);
                file.write(reinterpret_cast<const char*>(&g), 1);
                file.write(reinterpret_cast<const char*>(&b), 1);
            }
        }
    }
    
    return true;
}

// ============================================================================
// Main Read/Write Functions
// ============================================================================

/**
 * @brief Read image from file
 * @param filename Path to image file
 * @param flags Read flags (IMREAD_COLOR, IMREAD_GRAYSCALE, etc.)
 * @return Loaded image or empty image on failure
 */
inline Image imread(const std::string& filename, int flags = IMREAD_COLOR) {
    ImageFormat format = detectFormat(filename);
    
    Image img;
    
    switch (format) {
        case ImageFormat::BMP:
            img = readBMP(filename);
            break;
        case ImageFormat::PPM:
        case ImageFormat::PGM:
            img = readPPM(filename);
            break;
        case ImageFormat::PNG:
        case ImageFormat::JPEG:
        case ImageFormat::TIFF:
            // These formats require external libraries
            // Return empty for now - can be extended
            return Image();
        default:
            return Image();
    }
    
    // Convert based on flags
    if (!img.empty() && flags == IMREAD_GRAYSCALE && img.channels > 1) {
        Image gray(img.width, img.height, 1);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                float val = 0.299f * img.at(y, x, 0) + 
                           0.587f * img.at(y, x, 1) + 
                           0.114f * img.at(y, x, 2);
                gray.at(y, x, 0) = static_cast<uint8_t>(std::min(255.0f, val));
            }
        }
        return gray;
    }
    
    return img;
}

/**
 * @brief Write image to file
 * @param filename Output file path
 * @param img Image to save
 * @param params Optional encoding parameters
 * @return true on success
 */
inline bool imwrite(const std::string& filename, const Image& img,
                   const std::vector<int>& params = {}) {
    (void)params; // Currently unused, for API compatibility
    
    ImageFormat format = detectFormatFromExtension(filename);
    
    switch (format) {
        case ImageFormat::BMP:
            return writeBMP(filename, img);
        case ImageFormat::PPM:
        case ImageFormat::PGM:
            return writePPM(filename, img);
        case ImageFormat::PNG:
        case ImageFormat::JPEG:
        case ImageFormat::TIFF:
            // These require external libraries
            // Fall back to BMP
            return writeBMP(filename + ".bmp", img);
        default:
            return false;
    }
}

// ============================================================================
// Image Creation Utilities
// ============================================================================

/**
 * @brief Create a solid color image
 */
inline Image createSolidImage(int width, int height, int channels, uint8_t value = 0) {
    Image img(width, height, channels);
    std::fill(img.data.begin(), img.data.end(), value);
    return img;
}

/**
 * @brief Create image from raw buffer
 */
inline Image createFromBuffer(const uint8_t* data, int width, int height, int channels) {
    Image img(width, height, channels);
    std::memcpy(img.data.data(), data, img.data.size());
    return img;
}

/**
 * @brief Create grayscale gradient image
 */
inline Image createGradient(int width, int height, bool horizontal = true) {
    Image img(width, height, 1);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (horizontal) {
                img.at(y, x, 0) = static_cast<uint8_t>(255 * x / (width - 1));
            } else {
                img.at(y, x, 0) = static_cast<uint8_t>(255 * y / (height - 1));
            }
        }
    }
    
    return img;
}

/**
 * @brief Create checkerboard pattern
 */
inline Image createCheckerboard(int width, int height, int cellSize = 32) {
    Image img(width, height, 1);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool white = ((x / cellSize) + (y / cellSize)) % 2 == 0;
            img.at(y, x, 0) = white ? 255 : 0;
        }
    }
    
    return img;
}

// ============================================================================
// Image Conversion
// ============================================================================

/**
 * @brief Convert image to grayscale
 */
inline Image toGrayscale(const Image& img) {
    if (img.empty() || img.channels == 1) return img;
    
    Image gray(img.width, img.height, 1);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float val = 0.299f * img.at(y, x, 0) + 
                       0.587f * img.at(y, x, 1) + 
                       0.114f * img.at(y, x, 2);
            gray.at(y, x, 0) = static_cast<uint8_t>(std::min(255.0f, val));
        }
    }
    
    return gray;
}

/**
 * @brief Convert grayscale to RGB
 */
inline Image toRGB(const Image& img) {
    if (img.empty()) return img;
    if (img.channels == 3) return img;
    
    Image rgb(img.width, img.height, 3);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            uint8_t val = img.at(y, x, 0);
            rgb.at(y, x, 0) = val;
            rgb.at(y, x, 1) = val;
            rgb.at(y, x, 2) = val;
        }
    }
    
    return rgb;
}

/**
 * @brief Extract a single channel
 */
inline Image extractChannel(const Image& img, int channel) {
    if (img.empty() || channel >= img.channels) return Image();
    
    Image out(img.width, img.height, 1);
    
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            out.at(y, x, 0) = img.at(y, x, channel);
        }
    }
    
    return out;
}

/**
 * @brief Merge channels
 */
inline Image mergeChannels(const std::vector<Image>& channels) {
    if (channels.empty()) return Image();
    
    int width = channels[0].width;
    int height = channels[0].height;
    int nChannels = static_cast<int>(channels.size());
    
    Image out(width, height, nChannels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < nChannels; ++c) {
                out.at(y, x, c) = channels[c].at(y, x, 0);
            }
        }
    }
    
    return out;
}

// ============================================================================
// Image Information
// ============================================================================

/**
 * @brief Get image statistics
 */
struct ImageStats {
    float min;
    float max;
    float mean;
    float stddev;
};

inline ImageStats getImageStats(const Image& img) {
    ImageStats stats = {255.0f, 0.0f, 0.0f, 0.0f};
    
    if (img.empty()) return stats;
    
    double sum = 0.0;
    double sumSq = 0.0;
    size_t n = img.data.size();
    
    for (size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(img.data[i]);
        stats.min = std::min(stats.min, val);
        stats.max = std::max(stats.max, val);
        sum += val;
        sumSq += val * val;
    }
    
    stats.mean = static_cast<float>(sum / n);
    stats.stddev = static_cast<float>(std::sqrt(sumSq / n - stats.mean * stats.mean));
    
    return stats;
}

/**
 * @brief Print image information
 */
inline std::string getImageInfo(const Image& img) {
    if (img.empty()) return "Empty image";
    
    ImageStats stats = getImageStats(img);
    
    std::string info = "Size: " + std::to_string(img.width) + "x" + 
                       std::to_string(img.height) + "x" + 
                       std::to_string(img.channels) + "\n";
    info += "Min: " + std::to_string(stats.min) + "\n";
    info += "Max: " + std::to_string(stats.max) + "\n";
    info += "Mean: " + std::to_string(stats.mean) + "\n";
    info += "Stddev: " + std::to_string(stats.stddev) + "\n";
    
    return info;
}

} // namespace io
} // namespace neurova

#endif // NEUROVA_IO_HPP
