// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file color.hpp
 * @brief Color space conversion functions
 */

#ifndef NEUROVA_IMGPROC_COLOR_HPP
#define NEUROVA_IMGPROC_COLOR_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace neurova {
namespace imgproc {

// Color conversion codes
constexpr int COLOR_BGR2GRAY = 0;
constexpr int COLOR_RGB2GRAY = 1;
constexpr int COLOR_GRAY2BGR = 2;
constexpr int COLOR_GRAY2RGB = 3;
constexpr int COLOR_BGR2RGB = 4;
constexpr int COLOR_RGB2BGR = 5;
constexpr int COLOR_BGR2HSV = 6;
constexpr int COLOR_RGB2HSV = 7;
constexpr int COLOR_HSV2BGR = 8;
constexpr int COLOR_HSV2RGB = 9;
constexpr int COLOR_BGR2HLS = 10;
constexpr int COLOR_RGB2HLS = 11;
constexpr int COLOR_HLS2BGR = 12;
constexpr int COLOR_HLS2RGB = 13;
constexpr int COLOR_BGR2Lab = 14;
constexpr int COLOR_RGB2Lab = 15;
constexpr int COLOR_Lab2BGR = 16;
constexpr int COLOR_Lab2RGB = 17;
constexpr int COLOR_BGR2Luv = 18;
constexpr int COLOR_RGB2Luv = 19;
constexpr int COLOR_Luv2BGR = 20;
constexpr int COLOR_Luv2RGB = 21;
constexpr int COLOR_BGR2YCrCb = 22;
constexpr int COLOR_RGB2YCrCb = 23;
constexpr int COLOR_YCrCb2BGR = 24;
constexpr int COLOR_YCrCb2RGB = 25;
constexpr int COLOR_BGR2XYZ = 26;
constexpr int COLOR_RGB2XYZ = 27;
constexpr int COLOR_XYZ2BGR = 28;
constexpr int COLOR_XYZ2RGB = 29;
constexpr int COLOR_BGRA2GRAY = 30;
constexpr int COLOR_RGBA2GRAY = 31;
constexpr int COLOR_GRAY2BGRA = 32;
constexpr int COLOR_GRAY2RGBA = 33;
constexpr int COLOR_BGRA2BGR = 34;
constexpr int COLOR_RGBA2RGB = 35;
constexpr int COLOR_BGR2BGRA = 36;
constexpr int COLOR_RGB2RGBA = 37;

/**
 * @brief Convert RGB/BGR to grayscale
 */
inline std::vector<float> toGray(
    const float* input, int width, int height,
    bool isBGR = true
) {
    std::vector<float> output(width * height);
    
    // ITU-R BT.601 standard weights
    float rWeight = 0.299f;
    float gWeight = 0.587f;
    float bWeight = 0.114f;
    
    for (int i = 0; i < width * height; ++i) {
        float r, g, b;
        if (isBGR) {
            b = input[i * 3];
            g = input[i * 3 + 1];
            r = input[i * 3 + 2];
        } else {
            r = input[i * 3];
            g = input[i * 3 + 1];
            b = input[i * 3 + 2];
        }
        output[i] = r * rWeight + g * gWeight + b * bWeight;
    }
    
    return output;
}

/**
 * @brief Convert grayscale to RGB/BGR
 */
inline std::vector<float> grayToColor(
    const float* input, int width, int height
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float gray = input[i];
        output[i * 3] = gray;
        output[i * 3 + 1] = gray;
        output[i * 3 + 2] = gray;
    }
    
    return output;
}

/**
 * @brief Swap RGB and BGR channels
 */
inline std::vector<float> swapChannels(
    const float* input, int width, int height
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        output[i * 3] = input[i * 3 + 2];
        output[i * 3 + 1] = input[i * 3 + 1];
        output[i * 3 + 2] = input[i * 3];
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to HSV
 */
inline std::vector<float> toHSV(
    const float* input, int width, int height,
    bool isBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float r, g, b;
        if (isBGR) {
            b = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            r = input[i * 3 + 2] / 255.0f;
        } else {
            r = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            b = input[i * 3 + 2] / 255.0f;
        }
        
        float maxVal = std::max({r, g, b});
        float minVal = std::min({r, g, b});
        float delta = maxVal - minVal;
        
        float h, s, v;
        v = maxVal;
        
        if (delta < 1e-6f) {
            h = 0.0f;
            s = 0.0f;
        } else {
            s = delta / maxVal;
            
            if (maxVal == r) {
                h = 60.0f * std::fmod((g - b) / delta, 6.0f);
            } else if (maxVal == g) {
                h = 60.0f * ((b - r) / delta + 2.0f);
            } else {
                h = 60.0f * ((r - g) / delta + 4.0f);
            }
            
            if (h < 0.0f) h += 360.0f;
        }
        
        // standard convention: H [0, 180], S [0, 255], V [0, 255]
        output[i * 3] = h / 2.0f;
        output[i * 3 + 1] = s * 255.0f;
        output[i * 3 + 2] = v * 255.0f;
    }
    
    return output;
}

/**
 * @brief Convert HSV to RGB/BGR
 */
inline std::vector<float> hsvToColor(
    const float* input, int width, int height,
    bool toBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float h = input[i * 3] * 2.0f;  // [0, 360]
        float s = input[i * 3 + 1] / 255.0f;
        float v = input[i * 3 + 2] / 255.0f;
        
        float c = v * s;
        float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = v - c;
        
        float r, g, b;
        if (h < 60.0f) {
            r = c; g = x; b = 0.0f;
        } else if (h < 120.0f) {
            r = x; g = c; b = 0.0f;
        } else if (h < 180.0f) {
            r = 0.0f; g = c; b = x;
        } else if (h < 240.0f) {
            r = 0.0f; g = x; b = c;
        } else if (h < 300.0f) {
            r = x; g = 0.0f; b = c;
        } else {
            r = c; g = 0.0f; b = x;
        }
        
        r = (r + m) * 255.0f;
        g = (g + m) * 255.0f;
        b = (b + m) * 255.0f;
        
        if (toBGR) {
            output[i * 3] = b;
            output[i * 3 + 1] = g;
            output[i * 3 + 2] = r;
        } else {
            output[i * 3] = r;
            output[i * 3 + 1] = g;
            output[i * 3 + 2] = b;
        }
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to HLS
 */
inline std::vector<float> toHLS(
    const float* input, int width, int height,
    bool isBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float r, g, b;
        if (isBGR) {
            b = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            r = input[i * 3 + 2] / 255.0f;
        } else {
            r = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            b = input[i * 3 + 2] / 255.0f;
        }
        
        float maxVal = std::max({r, g, b});
        float minVal = std::min({r, g, b});
        float delta = maxVal - minVal;
        
        float h, l, s;
        l = (maxVal + minVal) / 2.0f;
        
        if (delta < 1e-6f) {
            h = 0.0f;
            s = 0.0f;
        } else {
            s = (l < 0.5f) ? (delta / (maxVal + minVal)) : (delta / (2.0f - maxVal - minVal));
            
            if (maxVal == r) {
                h = 60.0f * std::fmod((g - b) / delta, 6.0f);
            } else if (maxVal == g) {
                h = 60.0f * ((b - r) / delta + 2.0f);
            } else {
                h = 60.0f * ((r - g) / delta + 4.0f);
            }
            
            if (h < 0.0f) h += 360.0f;
        }
        
        output[i * 3] = h / 2.0f;
        output[i * 3 + 1] = l * 255.0f;
        output[i * 3 + 2] = s * 255.0f;
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to YCrCb
 */
inline std::vector<float> toYCrCb(
    const float* input, int width, int height,
    bool isBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float r, g, b;
        if (isBGR) {
            b = input[i * 3];
            g = input[i * 3 + 1];
            r = input[i * 3 + 2];
        } else {
            r = input[i * 3];
            g = input[i * 3 + 1];
            b = input[i * 3 + 2];
        }
        
        float Y = 0.299f * r + 0.587f * g + 0.114f * b;
        float Cr = (r - Y) * 0.713f + 128.0f;
        float Cb = (b - Y) * 0.564f + 128.0f;
        
        output[i * 3] = Y;
        output[i * 3 + 1] = Cr;
        output[i * 3 + 2] = Cb;
    }
    
    return output;
}

/**
 * @brief Convert YCrCb to RGB/BGR
 */
inline std::vector<float> ycrcbToColor(
    const float* input, int width, int height,
    bool toBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float Y = input[i * 3];
        float Cr = input[i * 3 + 1] - 128.0f;
        float Cb = input[i * 3 + 2] - 128.0f;
        
        float r = Y + 1.403f * Cr;
        float g = Y - 0.344f * Cb - 0.714f * Cr;
        float b = Y + 1.770f * Cb;
        
        r = std::clamp(r, 0.0f, 255.0f);
        g = std::clamp(g, 0.0f, 255.0f);
        b = std::clamp(b, 0.0f, 255.0f);
        
        if (toBGR) {
            output[i * 3] = b;
            output[i * 3 + 1] = g;
            output[i * 3 + 2] = r;
        } else {
            output[i * 3] = r;
            output[i * 3 + 1] = g;
            output[i * 3 + 2] = b;
        }
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to XYZ
 */
inline std::vector<float> toXYZ(
    const float* input, int width, int height,
    bool isBGR = true
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float r, g, b;
        if (isBGR) {
            b = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            r = input[i * 3 + 2] / 255.0f;
        } else {
            r = input[i * 3] / 255.0f;
            g = input[i * 3 + 1] / 255.0f;
            b = input[i * 3 + 2] / 255.0f;
        }
        
        // sRGB gamma correction
        auto linearize = [](float v) {
            return (v > 0.04045f) ? std::pow((v + 0.055f) / 1.055f, 2.4f) : (v / 12.92f);
        };
        
        r = linearize(r);
        g = linearize(g);
        b = linearize(b);
        
        // sRGB to XYZ (D65 illuminant)
        float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
        float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
        float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
        
        output[i * 3] = X * 255.0f;
        output[i * 3 + 1] = Y * 255.0f;
        output[i * 3 + 2] = Z * 255.0f;
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to Lab
 */
inline std::vector<float> toLab(
    const float* input, int width, int height,
    bool isBGR = true
) {
    // First convert to XYZ
    auto xyz = toXYZ(input, width, height, isBGR);
    
    std::vector<float> output(width * height * 3);
    
    // D65 white point
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;
    
    auto f = [](float t) {
        const float delta = 6.0f / 29.0f;
        return (t > delta * delta * delta) 
            ? std::cbrt(t) 
            : (t / (3.0f * delta * delta) + 4.0f / 29.0f);
    };
    
    for (int i = 0; i < width * height; ++i) {
        float X = xyz[i * 3] / 255.0f / Xn;
        float Y = xyz[i * 3 + 1] / 255.0f / Yn;
        float Z = xyz[i * 3 + 2] / 255.0f / Zn;
        
        float fY = f(Y);
        float L = 116.0f * fY - 16.0f;
        float a = 500.0f * (f(X) - fY);
        float b = 200.0f * (fY - f(Z));
        
        // Scale to 0-255 range
        output[i * 3] = L * 255.0f / 100.0f;
        output[i * 3 + 1] = a + 128.0f;
        output[i * 3 + 2] = b + 128.0f;
    }
    
    return output;
}

/**
 * @brief Convert RGB/BGR to Luv
 */
inline std::vector<float> toLuv(
    const float* input, int width, int height,
    bool isBGR = true
) {
    auto xyz = toXYZ(input, width, height, isBGR);
    
    std::vector<float> output(width * height * 3);
    
    // D65 white point
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;
    const float un = 4.0f * Xn / (Xn + 15.0f * Yn + 3.0f * Zn);
    const float vn = 9.0f * Yn / (Xn + 15.0f * Yn + 3.0f * Zn);
    
    for (int i = 0; i < width * height; ++i) {
        float X = xyz[i * 3] / 255.0f;
        float Y = xyz[i * 3 + 1] / 255.0f;
        float Z = xyz[i * 3 + 2] / 255.0f;
        
        float denom = X + 15.0f * Y + 3.0f * Z;
        float u = (denom > 1e-6f) ? (4.0f * X / denom) : 0.0f;
        float v = (denom > 1e-6f) ? (9.0f * Y / denom) : 0.0f;
        
        float yr = Y / Yn;
        float L = (yr > 0.008856f) ? (116.0f * std::cbrt(yr) - 16.0f) : (903.3f * yr);
        
        float Luv_u = 13.0f * L * (u - un);
        float Luv_v = 13.0f * L * (v - vn);
        
        output[i * 3] = L * 255.0f / 100.0f;
        output[i * 3 + 1] = Luv_u + 134.0f;
        output[i * 3 + 2] = Luv_v + 140.0f;
    }
    
    return output;
}

/**
 * @brief General color conversion function
 */
inline std::vector<float> cvtColor(
    const float* input, int width, int height,
    int code
) {
    switch (code) {
        case COLOR_BGR2GRAY:
            return toGray(input, width, height, true);
        case COLOR_RGB2GRAY:
            return toGray(input, width, height, false);
        case COLOR_GRAY2BGR:
        case COLOR_GRAY2RGB:
            return grayToColor(input, width, height);
        case COLOR_BGR2RGB:
        case COLOR_RGB2BGR:
            return swapChannels(input, width, height);
        case COLOR_BGR2HSV:
            return toHSV(input, width, height, true);
        case COLOR_RGB2HSV:
            return toHSV(input, width, height, false);
        case COLOR_HSV2BGR:
            return hsvToColor(input, width, height, true);
        case COLOR_HSV2RGB:
            return hsvToColor(input, width, height, false);
        case COLOR_BGR2HLS:
            return toHLS(input, width, height, true);
        case COLOR_RGB2HLS:
            return toHLS(input, width, height, false);
        case COLOR_BGR2Lab:
            return toLab(input, width, height, true);
        case COLOR_RGB2Lab:
            return toLab(input, width, height, false);
        case COLOR_BGR2Luv:
            return toLuv(input, width, height, true);
        case COLOR_RGB2Luv:
            return toLuv(input, width, height, false);
        case COLOR_BGR2YCrCb:
            return toYCrCb(input, width, height, true);
        case COLOR_RGB2YCrCb:
            return toYCrCb(input, width, height, false);
        case COLOR_YCrCb2BGR:
            return ycrcbToColor(input, width, height, true);
        case COLOR_YCrCb2RGB:
            return ycrcbToColor(input, width, height, false);
        case COLOR_BGR2XYZ:
            return toXYZ(input, width, height, true);
        case COLOR_RGB2XYZ:
            return toXYZ(input, width, height, false);
        default:
            return std::vector<float>(input, input + width * height * 3);
    }
}

/**
 * @brief Split color image into channels
 */
inline void split(
    const float* input, int width, int height, int channels,
    std::vector<std::vector<float>>& output
) {
    output.resize(channels);
    for (int c = 0; c < channels; ++c) {
        output[c].resize(width * height);
        for (int i = 0; i < width * height; ++i) {
            output[c][i] = input[i * channels + c];
        }
    }
}

/**
 * @brief Merge channels into color image
 */
inline std::vector<float> merge(
    const std::vector<std::vector<float>>& channels,
    int width, int height
) {
    int numChannels = static_cast<int>(channels.size());
    std::vector<float> output(width * height * numChannels);
    
    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < numChannels; ++c) {
            output[i * numChannels + c] = channels[c][i];
        }
    }
    
    return output;
}

/**
 * @brief Extract single channel
 */
inline std::vector<float> extractChannel(
    const float* input, int width, int height, int channels,
    int channelIdx
) {
    std::vector<float> output(width * height);
    for (int i = 0; i < width * height; ++i) {
        output[i] = input[i * channels + channelIdx];
    }
    return output;
}

/**
 * @brief In-place channel insertion
 */
inline void insertChannel(
    float* output, int width, int height, int channels,
    const float* channel, int channelIdx
) {
    for (int i = 0; i < width * height; ++i) {
        output[i * channels + channelIdx] = channel[i];
    }
}

/**
 * @brief Apply color map to grayscale image
 */
enum class ColorMap {
    AUTUMN,
    BONE,
    JET,
    WINTER,
    RAINBOW,
    OCEAN,
    SUMMER,
    SPRING,
    COOL,
    HSV,
    HOT,
    PARULA,
    MAGMA,
    INFERNO,
    PLASMA,
    VIRIDIS,
    TURBO
};

inline std::vector<float> applyColorMap(
    const float* input, int width, int height,
    ColorMap colormap = ColorMap::JET
) {
    std::vector<float> output(width * height * 3);
    
    for (int i = 0; i < width * height; ++i) {
        float t = input[i] / 255.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        
        float r, g, b;
        
        switch (colormap) {
            case ColorMap::JET:
                if (t < 0.125f) {
                    r = 0; g = 0; b = 0.5f + t * 4;
                } else if (t < 0.375f) {
                    r = 0; g = (t - 0.125f) * 4; b = 1;
                } else if (t < 0.625f) {
                    r = (t - 0.375f) * 4; g = 1; b = 1 - (t - 0.375f) * 4;
                } else if (t < 0.875f) {
                    r = 1; g = 1 - (t - 0.625f) * 4; b = 0;
                } else {
                    r = 1 - (t - 0.875f) * 4; g = 0; b = 0;
                }
                break;
                
            case ColorMap::HOT:
                r = std::min(1.0f, t * 2.5f);
                g = std::clamp((t - 0.4f) * 2.5f, 0.0f, 1.0f);
                b = std::clamp((t - 0.7f) * 3.33f, 0.0f, 1.0f);
                break;
                
            case ColorMap::COOL:
                r = t; g = 1 - t; b = 1;
                break;
                
            case ColorMap::RAINBOW:
                r = std::abs(2 * t - 0.5f);
                g = std::sin(t * 3.14159265f);
                b = std::cos(t * 3.14159265f * 0.5f);
                break;
                
            case ColorMap::HSV:
            default: {
                float h = t * 6.0f;
                int hi = static_cast<int>(h) % 6;
                float f = h - static_cast<int>(h);
                switch (hi) {
                    case 0: r = 1; g = f; b = 0; break;
                    case 1: r = 1 - f; g = 1; b = 0; break;
                    case 2: r = 0; g = 1; b = f; break;
                    case 3: r = 0; g = 1 - f; b = 1; break;
                    case 4: r = f; g = 0; b = 1; break;
                    default: r = 1; g = 0; b = 1 - f; break;
                }
                break;
            }
        }
        
        output[i * 3] = b * 255.0f;
        output[i * 3 + 1] = g * 255.0f;
        output[i * 3 + 2] = r * 255.0f;
    }
    
    return output;
}

} // namespace imgproc
} // namespace neurova

#endif // NEUROVA_IMGPROC_COLOR_HPP
