// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file photo.hpp
 * @brief Computational photography operations
 * 
 * Neurova implementation of photo enhancement and restoration.
 */

#pragma once

#include "core/image.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

namespace neurova {
namespace photo {

// ============================================================================
// Inpainting
// ============================================================================

/**
 * @brief Telea inpainting algorithm
 */
inline Image inpaint_telea(const Image& image, const Image& mask, int radius = 3) {
    Image result = image;
    Image band(image.width(), image.height(), 1);
    
    // Find pixels to inpaint (non-zero in mask)
    std::vector<std::tuple<int, int, float>> pixels_to_fill;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (mask.at(x, y, 0) > 0) {
                // Calculate distance to known region
                float min_dist = static_cast<float>(radius + 1);
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < image.width() && ny >= 0 && ny < image.height()) {
                            if (mask.at(nx, ny, 0) == 0) {
                                float dist = std::sqrt(static_cast<float>(dx*dx + dy*dy));
                                min_dist = std::min(min_dist, dist);
                            }
                        }
                    }
                }
                pixels_to_fill.push_back({x, y, min_dist});
            }
        }
    }
    
    // Sort by distance (fill from boundary inward)
    std::sort(pixels_to_fill.begin(), pixels_to_fill.end(),
              [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
    
    // Fill pixels
    for (const auto& [x, y, dist] : pixels_to_fill) {
        float sum_weight = 0;
        std::vector<float> sum_color(image.channels(), 0);
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < image.width() && ny >= 0 && ny < image.height()) {
                    // Only use known pixels (original or already filled)
                    if (mask.at(nx, ny, 0) == 0 || band.at(nx, ny, 0) > 0) {
                        float r = std::sqrt(static_cast<float>(dx*dx + dy*dy));
                        if (r > 0 && r <= radius) {
                            // Weight based on distance and direction
                            float weight = 1.0f / (r * r);
                            
                            sum_weight += weight;
                            for (int c = 0; c < image.channels(); ++c) {
                                sum_color[c] += result.at(nx, ny, c) * weight;
                            }
                        }
                    }
                }
            }
        }
        
        if (sum_weight > 0) {
            for (int c = 0; c < image.channels(); ++c) {
                result.at(x, y, c) = sum_color[c] / sum_weight;
            }
            band.at(x, y, 0) = 1;  // Mark as filled
        }
    }
    
    return result;
}

/**
 * @brief Navier-Stokes inpainting
 */
inline Image inpaint_ns(const Image& image, const Image& mask, int radius = 3) {
    // Simplified version using diffusion
    Image result = image;
    
    for (int iter = 0; iter < radius * 10; ++iter) {
        Image temp = result;
        
        for (int y = 1; y < image.height() - 1; ++y) {
            for (int x = 1; x < image.width() - 1; ++x) {
                if (mask.at(x, y, 0) > 0) {
                    for (int c = 0; c < image.channels(); ++c) {
                        // Laplacian diffusion
                        float laplacian = temp.at(x-1, y, c) + temp.at(x+1, y, c) +
                                         temp.at(x, y-1, c) + temp.at(x, y+1, c) -
                                         4.0f * temp.at(x, y, c);
                        result.at(x, y, c) = temp.at(x, y, c) + 0.25f * laplacian;
                    }
                }
            }
        }
    }
    
    return result;
}

// ============================================================================
// Denoising
// ============================================================================

/**
 * @brief Fast non-local means denoising
 */
inline Image fast_nl_means_denoising(const Image& image, float h = 10.0f,
                                      int template_size = 7, int search_size = 21) {
    Image result(image.width(), image.height(), image.channels());
    
    int t_half = template_size / 2;
    int s_half = search_size / 2;
    float h2 = h * h;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                float sum_weight = 0;
                float sum_value = 0;
                
                // Search window
                for (int sy = -s_half; sy <= s_half; ++sy) {
                    for (int sx = -s_half; sx <= s_half; ++sx) {
                        int nx = x + sx;
                        int ny = y + sy;
                        
                        if (nx < 0 || nx >= image.width() || ny < 0 || ny >= image.height()) continue;
                        
                        // Calculate patch distance
                        float dist = 0;
                        int count = 0;
                        
                        for (int ty = -t_half; ty <= t_half; ++ty) {
                            for (int tx = -t_half; tx <= t_half; ++tx) {
                                int px1 = x + tx;
                                int py1 = y + ty;
                                int px2 = nx + tx;
                                int py2 = ny + ty;
                                
                                if (px1 >= 0 && px1 < image.width() && py1 >= 0 && py1 < image.height() &&
                                    px2 >= 0 && px2 < image.width() && py2 >= 0 && py2 < image.height()) {
                                    float diff = image.at(px1, py1, c) - image.at(px2, py2, c);
                                    dist += diff * diff;
                                    count++;
                                }
                            }
                        }
                        
                        if (count > 0) {
                            dist /= count;
                            float weight = std::exp(-dist / h2);
                            sum_weight += weight;
                            sum_value += weight * image.at(nx, ny, c);
                        }
                    }
                }
                
                result.at(x, y, c) = sum_weight > 0 ? sum_value / sum_weight : image.at(x, y, c);
            }
        }
    }
    
    return result;
}

/**
 * @brief Bilateral filter denoising
 */
inline Image bilateral_filter(const Image& image, int d = 5, float sigma_color = 75.0f,
                               float sigma_space = 75.0f) {
    Image result(image.width(), image.height(), image.channels());
    int radius = d / 2;
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                float sum_weight = 0;
                float sum_value = 0;
                float center_val = image.at(x, y, c);
                
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < image.width() && ny >= 0 && ny < image.height()) {
                            float neighbor_val = image.at(nx, ny, c);
                            
                            // Spatial weight
                            float spatial_weight = std::exp(-(dx*dx + dy*dy) / (2 * sigma_space * sigma_space));
                            
                            // Color weight
                            float color_diff = center_val - neighbor_val;
                            float color_weight = std::exp(-(color_diff * color_diff) / (2 * sigma_color * sigma_color));
                            
                            float weight = spatial_weight * color_weight;
                            sum_weight += weight;
                            sum_value += weight * neighbor_val;
                        }
                    }
                }
                
                result.at(x, y, c) = sum_value / sum_weight;
            }
        }
    }
    
    return result;
}

// ============================================================================
// HDR and Tone Mapping
// ============================================================================

/**
 * @brief Create HDR image from exposure bracketed images
 */
inline Image create_hdr(const std::vector<Image>& images, const std::vector<float>& exposure_times) {
    if (images.empty()) return Image();
    
    int width = images[0].width();
    int height = images[0].height();
    int channels = images[0].channels();
    
    Image hdr(width, height, channels);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float sum_radiance = 0;
                float sum_weight = 0;
                
                for (size_t i = 0; i < images.size(); ++i) {
                    float val = images[i].at(x, y, c);
                    
                    // Weight function (hat function)
                    float weight;
                    if (val <= 127.5f) {
                        weight = val / 127.5f;
                    } else {
                        weight = (255.0f - val) / 127.5f;
                    }
                    weight = std::max(0.01f, weight);
                    
                    // Estimate radiance
                    float radiance = val / (255.0f * exposure_times[i]);
                    
                    sum_radiance += weight * radiance;
                    sum_weight += weight;
                }
                
                hdr.at(x, y, c) = sum_weight > 0 ? sum_radiance / sum_weight : 0;
            }
        }
    }
    
    return hdr;
}

/**
 * @brief Reinhard tone mapping
 */
inline Image tonemap_reinhard(const Image& hdr, float gamma = 2.2f, float intensity = 0.0f,
                               float light_adapt = 1.0f, float color_adapt = 0.0f) {
    Image result(hdr.width(), hdr.height(), hdr.channels());
    
    // Calculate log-average luminance
    float sum_log = 0;
    float delta = 1e-6f;
    
    for (int y = 0; y < hdr.height(); ++y) {
        for (int x = 0; x < hdr.width(); ++x) {
            float lum;
            if (hdr.channels() >= 3) {
                lum = 0.2126f * hdr.at(x, y, 0) + 0.7152f * hdr.at(x, y, 1) + 0.0722f * hdr.at(x, y, 2);
            } else {
                lum = hdr.at(x, y, 0);
            }
            sum_log += std::log(lum + delta);
        }
    }
    
    float log_avg = std::exp(sum_log / (hdr.width() * hdr.height()));
    float key = std::pow(2.0f, intensity);
    
    // Tone map
    for (int y = 0; y < hdr.height(); ++y) {
        for (int x = 0; x < hdr.width(); ++x) {
            for (int c = 0; c < hdr.channels(); ++c) {
                float val = hdr.at(x, y, c);
                float scaled = key * val / log_avg;
                float mapped = scaled / (1.0f + scaled);
                
                // Gamma correction
                mapped = std::pow(mapped, 1.0f / gamma);
                
                result.at(x, y, c) = std::max(0.0f, std::min(255.0f, mapped * 255.0f));
            }
        }
    }
    
    return result;
}

/**
 * @brief Drago tone mapping
 */
inline Image tonemap_drago(const Image& hdr, float gamma = 2.2f, float saturation = 1.0f,
                            float bias = 0.85f) {
    Image result(hdr.width(), hdr.height(), hdr.channels());
    
    // Find max luminance
    float max_lum = 0;
    for (int y = 0; y < hdr.height(); ++y) {
        for (int x = 0; x < hdr.width(); ++x) {
            float lum;
            if (hdr.channels() >= 3) {
                lum = 0.2126f * hdr.at(x, y, 0) + 0.7152f * hdr.at(x, y, 1) + 0.0722f * hdr.at(x, y, 2);
            } else {
                lum = hdr.at(x, y, 0);
            }
            max_lum = std::max(max_lum, lum);
        }
    }
    
    float divider = std::log10(1.0f + max_lum);
    float bias_p = std::log(bias) / std::log(0.5f);
    
    for (int y = 0; y < hdr.height(); ++y) {
        for (int x = 0; x < hdr.width(); ++x) {
            float lum;
            if (hdr.channels() >= 3) {
                lum = 0.2126f * hdr.at(x, y, 0) + 0.7152f * hdr.at(x, y, 1) + 0.0722f * hdr.at(x, y, 2);
            } else {
                lum = hdr.at(x, y, 0);
            }
            
            float lum_mapped = std::log10(1.0f + lum) / divider;
            lum_mapped = std::pow(lum_mapped, bias_p);
            
            float scale = lum > 0 ? lum_mapped / lum : 0;
            
            for (int c = 0; c < hdr.channels(); ++c) {
                float val = hdr.at(x, y, c) * scale;
                
                // Apply saturation
                if (saturation != 1.0f && hdr.channels() >= 3) {
                    val = lum_mapped + saturation * (val - lum_mapped);
                }
                
                // Gamma correction
                val = std::pow(std::max(0.0f, val), 1.0f / gamma);
                
                result.at(x, y, c) = std::max(0.0f, std::min(255.0f, val * 255.0f));
            }
        }
    }
    
    return result;
}

// ============================================================================
// Dehazing
// ============================================================================

/**
 * @brief Dark channel prior dehazing
 */
inline Image dehaze(const Image& image, float omega = 0.95f, int patch_size = 15) {
    int half = patch_size / 2;
    
    // Compute dark channel
    Image dark(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float min_val = 255.0f;
            
            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    int nx = std::max(0, std::min(image.width() - 1, x + dx));
                    int ny = std::max(0, std::min(image.height() - 1, y + dy));
                    
                    for (int c = 0; c < image.channels(); ++c) {
                        min_val = std::min(min_val, image.at(nx, ny, c));
                    }
                }
            }
            
            dark.at(x, y, 0) = min_val;
        }
    }
    
    // Estimate atmospheric light (top 0.1% brightest pixels in dark channel)
    std::vector<std::pair<float, std::pair<int, int>>> pixels;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            pixels.push_back({dark.at(x, y, 0), {x, y}});
        }
    }
    
    std::sort(pixels.rbegin(), pixels.rend());
    
    std::vector<float> A(image.channels(), 0);
    int n_top = std::max(1, static_cast<int>(pixels.size() * 0.001));
    
    for (int i = 0; i < n_top; ++i) {
        auto [x, y] = pixels[i].second;
        for (int c = 0; c < image.channels(); ++c) {
            A[c] = std::max(A[c], image.at(x, y, c));
        }
    }
    
    // Estimate transmission
    Image transmission(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float min_val = 1.0f;
            
            for (int dy = -half; dy <= half; ++dy) {
                for (int dx = -half; dx <= half; ++dx) {
                    int nx = std::max(0, std::min(image.width() - 1, x + dx));
                    int ny = std::max(0, std::min(image.height() - 1, y + dy));
                    
                    for (int c = 0; c < image.channels(); ++c) {
                        min_val = std::min(min_val, image.at(nx, ny, c) / A[c]);
                    }
                }
            }
            
            transmission.at(x, y, 0) = 1.0f - omega * min_val;
        }
    }
    
    // Recover scene radiance
    Image result(image.width(), image.height(), image.channels());
    float t0 = 0.1f;  // Minimum transmission
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float t = std::max(t0, transmission.at(x, y, 0));
            
            for (int c = 0; c < image.channels(); ++c) {
                float val = (image.at(x, y, c) - A[c]) / t + A[c];
                result.at(x, y, c) = std::max(0.0f, std::min(255.0f, val));
            }
        }
    }
    
    return result;
}

// ============================================================================
// Color Balance and Correction
// ============================================================================

/**
 * @brief White balance correction (Gray World assumption)
 */
inline Image white_balance_gray_world(const Image& image) {
    if (image.channels() < 3) return image;
    
    // Calculate channel means
    float mean_r = 0, mean_g = 0, mean_b = 0;
    int count = image.width() * image.height();
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            mean_r += image.at(x, y, 0);
            mean_g += image.at(x, y, 1);
            mean_b += image.at(x, y, 2);
        }
    }
    
    mean_r /= count;
    mean_g /= count;
    mean_b /= count;
    
    float avg = (mean_r + mean_g + mean_b) / 3.0f;
    float scale_r = avg / mean_r;
    float scale_g = avg / mean_g;
    float scale_b = avg / mean_b;
    
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            result.at(x, y, 0) = std::max(0.0f, std::min(255.0f, image.at(x, y, 0) * scale_r));
            result.at(x, y, 1) = std::max(0.0f, std::min(255.0f, image.at(x, y, 1) * scale_g));
            result.at(x, y, 2) = std::max(0.0f, std::min(255.0f, image.at(x, y, 2) * scale_b));
        }
    }
    
    return result;
}

/**
 * @brief Color transfer between images
 */
inline Image color_transfer(const Image& source, const Image& target) {
    if (source.channels() < 3 || target.channels() < 3) return source;
    
    // Calculate statistics in LAB-like space
    auto rgb_to_lab = [](float r, float g, float b) {
        // Simplified Lab conversion
        float l = 0.3811f * r + 0.5783f * g + 0.0402f * b;
        float m = 0.1967f * r + 0.7244f * g + 0.0782f * b;
        float s = 0.0241f * r + 0.1288f * g + 0.8444f * b;
        
        l = l > 0 ? std::log(l) : 0;
        m = m > 0 ? std::log(m) : 0;
        s = s > 0 ? std::log(s) : 0;
        
        return std::make_tuple(
            (l + m + s) / std::sqrt(3.0f),
            (l + m - 2*s) / std::sqrt(6.0f),
            (l - m) / std::sqrt(2.0f)
        );
    };
    
    auto lab_to_rgb = [](float L, float a, float b) {
        float l = L / std::sqrt(3.0f) + a / std::sqrt(6.0f) + b / std::sqrt(2.0f);
        float m = L / std::sqrt(3.0f) + a / std::sqrt(6.0f) - b / std::sqrt(2.0f);
        float s = L / std::sqrt(3.0f) - 2*a / std::sqrt(6.0f);
        
        l = std::exp(l);
        m = std::exp(m);
        s = std::exp(s);
        
        float r = 4.4679f * l - 3.5873f * m + 0.1193f * s;
        float g = -1.2186f * l + 2.3809f * m - 0.1624f * s;
        float B = 0.0497f * l - 0.2439f * m + 1.2045f * s;
        
        return std::make_tuple(r, g, B);
    };
    
    // Calculate source statistics
    float src_mean_L = 0, src_mean_a = 0, src_mean_b = 0;
    float src_std_L = 0, src_std_a = 0, src_std_b = 0;
    
    for (int y = 0; y < source.height(); ++y) {
        for (int x = 0; x < source.width(); ++x) {
            auto [L, a, b] = rgb_to_lab(source.at(x, y, 0), source.at(x, y, 1), source.at(x, y, 2));
            src_mean_L += L;
            src_mean_a += a;
            src_mean_b += b;
        }
    }
    
    int src_count = source.width() * source.height();
    src_mean_L /= src_count;
    src_mean_a /= src_count;
    src_mean_b /= src_count;
    
    for (int y = 0; y < source.height(); ++y) {
        for (int x = 0; x < source.width(); ++x) {
            auto [L, a, b] = rgb_to_lab(source.at(x, y, 0), source.at(x, y, 1), source.at(x, y, 2));
            src_std_L += (L - src_mean_L) * (L - src_mean_L);
            src_std_a += (a - src_mean_a) * (a - src_mean_a);
            src_std_b += (b - src_mean_b) * (b - src_mean_b);
        }
    }
    
    src_std_L = std::sqrt(src_std_L / src_count);
    src_std_a = std::sqrt(src_std_a / src_count);
    src_std_b = std::sqrt(src_std_b / src_count);
    
    // Calculate target statistics
    float tgt_mean_L = 0, tgt_mean_a = 0, tgt_mean_b = 0;
    float tgt_std_L = 0, tgt_std_a = 0, tgt_std_b = 0;
    
    for (int y = 0; y < target.height(); ++y) {
        for (int x = 0; x < target.width(); ++x) {
            auto [L, a, b] = rgb_to_lab(target.at(x, y, 0), target.at(x, y, 1), target.at(x, y, 2));
            tgt_mean_L += L;
            tgt_mean_a += a;
            tgt_mean_b += b;
        }
    }
    
    int tgt_count = target.width() * target.height();
    tgt_mean_L /= tgt_count;
    tgt_mean_a /= tgt_count;
    tgt_mean_b /= tgt_count;
    
    for (int y = 0; y < target.height(); ++y) {
        for (int x = 0; x < target.width(); ++x) {
            auto [L, a, b] = rgb_to_lab(target.at(x, y, 0), target.at(x, y, 1), target.at(x, y, 2));
            tgt_std_L += (L - tgt_mean_L) * (L - tgt_mean_L);
            tgt_std_a += (a - tgt_mean_a) * (a - tgt_mean_a);
            tgt_std_b += (b - tgt_mean_b) * (b - tgt_mean_b);
        }
    }
    
    tgt_std_L = std::sqrt(tgt_std_L / tgt_count);
    tgt_std_a = std::sqrt(tgt_std_a / tgt_count);
    tgt_std_b = std::sqrt(tgt_std_b / tgt_count);
    
    // Transfer colors
    Image result(source.width(), source.height(), source.channels());
    
    for (int y = 0; y < source.height(); ++y) {
        for (int x = 0; x < source.width(); ++x) {
            auto [L, a, b] = rgb_to_lab(source.at(x, y, 0), source.at(x, y, 1), source.at(x, y, 2));
            
            // Subtract mean, scale by std ratio, add target mean
            L = (L - src_mean_L) * (tgt_std_L / (src_std_L + 1e-6f)) + tgt_mean_L;
            a = (a - src_mean_a) * (tgt_std_a / (src_std_a + 1e-6f)) + tgt_mean_a;
            b = (b - src_mean_b) * (tgt_std_b / (src_std_b + 1e-6f)) + tgt_mean_b;
            
            auto [r, g, B] = lab_to_rgb(L, a, b);
            
            result.at(x, y, 0) = std::max(0.0f, std::min(255.0f, r));
            result.at(x, y, 1) = std::max(0.0f, std::min(255.0f, g));
            result.at(x, y, 2) = std::max(0.0f, std::min(255.0f, B));
        }
    }
    
    return result;
}

// ============================================================================
// Seamless Cloning
// ============================================================================

/**
 * @brief Seamless clone (Poisson blending - simplified)
 */
inline Image seamless_clone(const Image& source, const Image& target, const Image& mask,
                            int center_x, int center_y) {
    Image result = target;
    
    // Find mask bounds
    int min_x = source.width(), max_x = 0;
    int min_y = source.height(), max_y = 0;
    
    for (int y = 0; y < source.height(); ++y) {
        for (int x = 0; x < source.width(); ++x) {
            if (mask.at(x, y, 0) > 127) {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }
    
    int offset_x = center_x - (min_x + max_x) / 2;
    int offset_y = center_y - (min_y + max_y) / 2;
    
    // Simple gradient-domain blending
    for (int iter = 0; iter < 100; ++iter) {
        Image temp = result;
        
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                if (mask.at(x, y, 0) > 127) {
                    int tx = x + offset_x;
                    int ty = y + offset_y;
                    
                    if (tx >= 1 && tx < target.width() - 1 && 
                        ty >= 1 && ty < target.height() - 1) {
                        
                        for (int c = 0; c < source.channels() && c < target.channels(); ++c) {
                            // Calculate source gradient
                            float src_grad = 4 * source.at(x, y, c) -
                                           source.at(x-1, y, c) - source.at(x+1, y, c) -
                                           source.at(x, y-1, c) - source.at(x, y+1, c);
                            
                            // Solve Poisson equation
                            float val = (result.at(tx-1, ty, c) + result.at(tx+1, ty, c) +
                                        result.at(tx, ty-1, c) + result.at(tx, ty+1, c) + src_grad) / 4.0f;
                            
                            temp.at(tx, ty, c) = std::max(0.0f, std::min(255.0f, val));
                        }
                    }
                }
            }
        }
        
        result = temp;
    }
    
    return result;
}

// ============================================================================
// Stylization
// ============================================================================

/**
 * @brief Pencil sketch effect
 */
inline std::pair<Image, Image> pencil_sketch(const Image& image, float sigma_s = 60.0f,
                                              float sigma_r = 0.07f, float shade_factor = 0.02f) {
    // Convert to grayscale
    Image gray(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.channels() >= 3) {
                gray.at(x, y, 0) = 0.299f * image.at(x, y, 0) + 
                                  0.587f * image.at(x, y, 1) + 
                                  0.114f * image.at(x, y, 2);
            } else {
                gray.at(x, y, 0) = image.at(x, y, 0);
            }
        }
    }
    
    // Edge detection for sketch
    Image sketch(image.width(), image.height(), 1);
    for (int y = 1; y < image.height() - 1; ++y) {
        for (int x = 1; x < image.width() - 1; ++x) {
            float gx = gray.at(x+1, y, 0) - gray.at(x-1, y, 0);
            float gy = gray.at(x, y+1, 0) - gray.at(x, y-1, 0);
            float mag = std::sqrt(gx*gx + gy*gy);
            sketch.at(x, y, 0) = 255.0f - std::min(255.0f, mag * 2);
        }
    }
    
    // Color pencil (simplified bilateral + edge)
    Image color_sketch = bilateral_filter(image, 5, sigma_r * 255, sigma_s);
    
    // Blend with sketch
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float edge = sketch.at(x, y, 0) / 255.0f;
            for (int c = 0; c < color_sketch.channels(); ++c) {
                color_sketch.at(x, y, c) *= edge;
            }
        }
    }
    
    return {sketch, color_sketch};
}

/**
 * @brief Stylization filter
 */
inline Image stylize(const Image& image, float sigma_s = 60.0f, float sigma_r = 0.45f) {
    // Edge-preserving smoothing
    Image smoothed = bilateral_filter(image, static_cast<int>(sigma_s / 10), 
                                       sigma_r * 255, sigma_s);
    
    // Quantize colors
    int levels = 8;
    for (int y = 0; y < smoothed.height(); ++y) {
        for (int x = 0; x < smoothed.width(); ++x) {
            for (int c = 0; c < smoothed.channels(); ++c) {
                float val = smoothed.at(x, y, c);
                val = std::round(val * levels / 255.0f) * 255.0f / levels;
                smoothed.at(x, y, c) = val;
            }
        }
    }
    
    return smoothed;
}

/**
 * @brief Oil painting effect
 */
inline Image oil_painting(const Image& image, int radius = 4, int levels = 8) {
    Image result(image.width(), image.height(), image.channels());
    
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            // Histogram of intensity levels
            std::vector<int> hist(levels, 0);
            std::vector<std::vector<float>> sum(levels, std::vector<float>(image.channels(), 0));
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = std::max(0, std::min(image.width() - 1, x + dx));
                    int ny = std::max(0, std::min(image.height() - 1, y + dy));
                    
                    float intensity;
                    if (image.channels() >= 3) {
                        intensity = (image.at(nx, ny, 0) + image.at(nx, ny, 1) + image.at(nx, ny, 2)) / 3.0f;
                    } else {
                        intensity = image.at(nx, ny, 0);
                    }
                    
                    int bin = std::min(levels - 1, static_cast<int>(intensity * levels / 256.0f));
                    hist[bin]++;
                    
                    for (int c = 0; c < image.channels(); ++c) {
                        sum[bin][c] += image.at(nx, ny, c);
                    }
                }
            }
            
            // Find dominant bin
            int max_bin = 0;
            for (int i = 1; i < levels; ++i) {
                if (hist[i] > hist[max_bin]) {
                    max_bin = i;
                }
            }
            
            // Set pixel to average of dominant bin
            if (hist[max_bin] > 0) {
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = sum[max_bin][c] / hist[max_bin];
                }
            } else {
                for (int c = 0; c < image.channels(); ++c) {
                    result.at(x, y, c) = image.at(x, y, c);
                }
            }
        }
    }
    
    return result;
}

} // namespace photo
} // namespace neurova
