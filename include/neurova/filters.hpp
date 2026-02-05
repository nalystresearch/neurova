// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file filters.hpp
 * @brief Image filtering module - main header
 */

#ifndef NEUROVA_FILTERS_MODULE_HPP
#define NEUROVA_FILTERS_MODULE_HPP

#include "filters/kernels.hpp"
#include "filters/convolution.hpp"
#include "filters/blur.hpp"
#include "filters/edges.hpp"
#include "filters/bilateral.hpp"

namespace neurova {
namespace filters {

// Re-export commonly used functions at module level

/**
 * @brief Apply a general filter with automatic method selection
 */
enum class FilterType {
    BOX,
    GAUSSIAN,
    MEDIAN,
    BILATERAL,
    SHARPEN,
    EDGE,
    SOBEL,
    CANNY,
    LAPLACIAN
};

/**
 * @brief Apply filter by type
 */
inline std::vector<float> applyFilter(
    const float* input, int width, int height,
    FilterType type,
    int ksize = 3,
    float param1 = 0.0f,
    float param2 = 0.0f
) {
    switch (type) {
        case FilterType::BOX:
            return boxBlur(input, width, height, ksize);
            
        case FilterType::GAUSSIAN:
            return gaussianBlur(input, width, height, ksize, param1);
            
        case FilterType::MEDIAN:
            return medianBlur(input, width, height, ksize);
            
        case FilterType::BILATERAL:
            return bilateralFilter(input, width, height, ksize, 
                                  param1 > 0 ? param1 : 75.0f,
                                  param2 > 0 ? param2 : 75.0f);
            
        case FilterType::SHARPEN:
            return sharpen(input, width, height, param1 > 0 ? param1 : 1.0f);
            
        case FilterType::EDGE:
            return detectEdges(input, width, height);
            
        case FilterType::SOBEL:
            return sobel(input, width, height, 1, 1, ksize);
            
        case FilterType::CANNY:
            return canny(input, width, height, 
                        param1 > 0 ? param1 : 50.0f,
                        param2 > 0 ? param2 : 150.0f, ksize);
            
        case FilterType::LAPLACIAN:
            return laplacian(input, width, height, ksize);
            
        default:
            return std::vector<float>(input, input + width * height);
    }
}

/**
 * @brief Chain multiple filters together
 */
class FilterChain {
public:
    struct FilterOp {
        FilterType type;
        int ksize;
        float param1;
        float param2;
    };

    FilterChain& add(FilterType type, int ksize = 3, 
                     float param1 = 0.0f, float param2 = 0.0f) {
        ops_.push_back({type, ksize, param1, param2});
        return *this;
    }

    std::vector<float> apply(const float* input, int width, int height) const {
        std::vector<float> current(input, input + width * height);
        
        for (const auto& op : ops_) {
            current = applyFilter(current.data(), width, height,
                                 op.type, op.ksize, op.param1, op.param2);
        }
        
        return current;
    }

    void clear() { ops_.clear(); }

private:
    std::vector<FilterOp> ops_;
};

} // namespace filters
} // namespace neurova

#endif // NEUROVA_FILTERS_MODULE_HPP
