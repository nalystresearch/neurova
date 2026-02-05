/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cstdint>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void rgb_to_gray_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    int offset = idx * channels;
    float r = src[offset + 0];
    float g = src[offset + 1];
    float b = src[offset + 2];
    uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    dst[idx] = gray;
}

__global__ void normalize_kernel(const float* src, float* dst, float mean, float inv_std, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = (src[idx] - mean) * inv_std;
}

}  // namespace

py::bool_ is_cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return py::bool_(false);
    }
    return py::bool_(count > 0);
}

py::array_t<uint8_t> rgb_to_gray(py::array input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 3 || buf.shape[2] < 3) {
        throw std::runtime_error("Expected HxWxC input with at least 3 channels");
    }
    const int height = static_cast<int>(buf.shape[0]);
    const int width = static_cast<int>(buf.shape[1]);
    const int channels = static_cast<int>(buf.shape[2]);
    const size_t pixels = static_cast<size_t>(height) * width;
    const size_t src_bytes = pixels * channels;

    const uint8_t* host_src = static_cast<const uint8_t*>(buf.ptr);
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    check_cuda(cudaMalloc(&d_src, src_bytes), "cudaMalloc src");
    check_cuda(cudaMalloc(&d_dst, pixels), "cudaMalloc dst");
    check_cuda(cudaMemcpy(d_src, host_src, src_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    int threads = 256;
    int blocks = static_cast<int>((pixels + threads - 1) / threads);
    rgb_to_gray_kernel<<<blocks, threads>>>(d_src, d_dst, width, height, channels);
    check_cuda(cudaGetLastError(), "rgb_to_gray kernel");

    py::array_t<uint8_t> output({height, width});
    check_cuda(cudaMemcpy(output.mutable_data(), d_dst, pixels, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    cudaFree(d_src);
    cudaFree(d_dst);
    return output;
}

py::array_t<float> normalize(py::array input, py::float_ mean, py::float_ std) {
    py::buffer_info buf = input.request();
    if (buf.ndim < 1) {
        throw std::runtime_error("Expected array with at least 1 dimension");
    }
    const int n = static_cast<int>(buf.size);
    const float* host_src = static_cast<const float*>(buf.ptr);
    float* d_src = nullptr;
    float* d_dst = nullptr;
    check_cuda(cudaMalloc(&d_src, n * sizeof(float)), "cudaMalloc src");
    check_cuda(cudaMalloc(&d_dst, n * sizeof(float)), "cudaMalloc dst");
    check_cuda(cudaMemcpy(d_src, host_src, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    float inv_std = 1.0f / (static_cast<float>(std) + 1e-8f);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_src, d_dst, static_cast<float>(mean), inv_std, n);
    check_cuda(cudaGetLastError(), "normalize kernel");

    py::array_t<float> output(buf.shape);
    check_cuda(cudaMemcpy(output.mutable_data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    cudaFree(d_src);
    cudaFree(d_dst);
    return output;
}

std::string runtime_version() {
    int driver = 0;
    int runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    return "driver=" + std::to_string(driver) + ",runtime=" + std::to_string(runtime);
}

PYBIND11_MODULE(cuda_native, m) {
    m.doc() = "CUDA acceleration helpers";
    m.def("is_cuda_available", &is_cuda_available);
    m.def("rgb_to_gray", &rgb_to_gray, "Convert RGB image to grayscale on GPU");
    m.def("normalize", &normalize, py::arg("arr"), py::arg("mean"), py::arg("std"));
    m.def("runtime_version", &runtime_version);
}
