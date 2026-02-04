/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <webp/decode.h>
#include <webp/demux.h>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

py::array decode_webp(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open WebP file");
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw std::runtime_error("WebP file is empty");
    }

    WebPDecoderConfig config;
    if (!WebPInitDecoderConfig(&config)) {
        throw std::runtime_error("Failed to init WebP decoder config");
    }

    if (WebPGetFeatures(bytes.data(), bytes.size(), &config.input) != VP8_STATUS_OK) {
        throw std::runtime_error("Invalid WebP data");
    }

    const int width = config.input.width;
    const int height = config.input.height;
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4u);

    config.output.colorspace = MODE_RGBA;
    config.output.u.RGBA.rgba = rgba.data();
    config.output.u.RGBA.stride = width * 4;
    config.output.u.RGBA.size = rgba.size();
    config.output.is_external_memory = 1;

    VP8StatusCode status = WebPDecode(bytes.data(), bytes.size(), &config);
    if (status != VP8_STATUS_OK) {
        throw std::runtime_error("WebP decode failed");
    }

    py::array_t<uint8_t> array({height, width, 3});
    auto buf = array.request();
    auto* dst = static_cast<uint8_t*>(buf.ptr);
    for (int i = 0; i < width * height; ++i) {
        dst[i * 3 + 0] = rgba[i * 4 + 0];
        dst[i * 3 + 1] = rgba[i * 4 + 1];
        dst[i * 3 + 2] = rgba[i * 4 + 2];
    }

    return array;
}

}  // namespace

PYBIND11_MODULE(codecs_webp, m) {
    m.doc() = "WebP codec bindings";
    m.def("imread", &decode_webp, py::arg("path"));
}
