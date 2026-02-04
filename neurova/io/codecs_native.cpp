/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <jpeglib.h>
#include <png.h>

namespace py = pybind11;

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static py::array read_png_rgb(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Failed to open PNG file");

    png_byte header[8];
    if (std::fread(header, 1, 8, fp) != 8) {
        std::fclose(fp);
        throw std::runtime_error("Failed to read PNG header");
    }
    if (png_sig_cmp(header, 0, 8) != 0) {
        std::fclose(fp);
        throw std::runtime_error("Not a PNG file");
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::fclose(fp);
        throw std::runtime_error("png_create_read_struct failed");
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("libpng error while reading");
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width = 0, height = 0;
    int bit_depth = 0, color_type = 0;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);

    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);

    // Ensure we output RGB (strip alpha if present, convert gray->rgb).
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }
    png_set_strip_alpha(png_ptr);

    png_read_update_info(png_ptr, info_ptr);

    const png_size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    if (rowbytes != static_cast<png_size_t>(width) * 3) {
        // After conversions we expect packed RGB24.
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("Unexpected PNG row stride after conversion");
    }

    std::vector<unsigned char> buf(rowbytes * static_cast<size_t>(height));
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
        rows[y] = reinterpret_cast<png_bytep>(buf.data() + y * rowbytes);
    }

    png_read_image(png_ptr, rows.data());
    png_read_end(png_ptr, nullptr);

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    std::fclose(fp);

    py::array_t<unsigned char> out({static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), 3});
    std::memcpy(out.mutable_data(), buf.data(), buf.size());
    return out;
}

static py::array read_jpeg(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Failed to open JPEG file");

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, fp);

    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("Invalid JPEG header");
    }

    cinfo.out_color_space = (cinfo.num_components == 1) ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_start_decompress(&cinfo);

    const int width = static_cast<int>(cinfo.output_width);
    const int height = static_cast<int>(cinfo.output_height);
    const int channels = static_cast<int>(cinfo.output_components);

    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw std::runtime_error("Unsupported JPEG format");
    }

    const size_t row_stride = static_cast<size_t>(width) * static_cast<size_t>(channels);
    std::vector<unsigned char> buf(static_cast<size_t>(height) * row_stride);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* rowptr = buf.data() + static_cast<size_t>(cinfo.output_scanline) * row_stride;
        JSAMPROW row[1] = {rowptr};
        jpeg_read_scanlines(&cinfo, row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);

    if (channels == 1) {
        py::array_t<unsigned char> out({height, width});
        std::memcpy(out.mutable_data(), buf.data(), buf.size());
        return out;
    }

    py::array_t<unsigned char> out({height, width, 3});
    std::memcpy(out.mutable_data(), buf.data(), buf.size());
    return out;
}

static py::array imread_native(const std::string& path) {
    const std::string p = to_lower(path);
    if (ends_with(p, ".png")) return read_png_rgb(path);
    if (ends_with(p, ".jpg") || ends_with(p, ".jpeg")) return read_jpeg(path);
    throw std::runtime_error("Unsupported image format for native codecs");
}

PYBIND11_MODULE(codecs_native, m) {
    m.doc() = "Neurova native image codecs (libpng + libjpeg)";
    m.def("imread", &imread_native, "Read PNG/JPEG into a NumPy array (RGB or gray)");
}
