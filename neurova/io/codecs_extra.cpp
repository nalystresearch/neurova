/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <memory>

namespace py = pybind11;

#if __has_include(<tiffio.h>)
#define NEUROVA_HAS_TIFF 1
#include <tiffio.h>
#else
#define NEUROVA_HAS_TIFF 0
#endif

#if __has_include(<OpenEXR/ImfRgbaFile.h>) && __has_include(<Imath/ImathBox.h>)
#define NEUROVA_HAS_OPENEXR 1
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfArray.h>
#include <Imath/ImathBox.h>
using namespace OPENEXR_IMF_NAMESPACE;
using namespace IMATH_NAMESPACE;
#else
#define NEUROVA_HAS_OPENEXR 0
#endif

#if __has_include(<openjpeg.h>)
#define NEUROVA_HAS_JP2 1
#include <openjpeg.h>
#else
#define NEUROVA_HAS_JP2 0
#endif

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

#if NEUROVA_HAS_TIFF
py::array read_tiff(const std::string& path) {
    TIFF* tif = TIFFOpen(path.c_str(), "rb");
    if (!tif) {
        throw std::runtime_error("Failed to open TIFF file");
    }
    uint32 width = 0;
    uint32 height = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    std::vector<uint32> raster(width * height);
    if (!TIFFReadRGBAImageOriented(tif, width, height, raster.data(), ORIENTATION_TOPLEFT, 0)) {
        TIFFClose(tif);
        throw std::runtime_error("Failed to decode TIFF");
    }
    TIFFClose(tif);

    py::array_t<uint8_t> array({static_cast<int>(height), static_cast<int>(width), 3});
    auto buf = array.request();
    uint8_t* dst = static_cast<uint8_t*>(buf.ptr);
    for (size_t i = 0; i < raster.size(); ++i) {
        uint32 pixel = raster[i];
        dst[i * 3 + 0] = TIFFGetR(pixel);
        dst[i * 3 + 1] = TIFFGetG(pixel);
        dst[i * 3 + 2] = TIFFGetB(pixel);
    }
    return array;
}
#endif

#if NEUROVA_HAS_OPENEXR
py::array read_exr(const std::string& path) {
    RgbaInputFile file(path.c_str());
    Box2i dw = file.dataWindow();
    const int width = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;
    std::vector<Rgba> pixels(static_cast<size_t>(width) * height);
    file.setFrameBuffer(pixels.data() - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);

    py::array_t<float> array({height, width, 3});
    auto buf = array.request();
    float* dst = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < pixels.size(); ++i) {
        const Rgba& p = pixels[i];
        dst[i * 3 + 0] = p.r;
        dst[i * 3 + 1] = p.g;
        dst[i * 3 + 2] = p.b;
    }
    return array;
}
#endif

#if NEUROVA_HAS_JP2
struct StreamCloser {
    void operator()(opj_stream_t* stream) const {
        if (stream) {
            opj_stream_destroy(stream);
        }
    }
};

struct CodecCloser {
    void operator()(opj_codec_t* codec) const {
        if (codec) {
            opj_destroy_codec(codec);
        }
    }
};

py::array read_jp2(const std::string& path) {
    std::unique_ptr<opj_codec_t, CodecCloser> codec(opj_create_decompress(OPJ_CODEC_JP2));
    if (!codec) {
        throw std::runtime_error("Failed to create OpenJPEG codec");
    }
    opj_dparameters_t params;
    opj_set_default_decoder_parameters(&params);
    if (!opj_setup_decoder(codec.get(), &params)) {
        throw std::runtime_error("Failed to setup JP2 decoder");
    }
    std::unique_ptr<opj_stream_t, StreamCloser> stream(opj_stream_create_default_file_stream(path.c_str(), 1));
    if (!stream) {
        throw std::runtime_error("Failed to open JP2 stream");
    }
    opj_image_t* image = nullptr;
    if (!opj_read_header(stream.get(), codec.get(), &image)) {
        throw std::runtime_error("Failed to read JP2 header");
    }
    if (!opj_decode(codec.get(), stream.get(), image)) {
        opj_image_destroy(image);
        throw std::runtime_error("Failed to decode JP2");
    }
    if (!opj_end_decompress(codec.get(), stream.get())) {
        opj_image_destroy(image);
        throw std::runtime_error("Failed to finalize JP2 decode");
    }

    const int width = image->x1 - image->x0;
    const int height = image->y1 - image->y0;
    const int components = std::min(3, static_cast<int>(image->numcomps));
    py::array_t<uint16_t> array({height, width, components});
    auto buf = array.request();
    uint16_t* dst = static_cast<uint16_t*>(buf.ptr);
    for (int c = 0; c < components; ++c) {
        const opj_image_comp_t& comp = image->comps[c];
        for (int i = 0; i < width * height; ++i) {
            dst[i * components + c] = static_cast<uint16_t>(comp.data[i]);
        }
    }
    opj_image_destroy(image);
    return array;
}
#endif

py::array dispatch_read(const std::string& path) {
    const auto ext_pos = path.find_last_of('.') ;
    if (ext_pos == std::string::npos) {
        throw std::runtime_error("File has no extension");
    }
    const std::string ext = to_lower(path.substr(ext_pos + 1));

    if ((ext == "tiff" || ext == "tif") && NEUROVA_HAS_TIFF) {
#if NEUROVA_HAS_TIFF
        return read_tiff(path);
#endif
    } else if ((ext == "exr") && NEUROVA_HAS_OPENEXR) {
#if NEUROVA_HAS_OPENEXR
        return read_exr(path);
#endif
    } else if ((ext == "jp2" || ext == "j2k" || ext == "jpf") && NEUROVA_HAS_JP2) {
#if NEUROVA_HAS_JP2
        return read_jp2(path);
#endif
    }

    throw std::runtime_error("Requested codec not built");
}

py::dict capabilities() {
    py::dict info;
    info["tiff"] = py::bool_(NEUROVA_HAS_TIFF);
    info["exr"] = py::bool_(NEUROVA_HAS_OPENEXR);
    info["jp2"] = py::bool_(NEUROVA_HAS_JP2);
    return info;
}

}  // namespace

PYBIND11_MODULE(codecs_extra, m) {
    m.doc() = "Extended image codecs (TIFF/EXR/JPEG2000)";
    m.def("imread", &dispatch_read, "Read extended image formats");
    m.def("capabilities", &capabilities, "Return codec availability");
}
