# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Build script for Neurova native extensions.

This script is intentionally separate from the main Neurova package build so the
core install stays lightweight.

Modules built here:
- neurova.video.camera_native / neurova.video.display_native (platform APIs)
- neurova.io.codecs_native (optional; libpng + libjpeg)

Environment flags:
- NEUROVA_BUILD_CODECS=1 enables codecs_native build
- NEUROVA_PNG_INCLUDE / NEUROVA_PNG_LIB
- NEUROVA_JPEG_INCLUDE / NEUROVA_JPEG_LIB
"""

from __future__ import annotations

import os
import platform

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def build_extensions(self):
        if self.compiler is not None and hasattr(self.compiler, "src_extensions"):
            if ".mm" not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(".mm")
        super().build_extensions()


def _env_path_list(name: str) -> list[str]:
    val = os.environ.get(name, "").strip()
    if not val:
        return []
    return [p for p in val.split(os.pathsep) if p]


def _common_includes() -> list[str]:
    return [pybind11.get_include(), pybind11.get_include(user=True)]


system = platform.system()

ext_modules: list[Extension] = []

# Camera + display
if system == "Darwin":
    cam_src = ["neurova/video/camera_native_avf.mm"]
    cam_compile = ["-std=c++17", "-O3", "-stdlib=libc++", "-fobjc-arc"]
    cam_link = [
        "-framework",
        "AVFoundation",
        "-framework",
        "CoreMedia",
        "-framework",
        "CoreVideo",
        "-framework",
        "Foundation",
    ]
    cam_libs: list[str] = []

    disp_src = ["neurova/video/display_native.mm"]
    disp_compile = ["-std=c++17", "-O3", "-stdlib=libc++", "-fobjc-arc"]
    disp_link = ["-framework", "Cocoa", "-framework", "Foundation"]
    disp_libs: list[str] = []
elif system == "Windows":
    cam_src = ["neurova/video/camera_native_mf.cpp"]
    cam_compile = ["/std:c++17", "/O2"]
    cam_link = []
    cam_libs = ["mfplat", "mf", "mfreadwrite", "mfuuid", "d3d11", "ole32", "oleaut32"]

    disp_src = ["neurova/video/display_native_win.cpp"]
    disp_compile = ["/std:c++17", "/O2"]
    disp_link = []
    disp_libs = ["user32", "gdi32"]
elif system == "Linux":
    cam_src = ["neurova/video/camera_native_v4l2.cpp"]
    cam_compile = ["-std=c++17", "-O3"]
    cam_link = []
    cam_libs = ["v4l2"]

    disp_src = ["neurova/video/display_native_x11.cpp"]
    disp_compile = ["-std=c++17", "-O3"]
    disp_link = []
    disp_libs = ["X11"]
else:
    cam_src = ["neurova/video/camera_native_stub.cpp"]
    cam_compile, cam_link, cam_libs = ["-std=c++17", "-O3"], [], []
    disp_src = ["neurova/video/display_native_stub.cpp"]
    disp_compile, disp_link, disp_libs = ["-std=c++17", "-O3"], [], []

ext_modules.append(
    Extension(
        "neurova.video.camera_native",
        sources=cam_src,
        include_dirs=_common_includes(),
        language="c++",
        extra_compile_args=cam_compile,
        extra_link_args=cam_link,
        libraries=cam_libs,
    )
)

ext_modules.append(
    Extension(
        "neurova.video.display_native",
        sources=disp_src,
        include_dirs=_common_includes(),
        language="c++",
        extra_compile_args=disp_compile,
        extra_link_args=disp_link,
        libraries=disp_libs,
    )
)

# Optional codecs
if os.environ.get("NEUROVA_BUILD_CODECS", "0").strip() == "1":
    codec_compile: list[str]
    codec_link: list[str] = []
    if system == "Windows":
        codec_compile = ["/std:c++17", "/O2"]
    else:
        codec_compile = ["-std=c++17", "-O3"] + (["-stdlib=libc++"] if system == "Darwin" else [])

    include_dirs = _common_includes() + _env_path_list("NEUROVA_PNG_INCLUDE") + _env_path_list(
        "NEUROVA_JPEG_INCLUDE"
    )
    library_dirs = _env_path_list("NEUROVA_PNG_LIB") + _env_path_list("NEUROVA_JPEG_LIB")
    libraries = ["png", "jpeg"]

    ext_modules.append(
        Extension(
            "neurova.io.codecs_native",
            sources=["neurova/io/codecs_native.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++",
            extra_compile_args=codec_compile,
            extra_link_args=codec_link,
        )
    )

# Optional libav backend
if os.environ.get("NEUROVA_BUILD_LIBAV", "0").strip() == "1":
    libav_compile: list[str]
    libav_link: list[str] = []
    if system == "Windows":
        libav_compile = ["/std:c++17", "/O2"]
    else:
        libav_compile = ["-std=c++17", "-O3"] + (['-stdlib=libc++'] if system == 'Darwin' else [])

    include_dirs = _common_includes() + _env_path_list("NEUROVA_LIBAV_INCLUDE")
    library_dirs = _env_path_list("NEUROVA_LIBAV_LIB")
    libraries = [
        os.environ.get("NEUROVA_LIBAVFORMAT", "avformat"),
        os.environ.get("NEUROVA_LIBAVCODEC", "avcodec"),
        os.environ.get("NEUROVA_LIBAVUTIL", "avutil"),
        os.environ.get("NEUROVA_LIBSWSCALE", "swscale"),
    ]

    ext_modules.append(
        Extension(
            "neurova.video.ffmpeg_backend",
            sources=["neurova/video/ffmpeg_backend.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++",
            extra_compile_args=libav_compile,
            extra_link_args=libav_link,
        )
    )

# Optional extra codecs (TIFF/EXR/JPEG2000)
if os.environ.get("NEUROVA_BUILD_EXTRA_CODECS", "0").strip() == "1":
    extra_compile: list[str]
    if system == "Windows":
        extra_compile = ["/std:c++17", "/O2"]
    else:
        extra_compile = ["-std=c++17", "-O3"] + (['-stdlib=libc++'] if system == 'Darwin' else [])

    include_dirs = _common_includes()
    include_dirs += _env_path_list("NEUROVA_TIFF_INCLUDE")
    include_dirs += _env_path_list("NEUROVA_EXR_INCLUDE")
    include_dirs += _env_path_list("NEUROVA_JP2_INCLUDE")

    library_dirs = []
    library_dirs += _env_path_list("NEUROVA_TIFF_LIB")
    library_dirs += _env_path_list("NEUROVA_EXR_LIB")
    library_dirs += _env_path_list("NEUROVA_JP2_LIB")

    libraries = []
    if os.environ.get("NEUROVA_USE_TIFF", "1") != "0":
        libraries.append(os.environ.get("NEUROVA_TIFF_LIBNAME", "tiff"))
    if os.environ.get("NEUROVA_USE_EXR", "1") != "0":
        libraries.append(os.environ.get("NEUROVA_EXR_LIBNAME", "IlmImf"))
    if os.environ.get("NEUROVA_USE_JP2", "1") != "0":
        libraries.append(os.environ.get("NEUROVA_JP2_LIBNAME", "openjp2"))

    ext_modules.append(
        Extension(
            "neurova.io.codecs_extra",
            sources=["neurova/io/codecs_extra.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++",
            extra_compile_args=extra_compile,
        )
    )

# Optional WebP codec
if os.environ.get("NEUROVA_BUILD_WEBP", "0").strip() == "1":
    if system == "Windows":
        webp_compile = ["/std:c++17", "/O2"]
    else:
        webp_compile = ["-std=c++17", "-O3"] + (["-stdlib=libc++"] if system == "Darwin" else [])

    include_dirs = _common_includes() + _env_path_list("NEUROVA_WEBP_INCLUDE")
    library_dirs = _env_path_list("NEUROVA_WEBP_LIB")
    libraries = [os.environ.get("NEUROVA_WEBP_LIBNAME", "webp"), os.environ.get("NEUROVA_WEBP_DEMUX_LIBNAME", "webpdemux")]

    ext_modules.append(
        Extension(
            "neurova.io.codecs_webp",
            sources=["neurova/io/codecs_webp.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            language="c++",
            extra_compile_args=webp_compile,
        )
    )

# Optional CUDA/NPP acceleration
if os.environ.get("NEUROVA_BUILD_CUDA", "0").strip() == "1":
    cuda_include = _env_path_list("NEUROVA_CUDA_INCLUDE")
    cuda_lib = _env_path_list("NEUROVA_CUDA_LIB")
    cuda_compile = ["-std=c++17", "-O3"]
    cuda_link = []
    libraries = [os.environ.get("NEUROVA_CUDA_RUNTIME", "cudart"), os.environ.get("NEUROVA_NPP_LIB", "nppif")]

    ext_modules.append(
        Extension(
            "neurova.accel.cuda_native",
            sources=["neurova/accel/cuda_native.cpp"],
            include_dirs=_common_includes() + cuda_include,
            library_dirs=cuda_lib,
            libraries=libraries,
            language="c++",
            extra_compile_args=cuda_compile,
            extra_link_args=cuda_link,
        )
    )


setup(
    name="neurova-native",
    version="1.0.0",
    ext_modules=ext_modules,
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)
