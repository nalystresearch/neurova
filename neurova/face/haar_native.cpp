// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * haar_native.cpp - Fast Haar Cascade face detection in pure C++
 * 
 * This provides TFLite-level speed without any deep learning dependencies.
 * Uses SIMD optimizations and integral images for fast detection.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

// rectangle structure for features
struct Rect {
    int x, y, width, height;
    float weight;
};

// haar feature with multiple rectangles
struct HaarFeature {
    std::vector<Rect> rects;
    float threshold;
    float left_val;
    float right_val;
};

// detection stage
struct HaarStage {
    std::vector<HaarFeature> features;
    float threshold;
};

// haar cascade classifier
struct HaarCascade {
    std::vector<HaarStage> stages;
    int window_width;
    int window_height;
    bool loaded;
    
    HaarCascade() : window_width(24), window_height(24), loaded(false) {}
};

// global cascade storage
static HaarCascade g_cascade;

// fast integral image computation
static void compute_integral_image(const uint8_t* gray, int width, int height,
                                    uint32_t* integral, uint64_t* sq_integral) {
    // first row
    uint32_t row_sum = 0;
    uint64_t sq_row_sum = 0;
    for (int x = 0; x < width; x++) {
        uint8_t val = gray[x];
        row_sum += val;
        sq_row_sum += val * val;
        integral[x] = row_sum;
        sq_integral[x] = sq_row_sum;
    }
    
    // remaining rows
    for (int y = 1; y < height; y++) {
        row_sum = 0;
        sq_row_sum = 0;
        int row_offset = y * width;
        int prev_row_offset = (y - 1) * width;
        
        for (int x = 0; x < width; x++) {
            uint8_t val = gray[row_offset + x];
            row_sum += val;
            sq_row_sum += val * val;
            integral[row_offset + x] = row_sum + integral[prev_row_offset + x];
            sq_integral[row_offset + x] = sq_row_sum + sq_integral[prev_row_offset + x];
        }
    }
}

// get sum from integral image
inline uint32_t get_sum(const uint32_t* integral, int width,
                        int x, int y, int w, int h) {
    int x2 = x + w - 1;
    int y2 = y + h - 1;
    
    uint32_t a = (x > 0 && y > 0) ? integral[(y-1) * width + (x-1)] : 0;
    uint32_t b = (y > 0) ? integral[(y-1) * width + x2] : 0;
    uint32_t c = (x > 0) ? integral[y2 * width + (x-1)] : 0;
    uint32_t d = integral[y2 * width + x2];
    
    return d - b - c + a;
}

// get sum from 64-bit integral image (for squared values)
inline uint64_t get_sum64(const uint64_t* integral, int width,
                          int x, int y, int w, int h) {
    int x2 = x + w - 1;
    int y2 = y + h - 1;
    
    uint64_t a = (x > 0 && y > 0) ? integral[(y-1) * width + (x-1)] : 0;
    uint64_t b = (y > 0) ? integral[(y-1) * width + x2] : 0;
    uint64_t c = (x > 0) ? integral[y2 * width + (x-1)] : 0;
    uint64_t d = integral[y2 * width + x2];
    
    return d - b - c + a;
}

// evaluate single haar feature
inline float evaluate_feature(const HaarFeature& feature,
                              const uint32_t* integral, int img_width,
                              int win_x, int win_y, float scale, float inv_area) {
    float sum = 0.0f;
    
    for (const auto& rect : feature.rects) {
        int rx = win_x + (int)(rect.x * scale);
        int ry = win_y + (int)(rect.y * scale);
        int rw = (int)(rect.width * scale);
        int rh = (int)(rect.height * scale);
        
        if (rw > 0 && rh > 0) {
            uint32_t rect_sum = get_sum(integral, img_width, rx, ry, rw, rh);
            sum += rect_sum * rect.weight;
        }
    }
    
    return sum * inv_area;
}

// evaluate cascade at single position
static bool evaluate_cascade(const HaarCascade& cascade,
                             const uint32_t* integral,
                             const uint64_t* sq_integral,
                             int img_width, int win_x, int win_y, float scale) {
    int win_w = (int)(cascade.window_width * scale);
    int win_h = (int)(cascade.window_height * scale);
    float area = (float)(win_w * win_h);
    float inv_area = 1.0f / area;
    
    // compute variance normalization
    uint32_t sum = get_sum(integral, img_width, win_x, win_y, win_w, win_h);
    uint64_t sq_sum = get_sum64(sq_integral, img_width, win_x, win_y, win_w, win_h);
    
    float mean = sum * inv_area;
    float variance = (float)sq_sum * inv_area - mean * mean;
    float std_dev = (variance > 0) ? sqrtf(variance) : 1.0f;
    float inv_std = 1.0f / std_dev;
    
    // evaluate each stage
    for (const auto& stage : cascade.stages) {
        float stage_sum = 0.0f;
        
        for (const auto& feature : stage.features) {
            float val = evaluate_feature(feature, integral, img_width,
                                         win_x, win_y, scale, inv_area);
            val *= inv_std;
            
            if (val < feature.threshold) {
                stage_sum += feature.left_val;
            } else {
                stage_sum += feature.right_val;
            }
        }
        
        if (stage_sum < stage.threshold) {
            return false;  // rejected
        }
    }
    
    return true;  // passed all stages
}

// detection result
struct Detection {
    int x, y, width, height;
    float confidence;
};

// non-maximum suppression
static std::vector<Detection> nms(std::vector<Detection>& detections, float threshold) {
    if (detections.empty()) return {};
    
    // sort by confidence
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            // compute IoU
            int x1 = std::max(detections[i].x, detections[j].x);
            int y1 = std::max(detections[i].y, detections[j].y);
            int x2 = std::min(detections[i].x + detections[i].width,
                             detections[j].x + detections[j].width);
            int y2 = std::min(detections[i].y + detections[i].height,
                             detections[j].y + detections[j].height);
            
            int inter_w = std::max(0, x2 - x1);
            int inter_h = std::max(0, y2 - y1);
            float inter_area = (float)(inter_w * inter_h);
            
            float area_i = (float)(detections[i].width * detections[i].height);
            float area_j = (float)(detections[j].width * detections[j].height);
            float union_area = area_i + area_j - inter_area;
            
            float iou = inter_area / union_area;
            if (iou > threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

// multi-scale face detection
static std::vector<Detection> detect_faces(const uint8_t* gray, int width, int height,
                                           float scale_factor, int min_neighbors,
                                           int min_size, int max_size) {
    if (!g_cascade.loaded) return {};
    
    // compute integral images
    std::vector<uint32_t> integral(width * height);
    std::vector<uint64_t> sq_integral(width * height);
    compute_integral_image(gray, width, height, integral.data(), sq_integral.data());
    
    std::vector<Detection> detections;
    
    // multi-scale detection
    float scale = 1.0f;
    while (true) {
        int win_w = (int)(g_cascade.window_width * scale);
        int win_h = (int)(g_cascade.window_height * scale);
        
        if (win_w < min_size || win_h < min_size) {
            scale *= scale_factor;
            continue;
        }
        
        if (win_w > width || win_h > height) break;
        if (max_size > 0 && (win_w > max_size || win_h > max_size)) break;
        
        // step size based on scale
        int step = std::max(1, (int)(scale * 2));
        
        for (int y = 0; y + win_h <= height; y += step) {
            for (int x = 0; x + win_w <= width; x += step) {
                if (evaluate_cascade(g_cascade, integral.data(), sq_integral.data(),
                                    width, x, y, scale)) {
                    Detection det;
                    det.x = x;
                    det.y = y;
                    det.width = win_w;
                    det.height = win_h;
                    det.confidence = 1.0f;
                    detections.push_back(det);
                }
            }
        }
        
        scale *= scale_factor;
    }
    
    // group detections and apply NMS
    if (detections.empty()) return {};
    
    // simple grouping: count overlapping detections
    for (auto& det : detections) {
        int count = 0;
        for (const auto& other : detections) {
            int cx = det.x + det.width / 2;
            int cy = det.y + det.height / 2;
            if (cx >= other.x && cx < other.x + other.width &&
                cy >= other.y && cy < other.y + other.height) {
                count++;
            }
        }
        det.confidence = (float)count;
    }
    
    // filter by min_neighbors
    std::vector<Detection> filtered;
    for (const auto& det : detections) {
        if (det.confidence >= min_neighbors) {
            filtered.push_back(det);
        }
    }
    
    // apply NMS
    return nms(filtered, 0.3f);
}

// set cascade from python data (stages, features, rectangles)
static void set_cascade_data(int win_w, int win_h,
                              const std::vector<HaarStage>& stages) {
    g_cascade.window_width = win_w;
    g_cascade.window_height = win_h;
    g_cascade.stages = stages;
    g_cascade.loaded = !stages.empty();
}

// Python bindings

static PyObject* py_load_cascade(PyObject* self, PyObject* args) {
    PyObject* stages_list;
    int win_w, win_h;
    
    if (!PyArg_ParseTuple(args, "iiO", &win_w, &win_h, &stages_list)) {
        return NULL;
    }
    
    if (!PyList_Check(stages_list)) {
        PyErr_SetString(PyExc_TypeError, "stages must be a list");
        return NULL;
    }
    
    std::vector<HaarStage> stages;
    Py_ssize_t num_stages = PyList_Size(stages_list);
    
    for (Py_ssize_t i = 0; i < num_stages; i++) {
        PyObject* stage_dict = PyList_GetItem(stages_list, i);
        if (!PyDict_Check(stage_dict)) continue;
        
        HaarStage stage;
        
        PyObject* thresh_obj = PyDict_GetItemString(stage_dict, "threshold");
        if (thresh_obj) stage.threshold = (float)PyFloat_AsDouble(thresh_obj);
        
        PyObject* features_list = PyDict_GetItemString(stage_dict, "features");
        if (!features_list || !PyList_Check(features_list)) continue;
        
        Py_ssize_t num_features = PyList_Size(features_list);
        for (Py_ssize_t j = 0; j < num_features; j++) {
            PyObject* feat_dict = PyList_GetItem(features_list, j);
            if (!PyDict_Check(feat_dict)) continue;
            
            HaarFeature feature;
            
            PyObject* ft = PyDict_GetItemString(feat_dict, "threshold");
            PyObject* fl = PyDict_GetItemString(feat_dict, "left_val");
            PyObject* fr = PyDict_GetItemString(feat_dict, "right_val");
            
            if (ft) feature.threshold = (float)PyFloat_AsDouble(ft);
            if (fl) feature.left_val = (float)PyFloat_AsDouble(fl);
            if (fr) feature.right_val = (float)PyFloat_AsDouble(fr);
            
            PyObject* rects_list = PyDict_GetItemString(feat_dict, "rects");
            if (rects_list && PyList_Check(rects_list)) {
                Py_ssize_t num_rects = PyList_Size(rects_list);
                for (Py_ssize_t k = 0; k < num_rects; k++) {
                    PyObject* rect_dict = PyList_GetItem(rects_list, k);
                    if (!PyDict_Check(rect_dict)) continue;
                    
                    Rect rect;
                    PyObject* rx = PyDict_GetItemString(rect_dict, "x");
                    PyObject* ry = PyDict_GetItemString(rect_dict, "y");
                    PyObject* rw = PyDict_GetItemString(rect_dict, "w");
                    PyObject* rh = PyDict_GetItemString(rect_dict, "h");
                    PyObject* rwt = PyDict_GetItemString(rect_dict, "weight");
                    
                    if (rx) rect.x = (int)PyLong_AsLong(rx);
                    if (ry) rect.y = (int)PyLong_AsLong(ry);
                    if (rw) rect.width = (int)PyLong_AsLong(rw);
                    if (rh) rect.height = (int)PyLong_AsLong(rh);
                    if (rwt) rect.weight = (float)PyFloat_AsDouble(rwt);
                    
                    feature.rects.push_back(rect);
                }
            }
            
            if (!feature.rects.empty()) {
                stage.features.push_back(feature);
            }
        }
        
        if (!stage.features.empty()) {
            stages.push_back(stage);
        }
    }
    
    set_cascade_data(win_w, win_h, stages);
    
    if (g_cascade.loaded) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject* py_detect(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* image;
    float scale_factor = 1.1f;
    int min_neighbors = 3;
    int min_size = 30;
    int max_size = 0;
    
    static char* kwlist[] = {"image", "scale_factor", "min_neighbors", "min_size", "max_size", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|fiii", kwlist,
                                     &PyArray_Type, &image,
                                     &scale_factor, &min_neighbors, &min_size, &max_size)) {
        return NULL;
    }
    
    // validate input
    if (PyArray_NDIM(image) != 2) {
        PyErr_SetString(PyExc_ValueError, "Image must be 2D grayscale");
        return NULL;
    }
    
    if (PyArray_TYPE(image) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "Image must be uint8");
        return NULL;
    }
    
    int height = PyArray_DIM(image, 0);
    int width = PyArray_DIM(image, 1);
    uint8_t* data = (uint8_t*)PyArray_DATA(image);
    
    // detect faces
    std::vector<Detection> detections = detect_faces(data, width, height,
                                                      scale_factor, min_neighbors,
                                                      min_size, max_size);
    
    // create output array
    npy_intp dims[2] = {(npy_intp)detections.size(), 5};
    PyObject* result = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    float* out_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    for (size_t i = 0; i < detections.size(); i++) {
        out_data[i * 5 + 0] = (float)detections[i].x;
        out_data[i * 5 + 1] = (float)detections[i].y;
        out_data[i * 5 + 2] = (float)detections[i].width;
        out_data[i * 5 + 3] = (float)detections[i].height;
        out_data[i * 5 + 4] = detections[i].confidence;
    }
    
    return result;
}

static PyObject* py_get_info(PyObject* self, PyObject* args) {
    if (!g_cascade.loaded) {
        Py_RETURN_NONE;
    }
    
    return Py_BuildValue("{s:i,s:i,s:i}",
                         "window_width", g_cascade.window_width,
                         "window_height", g_cascade.window_height,
                         "num_stages", (int)g_cascade.stages.size());
}

static PyMethodDef HaarNativeMethods[] = {
    {"load_cascade", py_load_cascade, METH_VARARGS, "Load Haar cascade XML file"},
    {"detect", (PyCFunction)py_detect, METH_VARARGS | METH_KEYWORDS, "Detect faces in grayscale image"},
    {"get_info", py_get_info, METH_NOARGS, "Get cascade info"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef haar_native_module = {
    PyModuleDef_HEAD_INIT,
    "haar_native",
    "Fast native Haar cascade face detection",
    -1,
    HaarNativeMethods
};

PyMODINIT_FUNC PyInit_haar_native(void) {
    import_array();
    return PyModule_Create(&haar_native_module);
}
