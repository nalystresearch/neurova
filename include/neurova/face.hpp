// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/face.hpp - Face Detection and Recognition
 * 
 * This header provides face-related functionality:
 * - Face detection (Haar cascade, HOG)
 * - Facial landmark detection
 * - Face recognition/verification
 * - Face alignment
 */

#ifndef NEUROVA_FACE_HPP
#define NEUROVA_FACE_HPP

#include "core.hpp"
#include "imgproc.hpp"
#include <string>

namespace neurova {
namespace face {

// ============================================================================
// Structures
// ============================================================================

struct BoundingBox {
    int x, y, width, height;
    double confidence;
    
    BoundingBox() : x(0), y(0), width(0), height(0), confidence(0.0) {}
    BoundingBox(int x_, int y_, int w_, int h_, double conf_ = 1.0)
        : x(x_), y(y_), width(w_), height(h_), confidence(conf_) {}
    
    int center_x() const { return x + width / 2; }
    int center_y() const { return y + height / 2; }
    int right() const { return x + width; }
    int bottom() const { return y + height; }
    int area() const { return width * height; }
    
    double iou(const BoundingBox& other) const;
};

struct Landmark {
    double x, y;
    double confidence;
    std::string name;
    
    Landmark() : x(0), y(0), confidence(0.0) {}
    Landmark(double x_, double y_, double conf_ = 1.0)
        : x(x_), y(y_), confidence(conf_) {}
};

struct FaceDetection {
    BoundingBox bbox;
    std::vector<Landmark> landmarks;
    double confidence;
    
    FaceDetection() : confidence(0.0) {}
};

struct FaceEmbedding {
    Tensor embedding;
    double quality_score;
    
    double cosine_similarity(const FaceEmbedding& other) const;
    double euclidean_distance(const FaceEmbedding& other) const;
};

// ============================================================================
// Haar Cascade Detector
// ============================================================================

struct HaarFeature {
    int type;
    int x, y, width, height;
    double threshold;
    double left_val, right_val;
};

struct HaarStage {
    double threshold;
    std::vector<HaarFeature> features;
};

class HaarCascade {
public:
    HaarCascade();
    ~HaarCascade();
    
    // Load cascade from XML file
    bool load(const std::string& filename);
    
    // Load cascade from memory
    bool load(const std::vector<HaarStage>& stages, int window_width, int window_height);
    
    // Detect faces
    std::vector<BoundingBox> detectMultiScale(
        const Image& image,
        double scale_factor = 1.1,
        int min_neighbors = 3,
        int min_size = 30,
        int max_size = 0
    ) const;
    
    // Check if loaded
    bool empty() const { return stages_.empty(); }
    
    // Window size
    int window_width() const { return window_width_; }
    int window_height() const { return window_height_; }

private:
    std::vector<HaarStage> stages_;
    int window_width_;
    int window_height_;
    
    // Internal detection
    bool evaluateCascade(const Tensor& integral, const Tensor& sq_integral,
                         int x, int y, double scale, double inv_area) const;
    std::vector<BoundingBox> detectSingleScale(const Image& gray, double scale,
                                               int min_neighbors) const;
    std::vector<BoundingBox> groupRectangles(std::vector<BoundingBox>& rects,
                                             int min_neighbors, double eps = 0.2) const;
};

// Load built-in cascade
HaarCascade loadFrontalFaceCascade();
HaarCascade loadProfileFaceCascade();
HaarCascade loadEyeCascade();

// ============================================================================
// HOG Face Detector
// ============================================================================

class HOGDescriptor {
public:
    HOGDescriptor(int win_width = 64, int win_height = 128,
                  int block_size = 16, int block_stride = 8,
                  int cell_size = 8, int nbins = 9);
    
    // Compute HOG features for an image
    Tensor compute(const Image& image) const;
    
    // Compute HOG features for multiple windows
    std::vector<Tensor> compute(const Image& image, 
                                const std::vector<BoundingBox>& locations) const;
    
    // Descriptor size
    size_t getDescriptorSize() const;

private:
    int win_width_, win_height_;
    int block_size_, block_stride_;
    int cell_size_, nbins_;
    
    Tensor computeGradients(const Image& image) const;
    Tensor computeHistogram(const Tensor& magnitude, const Tensor& orientation,
                           int start_x, int start_y) const;
};

class HOGFaceDetector {
public:
    HOGFaceDetector();
    
    // Set SVM weights (linear classifier)
    void setSVMDetector(const Tensor& weights);
    
    // Detect faces
    std::vector<BoundingBox> detectMultiScale(
        const Image& image,
        double hit_threshold = 0.0,
        double scale_factor = 1.05,
        int min_size = 64,
        int max_size = 0,
        double group_threshold = 2.0
    ) const;

private:
    HOGDescriptor hog_;
    Tensor svm_weights_;
    double svm_bias_;
};

// ============================================================================
// Face Landmark Detector
// ============================================================================

// 5-point landmarks (eyes, nose)
struct FiveLandmarks {
    Landmark left_eye;
    Landmark right_eye;
    Landmark nose;
    Landmark left_mouth;
    Landmark right_mouth;
};

// 68-point landmarks (full face)
struct FullLandmarks {
    std::vector<Landmark> jaw;           // 0-16
    std::vector<Landmark> right_eyebrow; // 17-21
    std::vector<Landmark> left_eyebrow;  // 22-26
    std::vector<Landmark> nose;          // 27-35
    std::vector<Landmark> right_eye;     // 36-41
    std::vector<Landmark> left_eye;      // 42-47
    std::vector<Landmark> outer_lip;     // 48-59
    std::vector<Landmark> inner_lip;     // 60-67
    
    std::vector<Landmark> all() const;
};

class LandmarkDetector {
public:
    LandmarkDetector();
    
    // Load model
    bool load(const std::string& model_path);
    
    // Detect 5-point landmarks
    FiveLandmarks detect5(const Image& image, const BoundingBox& face) const;
    
    // Detect 68-point landmarks
    FullLandmarks detect68(const Image& image, const BoundingBox& face) const;
    
    // Batch detection
    std::vector<FiveLandmarks> detect5Batch(const Image& image,
                                            const std::vector<BoundingBox>& faces) const;
    std::vector<FullLandmarks> detect68Batch(const Image& image,
                                             const std::vector<BoundingBox>& faces) const;

private:
    Tensor model_weights_;
    bool loaded_;
    
    Tensor extractFaceRegion(const Image& image, const BoundingBox& face) const;
    Tensor runModel(const Tensor& face_tensor) const;
};

// ============================================================================
// Face Alignment
// ============================================================================

class FaceAligner {
public:
    FaceAligner(int output_size = 112, double left_eye_ratio_x = 0.35,
                double left_eye_ratio_y = 0.35);
    
    // Align face using eye positions
    Image align(const Image& image, const Landmark& left_eye, 
                const Landmark& right_eye) const;
    
    // Align using 5-point landmarks
    Image align(const Image& image, const FiveLandmarks& landmarks) const;
    
    // Align using 68-point landmarks
    Image align(const Image& image, const FullLandmarks& landmarks) const;
    
    // Align using detection result
    Image align(const Image& image, const FaceDetection& detection) const;
    
    // Get transformation matrix
    Tensor getTransformMatrix(const Landmark& left_eye, 
                              const Landmark& right_eye) const;

private:
    int output_size_;
    double left_eye_ratio_x_;
    double left_eye_ratio_y_;
    
    std::pair<double, double> computeDesiredEyePositions() const;
};

// ============================================================================
// Face Recognition/Embedding
// ============================================================================

class FaceEmbedder {
public:
    FaceEmbedder();
    
    // Load model
    bool load(const std::string& model_path);
    
    // Get embedding for aligned face
    FaceEmbedding embed(const Image& aligned_face) const;
    
    // Batch embedding
    std::vector<FaceEmbedding> embedBatch(const std::vector<Image>& aligned_faces) const;
    
    // Embedding dimension
    size_t embeddingDim() const { return embedding_dim_; }

private:
    Tensor model_weights_;
    size_t embedding_dim_;
    bool loaded_;
};

class FaceVerifier {
public:
    FaceVerifier(double threshold = 0.5);
    
    // Verify if two faces are the same person
    bool verify(const FaceEmbedding& emb1, const FaceEmbedding& emb2) const;
    
    // Get similarity score
    double similarity(const FaceEmbedding& emb1, const FaceEmbedding& emb2) const;
    
    // Set threshold
    void setThreshold(double threshold) { threshold_ = threshold; }
    double getThreshold() const { return threshold_; }

private:
    double threshold_;
};

class FaceIdentifier {
public:
    FaceIdentifier(double threshold = 0.5);
    
    // Add identity to database
    void addIdentity(const std::string& name, const FaceEmbedding& embedding);
    void addIdentity(const std::string& name, const std::vector<FaceEmbedding>& embeddings);
    
    // Remove identity
    void removeIdentity(const std::string& name);
    
    // Clear database
    void clear();
    
    // Identify face
    std::pair<std::string, double> identify(const FaceEmbedding& embedding) const;
    
    // Get top-k matches
    std::vector<std::pair<std::string, double>> identifyTopK(
        const FaceEmbedding& embedding, int k = 5) const;
    
    // Database size
    size_t numIdentities() const { return database_.size(); }
    
    // Save/load database
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);

private:
    std::map<std::string, std::vector<FaceEmbedding>> database_;
    double threshold_;
};

// ============================================================================
// Complete Face Pipeline
// ============================================================================

class FaceDetector {
public:
    enum class Method {
        HAAR,
        HOG
    };
    
    FaceDetector(Method method = Method::HAAR);
    
    // Load model (if needed)
    bool load(const std::string& model_path = "");
    
    // Detect faces
    std::vector<FaceDetection> detect(
        const Image& image,
        double scale_factor = 1.1,
        int min_neighbors = 3,
        int min_size = 30,
        int max_size = 0,
        bool detect_landmarks = false
    ) const;
    
    // Set method
    void setMethod(Method method);
    Method getMethod() const { return method_; }

private:
    Method method_;
    HaarCascade haar_;
    HOGFaceDetector hog_;
    LandmarkDetector landmark_detector_;
    bool detect_landmarks_;
};

class FaceRecognitionPipeline {
public:
    FaceRecognitionPipeline();
    
    // Initialize all components
    bool initialize(const std::string& detection_model = "",
                    const std::string& landmark_model = "",
                    const std::string& embedding_model = "");
    
    // Process image - detect, align, embed
    std::vector<std::pair<FaceDetection, FaceEmbedding>> process(const Image& image) const;
    
    // Identify faces in image
    std::vector<std::tuple<FaceDetection, std::string, double>> identify(
        const Image& image) const;
    
    // Add identity from image
    bool enrollIdentity(const std::string& name, const Image& image);
    bool enrollIdentity(const std::string& name, const std::vector<Image>& images);
    
    // Components access
    FaceDetector& detector() { return detector_; }
    FaceAligner& aligner() { return aligner_; }
    FaceEmbedder& embedder() { return embedder_; }
    FaceIdentifier& identifier() { return identifier_; }

private:
    FaceDetector detector_;
    LandmarkDetector landmark_detector_;
    FaceAligner aligner_;
    FaceEmbedder embedder_;
    FaceIdentifier identifier_;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Draw detection results
void drawFaceDetection(Image& image, const FaceDetection& detection,
                       const std::vector<uint8_t>& color = {0, 255, 0},
                       int thickness = 2, bool draw_landmarks = true);

void drawFaceDetections(Image& image, const std::vector<FaceDetection>& detections,
                        const std::vector<uint8_t>& color = {0, 255, 0},
                        int thickness = 2, bool draw_landmarks = true);

// Draw bounding box
void drawBoundingBox(Image& image, const BoundingBox& bbox,
                     const std::vector<uint8_t>& color = {0, 255, 0},
                     int thickness = 2);

// Draw landmarks
void drawLandmarks(Image& image, const std::vector<Landmark>& landmarks,
                   const std::vector<uint8_t>& color = {0, 0, 255},
                   int radius = 2);

void drawLandmarks(Image& image, const FiveLandmarks& landmarks,
                   const std::vector<uint8_t>& color = {0, 0, 255},
                   int radius = 2);

void drawLandmarks(Image& image, const FullLandmarks& landmarks,
                   const std::vector<uint8_t>& color = {0, 0, 255},
                   int radius = 2, bool connect_points = true);

// Non-maximum suppression
std::vector<BoundingBox> nms(std::vector<BoundingBox>& boxes, 
                             double iou_threshold = 0.5);

// Crop face from image
Image cropFace(const Image& image, const BoundingBox& bbox, double padding = 0.0);

// Face quality assessment
double assessFaceQuality(const Image& face_image);
bool isFrontal(const FiveLandmarks& landmarks, double threshold = 0.1);
bool isBlurry(const Image& image, double threshold = 100.0);

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_HPP
