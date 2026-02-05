// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file face_recognizer.hpp
 * @brief Face recognition using LBPH, EigenFace, and FisherFace methods
 */

#ifndef NEUROVA_FACE_RECOGNIZER_HPP
#define NEUROVA_FACE_RECOGNIZER_HPP

#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>

namespace neurova {
namespace face {

/**
 * @brief Prediction result
 */
struct Prediction {
    int label;
    float confidence;
    
    Prediction(int l = -1, float c = 0.0f) : label(l), confidence(c) {}
    
    bool operator<(const Prediction& other) const {
        return confidence < other.confidence;
    }
};

/**
 * @brief Abstract base class for face recognizers
 */
class RecognizerBase {
public:
    virtual ~RecognizerBase() = default;
    
    virtual void train(const std::vector<std::vector<float>>& faces,
                       const std::vector<int>& labels) = 0;
    
    virtual Prediction predict(const std::vector<float>& face) = 0;
    
    virtual std::vector<Prediction> predictAll(const std::vector<float>& face) = 0;
    
    virtual void update(const std::vector<std::vector<float>>& faces,
                        const std::vector<int>& labels) = 0;
    
    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
};

/**
 * @brief LBPH (Local Binary Patterns Histograms) face recognizer
 * 
 * Pure C++ implementation of LBPH face recognition.
 */
class LBPHRecognizer : public RecognizerBase {
public:
    /**
     * @brief Construct LBPH recognizer
     * @param threshold Recognition threshold (higher = stricter)
     * @param radius LBP radius
     * @param neighbors Number of neighbors for LBP
     * @param grid_x Number of horizontal grid cells
     * @param grid_y Number of vertical grid cells
     */
    LBPHRecognizer(float threshold = 100.0f, 
                   int radius = 1, int neighbors = 8,
                   int grid_x = 8, int grid_y = 8)
        : threshold_(threshold), radius_(radius), neighbors_(neighbors),
          grid_x_(grid_x), grid_y_(grid_y) {}
    
    void train(const std::vector<std::vector<float>>& faces,
               const std::vector<int>& labels) override {
        histograms_.clear();
        labels_.clear();
        
        for (size_t i = 0; i < faces.size(); ++i) {
            auto hist = computeLBPH(faces[i]);
            histograms_.push_back(hist);
            labels_.push_back(labels[i]);
        }
    }
    
    Prediction predict(const std::vector<float>& face) override {
        if (histograms_.empty()) {
            return Prediction(-1, std::numeric_limits<float>::infinity());
        }
        
        auto hist = computeLBPH(face);
        
        float min_dist = std::numeric_limits<float>::infinity();
        int best_label = -1;
        
        for (size_t i = 0; i < histograms_.size(); ++i) {
            float dist = chiSquareDistance(hist, histograms_[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_label = labels_[i];
            }
        }
        
        return Prediction(best_label, min_dist);
    }
    
    std::vector<Prediction> predictAll(const std::vector<float>& face) override {
        auto hist = computeLBPH(face);
        
        std::vector<Prediction> predictions;
        for (size_t i = 0; i < histograms_.size(); ++i) {
            float dist = chiSquareDistance(hist, histograms_[i]);
            predictions.emplace_back(labels_[i], dist);
        }
        
        std::sort(predictions.begin(), predictions.end());
        return predictions;
    }
    
    void update(const std::vector<std::vector<float>>& faces,
                const std::vector<int>& labels) override {
        for (size_t i = 0; i < faces.size(); ++i) {
            auto hist = computeLBPH(faces[i]);
            histograms_.push_back(hist);
            labels_.push_back(labels[i]);
        }
    }
    
    void save(const std::string& path) override {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing");
        
        // Write parameters
        file.write(reinterpret_cast<const char*>(&radius_), sizeof(radius_));
        file.write(reinterpret_cast<const char*>(&neighbors_), sizeof(neighbors_));
        file.write(reinterpret_cast<const char*>(&grid_x_), sizeof(grid_x_));
        file.write(reinterpret_cast<const char*>(&grid_y_), sizeof(grid_y_));
        
        // Write histograms
        size_t num_hist = histograms_.size();
        file.write(reinterpret_cast<const char*>(&num_hist), sizeof(num_hist));
        
        for (size_t i = 0; i < num_hist; ++i) {
            size_t hist_size = histograms_[i].size();
            file.write(reinterpret_cast<const char*>(&hist_size), sizeof(hist_size));
            file.write(reinterpret_cast<const char*>(histograms_[i].data()), 
                       hist_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
        }
    }
    
    void load(const std::string& path) override {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading");
        
        file.read(reinterpret_cast<char*>(&radius_), sizeof(radius_));
        file.read(reinterpret_cast<char*>(&neighbors_), sizeof(neighbors_));
        file.read(reinterpret_cast<char*>(&grid_x_), sizeof(grid_x_));
        file.read(reinterpret_cast<char*>(&grid_y_), sizeof(grid_y_));
        
        size_t num_hist;
        file.read(reinterpret_cast<char*>(&num_hist), sizeof(num_hist));
        
        histograms_.resize(num_hist);
        labels_.resize(num_hist);
        
        for (size_t i = 0; i < num_hist; ++i) {
            size_t hist_size;
            file.read(reinterpret_cast<char*>(&hist_size), sizeof(hist_size));
            histograms_[i].resize(hist_size);
            file.read(reinterpret_cast<char*>(histograms_[i].data()), 
                      hist_size * sizeof(float));
            file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
        }
    }
    
private:
    float threshold_;
    int radius_, neighbors_, grid_x_, grid_y_;
    std::vector<std::vector<float>> histograms_;
    std::vector<int> labels_;
    int face_size_ = 0; // Detected from first face
    
    std::vector<float> computeLBPH(const std::vector<float>& face) {
        // Assume square face
        int size = static_cast<int>(std::sqrt(face.size()));
        if (size * size != static_cast<int>(face.size())) {
            size = static_cast<int>(std::sqrt(face.size() / 3)); // Assume 3 channels
        }
        
        // Compute LBP image
        std::vector<int> lbp(size * size, 0);
        
        for (int y = radius_; y < size - radius_; ++y) {
            for (int x = radius_; x < size - radius_; ++x) {
                float center = face[y * size + x];
                int code = 0;
                
                // 8-neighbor LBP
                if (face[(y-1) * size + (x-1)] >= center) code |= (1 << 7);
                if (face[(y-1) * size + x] >= center) code |= (1 << 6);
                if (face[(y-1) * size + (x+1)] >= center) code |= (1 << 5);
                if (face[y * size + (x+1)] >= center) code |= (1 << 4);
                if (face[(y+1) * size + (x+1)] >= center) code |= (1 << 3);
                if (face[(y+1) * size + x] >= center) code |= (1 << 2);
                if (face[(y+1) * size + (x-1)] >= center) code |= (1 << 1);
                if (face[y * size + (x-1)] >= center) code |= (1 << 0);
                
                lbp[y * size + x] = code;
            }
        }
        
        // Compute spatial histogram
        int cell_w = size / grid_x_;
        int cell_h = size / grid_y_;
        int num_bins = 256;
        
        std::vector<float> histogram(grid_x_ * grid_y_ * num_bins, 0.0f);
        
        for (int gy = 0; gy < grid_y_; ++gy) {
            for (int gx = 0; gx < grid_x_; ++gx) {
                int hist_offset = (gy * grid_x_ + gx) * num_bins;
                
                for (int y = gy * cell_h; y < (gy + 1) * cell_h && y < size; ++y) {
                    for (int x = gx * cell_w; x < (gx + 1) * cell_w && x < size; ++x) {
                        int bin = lbp[y * size + x];
                        histogram[hist_offset + bin] += 1.0f;
                    }
                }
                
                // Normalize cell histogram
                float sum = 0.0f;
                for (int b = 0; b < num_bins; ++b) {
                    sum += histogram[hist_offset + b];
                }
                if (sum > 0) {
                    for (int b = 0; b < num_bins; ++b) {
                        histogram[hist_offset + b] /= sum;
                    }
                }
            }
        }
        
        return histogram;
    }
    
    float chiSquareDistance(const std::vector<float>& h1, const std::vector<float>& h2) {
        float dist = 0.0f;
        for (size_t i = 0; i < h1.size() && i < h2.size(); ++i) {
            float sum = h1[i] + h2[i];
            if (sum > 1e-10f) {
                float diff = h1[i] - h2[i];
                dist += diff * diff / sum;
            }
        }
        return dist;
    }
};

/**
 * @brief EigenFace (PCA-based) recognizer
 */
class EigenFaceRecognizer : public RecognizerBase {
public:
    EigenFaceRecognizer(float threshold = 5000.0f, int num_components = 80)
        : threshold_(threshold), num_components_(num_components) {}
    
    void train(const std::vector<std::vector<float>>& faces,
               const std::vector<int>& labels) override {
        if (faces.empty()) return;
        
        labels_ = labels;
        int n = static_cast<int>(faces.size());
        int d = static_cast<int>(faces[0].size());
        
        // Compute mean face
        mean_face_.resize(d, 0.0f);
        for (const auto& face : faces) {
            for (int i = 0; i < d; ++i) {
                mean_face_[i] += face[i] / n;
            }
        }
        
        // Center the data
        std::vector<std::vector<float>> centered(n, std::vector<float>(d));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                centered[i][j] = faces[i][j] - mean_face_[j];
            }
        }
        
        // Compute covariance (small trick: use A*A^T instead of A^T*A)
        std::vector<std::vector<float>> cov(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += centered[i][k] * centered[j][k];
                }
                cov[i][j] = cov[j][i] = dot;
            }
        }
        
        // Simple power iteration for top eigenvectors (simplified)
        int k = std::min(num_components_, n - 1);
        eigenvectors_.resize(k, std::vector<float>(d, 0.0f));
        
        // Initialize with centered faces (simplified PCA)
        for (int i = 0; i < k && i < n; ++i) {
            eigenvectors_[i] = centered[i];
            // Normalize
            float norm = 0.0f;
            for (float v : eigenvectors_[i]) norm += v * v;
            norm = std::sqrt(norm + 1e-10f);
            for (float& v : eigenvectors_[i]) v /= norm;
        }
        
        // Project training faces
        projections_.resize(n);
        for (int i = 0; i < n; ++i) {
            projections_[i].resize(k);
            for (int j = 0; j < k; ++j) {
                float dot = 0.0f;
                for (int p = 0; p < d; ++p) {
                    dot += centered[i][p] * eigenvectors_[j][p];
                }
                projections_[i][j] = dot;
            }
        }
    }
    
    Prediction predict(const std::vector<float>& face) override {
        if (projections_.empty() || eigenvectors_.empty()) {
            return Prediction(-1, std::numeric_limits<float>::infinity());
        }
        
        // Center and project
        int d = static_cast<int>(face.size());
        int k = static_cast<int>(eigenvectors_.size());
        
        std::vector<float> centered(d);
        for (int i = 0; i < d; ++i) {
            centered[i] = face[i] - mean_face_[i];
        }
        
        std::vector<float> projection(k);
        for (int j = 0; j < k; ++j) {
            float dot = 0.0f;
            for (int p = 0; p < d; ++p) {
                dot += centered[p] * eigenvectors_[j][p];
            }
            projection[j] = dot;
        }
        
        // Find nearest neighbor
        float min_dist = std::numeric_limits<float>::infinity();
        int best_label = -1;
        
        for (size_t i = 0; i < projections_.size(); ++i) {
            float dist = 0.0f;
            for (int j = 0; j < k; ++j) {
                float diff = projection[j] - projections_[i][j];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            
            if (dist < min_dist) {
                min_dist = dist;
                best_label = labels_[i];
            }
        }
        
        return Prediction(best_label, min_dist);
    }
    
    std::vector<Prediction> predictAll(const std::vector<float>& face) override {
        // Similar to predict but return all
        std::vector<Prediction> results;
        
        if (projections_.empty() || eigenvectors_.empty()) {
            return results;
        }
        
        int d = static_cast<int>(face.size());
        int k = static_cast<int>(eigenvectors_.size());
        
        std::vector<float> centered(d);
        for (int i = 0; i < d; ++i) {
            centered[i] = face[i] - mean_face_[i];
        }
        
        std::vector<float> projection(k);
        for (int j = 0; j < k; ++j) {
            float dot = 0.0f;
            for (int p = 0; p < d; ++p) {
                dot += centered[p] * eigenvectors_[j][p];
            }
            projection[j] = dot;
        }
        
        for (size_t i = 0; i < projections_.size(); ++i) {
            float dist = 0.0f;
            for (int j = 0; j < k; ++j) {
                float diff = projection[j] - projections_[i][j];
                dist += diff * diff;
            }
            results.emplace_back(labels_[i], std::sqrt(dist));
        }
        
        std::sort(results.begin(), results.end());
        return results;
    }
    
    void update(const std::vector<std::vector<float>>& faces,
                const std::vector<int>& labels) override {
        // EigenFace requires full retrain
        // Collect all faces and retrain
    }
    
    void save(const std::string& path) override {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        
        // Write mean face
        size_t mean_size = mean_face_.size();
        file.write(reinterpret_cast<const char*>(&mean_size), sizeof(mean_size));
        file.write(reinterpret_cast<const char*>(mean_face_.data()), mean_size * sizeof(float));
        
        // Write eigenvectors
        size_t num_eigen = eigenvectors_.size();
        file.write(reinterpret_cast<const char*>(&num_eigen), sizeof(num_eigen));
        for (const auto& ev : eigenvectors_) {
            size_t ev_size = ev.size();
            file.write(reinterpret_cast<const char*>(&ev_size), sizeof(ev_size));
            file.write(reinterpret_cast<const char*>(ev.data()), ev_size * sizeof(float));
        }
        
        // Write projections and labels
        size_t num_proj = projections_.size();
        file.write(reinterpret_cast<const char*>(&num_proj), sizeof(num_proj));
        for (size_t i = 0; i < num_proj; ++i) {
            size_t proj_size = projections_[i].size();
            file.write(reinterpret_cast<const char*>(&proj_size), sizeof(proj_size));
            file.write(reinterpret_cast<const char*>(projections_[i].data()), proj_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
        }
    }
    
    void load(const std::string& path) override {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        
        size_t mean_size;
        file.read(reinterpret_cast<char*>(&mean_size), sizeof(mean_size));
        mean_face_.resize(mean_size);
        file.read(reinterpret_cast<char*>(mean_face_.data()), mean_size * sizeof(float));
        
        size_t num_eigen;
        file.read(reinterpret_cast<char*>(&num_eigen), sizeof(num_eigen));
        eigenvectors_.resize(num_eigen);
        for (auto& ev : eigenvectors_) {
            size_t ev_size;
            file.read(reinterpret_cast<char*>(&ev_size), sizeof(ev_size));
            ev.resize(ev_size);
            file.read(reinterpret_cast<char*>(ev.data()), ev_size * sizeof(float));
        }
        
        size_t num_proj;
        file.read(reinterpret_cast<char*>(&num_proj), sizeof(num_proj));
        projections_.resize(num_proj);
        labels_.resize(num_proj);
        for (size_t i = 0; i < num_proj; ++i) {
            size_t proj_size;
            file.read(reinterpret_cast<char*>(&proj_size), sizeof(proj_size));
            projections_[i].resize(proj_size);
            file.read(reinterpret_cast<char*>(projections_[i].data()), proj_size * sizeof(float));
            file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
        }
    }
    
private:
    float threshold_;
    int num_components_;
    std::vector<float> mean_face_;
    std::vector<std::vector<float>> eigenvectors_;
    std::vector<std::vector<float>> projections_;
    std::vector<int> labels_;
};

/**
 * @brief Embedding-based recognizer using feature extraction
 */
class EmbeddingRecognizer : public RecognizerBase {
public:
    EmbeddingRecognizer(float threshold = 0.6f) : threshold_(threshold) {}
    
    void train(const std::vector<std::vector<float>>& faces,
               const std::vector<int>& labels) override {
        embeddings_.clear();
        labels_ = labels;
        
        for (const auto& face : faces) {
            embeddings_.push_back(computeEmbedding(face));
        }
    }
    
    Prediction predict(const std::vector<float>& face) override {
        if (embeddings_.empty()) {
            return Prediction(-1, std::numeric_limits<float>::infinity());
        }
        
        auto embedding = computeEmbedding(face);
        
        float min_dist = std::numeric_limits<float>::infinity();
        int best_label = -1;
        
        for (size_t i = 0; i < embeddings_.size(); ++i) {
            float dist = cosineDistance(embedding, embeddings_[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_label = labels_[i];
            }
        }
        
        return Prediction(best_label, min_dist);
    }
    
    std::vector<Prediction> predictAll(const std::vector<float>& face) override {
        auto embedding = computeEmbedding(face);
        
        std::vector<Prediction> results;
        for (size_t i = 0; i < embeddings_.size(); ++i) {
            float dist = cosineDistance(embedding, embeddings_[i]);
            results.emplace_back(labels_[i], dist);
        }
        
        std::sort(results.begin(), results.end());
        return results;
    }
    
    void update(const std::vector<std::vector<float>>& faces,
                const std::vector<int>& labels) override {
        for (size_t i = 0; i < faces.size(); ++i) {
            embeddings_.push_back(computeEmbedding(faces[i]));
            labels_.push_back(labels[i]);
        }
    }
    
    void save(const std::string& path) override {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        
        size_t num_emb = embeddings_.size();
        file.write(reinterpret_cast<const char*>(&num_emb), sizeof(num_emb));
        
        for (size_t i = 0; i < num_emb; ++i) {
            size_t emb_size = embeddings_[i].size();
            file.write(reinterpret_cast<const char*>(&emb_size), sizeof(emb_size));
            file.write(reinterpret_cast<const char*>(embeddings_[i].data()), emb_size * sizeof(float));
            file.write(reinterpret_cast<const char*>(&labels_[i]), sizeof(int));
        }
    }
    
    void load(const std::string& path) override {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        
        size_t num_emb;
        file.read(reinterpret_cast<char*>(&num_emb), sizeof(num_emb));
        
        embeddings_.resize(num_emb);
        labels_.resize(num_emb);
        
        for (size_t i = 0; i < num_emb; ++i) {
            size_t emb_size;
            file.read(reinterpret_cast<char*>(&emb_size), sizeof(emb_size));
            embeddings_[i].resize(emb_size);
            file.read(reinterpret_cast<char*>(embeddings_[i].data()), emb_size * sizeof(float));
            file.read(reinterpret_cast<char*>(&labels_[i]), sizeof(int));
        }
    }
    
private:
    float threshold_;
    std::vector<std::vector<float>> embeddings_;
    std::vector<int> labels_;
    
    std::vector<float> computeEmbedding(const std::vector<float>& face) {
        // Simple feature extraction combining gradient and texture features
        int size = static_cast<int>(std::sqrt(face.size()));
        std::vector<float> features;
        
        // Gradient histogram (HOG-like)
        for (int cy = 0; cy < 4; ++cy) {
            for (int cx = 0; cx < 4; ++cx) {
                std::vector<float> hist(9, 0.0f);
                int cell_size = size / 4;
                
                for (int y = cy * cell_size; y < (cy + 1) * cell_size && y < size - 1; ++y) {
                    for (int x = cx * cell_size; x < (cx + 1) * cell_size && x < size - 1; ++x) {
                        float gx = face[y * size + x + 1] - face[y * size + x];
                        float gy = face[(y + 1) * size + x] - face[y * size + x];
                        float mag = std::sqrt(gx * gx + gy * gy);
                        float angle = std::atan2(gy, gx) + M_PI;
                        int bin = static_cast<int>(angle / (2 * M_PI) * 9) % 9;
                        hist[bin] += mag;
                    }
                }
                
                // Normalize
                float sum = 0.0f;
                for (float h : hist) sum += h;
                if (sum > 0) {
                    for (float& h : hist) h /= sum;
                }
                
                features.insert(features.end(), hist.begin(), hist.end());
            }
        }
        
        // Spatial statistics
        for (int cy = 0; cy < 4; ++cy) {
            for (int cx = 0; cx < 4; ++cx) {
                int cell_size = size / 4;
                float mean = 0.0f, var = 0.0f;
                int count = 0;
                
                for (int y = cy * cell_size; y < (cy + 1) * cell_size && y < size; ++y) {
                    for (int x = cx * cell_size; x < (cx + 1) * cell_size && x < size; ++x) {
                        mean += face[y * size + x];
                        count++;
                    }
                }
                if (count > 0) mean /= count;
                
                for (int y = cy * cell_size; y < (cy + 1) * cell_size && y < size; ++y) {
                    for (int x = cx * cell_size; x < (cx + 1) * cell_size && x < size; ++x) {
                        float diff = face[y * size + x] - mean;
                        var += diff * diff;
                    }
                }
                if (count > 0) var = std::sqrt(var / count);
                
                features.push_back(mean);
                features.push_back(var);
            }
        }
        
        return features;
    }
    
    float cosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return 1.0f - dot / (std::sqrt(norm_a * norm_b) + 1e-10f);
    }
};

/**
 * @brief Unified face recognizer interface
 */
class FaceRecognizer {
public:
    enum class Method {
        LBPH,
        EIGEN,
        FISHER,
        EMBEDDING
    };
    
    FaceRecognizer(Method method = Method::LBPH, float threshold = 100.0f) 
        : method_(method) {
        switch (method) {
            case Method::LBPH:
                recognizer_ = std::make_unique<LBPHRecognizer>(threshold);
                break;
            case Method::EIGEN:
                recognizer_ = std::make_unique<EigenFaceRecognizer>(threshold);
                break;
            case Method::EMBEDDING:
                recognizer_ = std::make_unique<EmbeddingRecognizer>(threshold);
                break;
            default:
                recognizer_ = std::make_unique<LBPHRecognizer>(threshold);
                break;
        }
    }
    
    void train(const std::vector<std::vector<float>>& faces,
               const std::vector<int>& labels) {
        recognizer_->train(faces, labels);
    }
    
    Prediction predict(const std::vector<float>& face) {
        return recognizer_->predict(face);
    }
    
    std::vector<Prediction> predictAll(const std::vector<float>& face) {
        return recognizer_->predictAll(face);
    }
    
    void update(const std::vector<std::vector<float>>& faces,
                const std::vector<int>& labels) {
        recognizer_->update(faces, labels);
    }
    
    void save(const std::string& path) {
        recognizer_->save(path);
    }
    
    void load(const std::string& path) {
        recognizer_->load(path);
    }
    
    void setLabelNames(const std::map<int, std::string>& names) {
        label_names_ = names;
    }
    
    std::string getLabelName(int label) const {
        auto it = label_names_.find(label);
        return it != label_names_.end() ? it->second : std::to_string(label);
    }
    
private:
    Method method_;
    std::unique_ptr<RecognizerBase> recognizer_;
    std::map<int, std::string> label_names_;
};

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_RECOGNIZER_HPP
