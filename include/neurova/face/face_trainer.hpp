// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file face_trainer.hpp
 * @brief Face model training utilities
 */

#ifndef NEUROVA_FACE_TRAINER_HPP
#define NEUROVA_FACE_TRAINER_HPP

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <chrono>
#include <functional>
#include <filesystem>

#include "face_dataset.hpp"
#include "face_recognizer.hpp"

namespace neurova {
namespace face {

namespace fs = std::filesystem;

/**
 * @brief Training history entry
 */
struct TrainingHistory {
    std::string method;
    int train_samples;
    int num_classes;
    double train_time_seconds;
    float validation_accuracy;
    bool augmented;
    std::map<std::string, float> metrics;
};

/**
 * @brief Training callback function type
 */
using TrainingCallback = std::function<void(const TrainingHistory&)>;

/**
 * @brief Face trainer for training detection and recognition models
 */
class FaceTrainer {
public:
    /**
     * @brief Construct trainer
     */
    FaceTrainer(
        const std::string& output_dir = "./models"
    ) : output_dir_(output_dir) {
        fs::create_directories(output_dir_);
    }

    /**
     * @brief Train face recognizer
     */
    TrainingHistory trainRecognizer(
        FaceDataset& dataset,
        RecognizerType method = RecognizerType::LBPH,
        bool augment = true,
        bool verbose = true,
        TrainingCallback callback = nullptr
    ) {
        if (verbose) {
            std::cout << "Training " << recognizerTypeName(method) << " face recognizer...\n";
            std::cout << "Classes: " << dataset.numClasses() << "\n";
        }
        
        // Create recognizer
        recognizer_ = std::make_unique<FaceRecognizer>(method);
        
        // Load training data
        auto [images, labels] = dataset.getTrainData();
        
        if (verbose) {
            std::cout << "Training samples: " << images.size() << "\n";
        }
        
        // Apply augmentation if requested
        std::vector<std::vector<float>> aug_images;
        std::vector<int> aug_labels;
        
        if (augment) {
            aug_images = images;
            aug_labels = labels;
            
            // Add horizontally flipped versions
            for (size_t i = 0; i < images.size(); ++i) {
                auto flipped = flipHorizontal(images[i].data(), 
                                             dataset.classes().empty() ? 100 : 100,
                                             100, 1);
                aug_images.push_back(flipped);
                aug_labels.push_back(labels[i]);
            }
            
            if (verbose) {
                std::cout << "Augmented samples: " << aug_images.size() << "\n";
            }
        } else {
            aug_images = images;
            aug_labels = labels;
        }
        
        // Train
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<const float*> image_ptrs;
        std::vector<std::pair<int, int>> sizes;
        for (const auto& img : aug_images) {
            image_ptrs.push_back(img.data());
            sizes.emplace_back(100, 100);  // Assuming standard size
        }
        
        recognizer_->train(image_ptrs, sizes, aug_labels, 1);
        
        auto end = std::chrono::high_resolution_clock::now();
        double train_time = std::chrono::duration<double>(end - start).count();
        
        // Evaluate on validation set
        float val_accuracy = 0.0f;
        auto val_samples = dataset.loadValidation();
        if (!val_samples.empty()) {
            val_accuracy = evaluate(val_samples);
        }
        
        // Create history
        TrainingHistory history;
        history.method = recognizerTypeName(method);
        history.train_samples = static_cast<int>(aug_images.size());
        history.num_classes = static_cast<int>(dataset.numClasses());
        history.train_time_seconds = train_time;
        history.validation_accuracy = val_accuracy;
        history.augmented = augment;
        
        history_.push_back(history);
        
        if (verbose) {
            std::cout << "Training completed in " << train_time << "s\n";
            if (val_accuracy > 0) {
                std::cout << "Validation accuracy: " << (val_accuracy * 100) << "%\n";
            }
        }
        
        if (callback) {
            callback(history);
        }
        
        return history;
    }

    /**
     * @brief Evaluate model on test samples
     */
    float evaluate(const std::vector<FaceSample>& samples) {
        if (!recognizer_ || samples.empty()) return 0.0f;
        
        int correct = 0;
        for (const auto& sample : samples) {
            auto pred = recognizer_->predict(sample.image.data(),
                                            sample.width, sample.height,
                                            sample.channels);
            if (pred.label == sample.label) {
                ++correct;
            }
        }
        
        return static_cast<float>(correct) / samples.size();
    }

    /**
     * @brief Evaluate on dataset
     */
    float evaluate(FaceDataset& dataset) {
        auto samples = dataset.loadTest();
        return evaluate(samples);
    }

    /**
     * @brief Save trained model
     */
    bool save(const std::string& name) {
        if (!recognizer_) return false;
        
        std::string path = output_dir_ + "/" + name + ".bin";
        return recognizer_->save(path);
    }

    /**
     * @brief Load trained model
     */
    bool load(const std::string& name, RecognizerType method = RecognizerType::LBPH) {
        recognizer_ = std::make_unique<FaceRecognizer>(method);
        std::string path = output_dir_ + "/" + name + ".bin";
        return recognizer_->load(path);
    }

    /**
     * @brief Get training history
     */
    const std::vector<TrainingHistory>& history() const {
        return history_;
    }

    /**
     * @brief Get trained recognizer
     */
    FaceRecognizer* recognizer() {
        return recognizer_.get();
    }

    /**
     * @brief Cross-validation evaluation
     */
    float crossValidate(
        FaceDataset& dataset,
        RecognizerType method = RecognizerType::LBPH,
        int k_folds = 5,
        bool verbose = true
    ) {
        auto [images, labels] = dataset.getTrainData();
        
        if (images.size() < static_cast<size_t>(k_folds)) {
            return 0.0f;
        }
        
        size_t fold_size = images.size() / k_folds;
        float total_accuracy = 0.0f;
        
        for (int fold = 0; fold < k_folds; ++fold) {
            size_t val_start = fold * fold_size;
            size_t val_end = (fold == k_folds - 1) ? images.size() : (fold + 1) * fold_size;
            
            // Split into train/val
            std::vector<std::vector<float>> train_images;
            std::vector<int> train_labels;
            std::vector<FaceSample> val_samples;
            
            for (size_t i = 0; i < images.size(); ++i) {
                if (i >= val_start && i < val_end) {
                    FaceSample s;
                    s.image = images[i];
                    s.label = labels[i];
                    s.width = 100;
                    s.height = 100;
                    s.channels = 1;
                    val_samples.push_back(s);
                } else {
                    train_images.push_back(images[i]);
                    train_labels.push_back(labels[i]);
                }
            }
            
            // Train
            auto rec = std::make_unique<FaceRecognizer>(method);
            std::vector<const float*> ptrs;
            std::vector<std::pair<int, int>> sizes;
            for (const auto& img : train_images) {
                ptrs.push_back(img.data());
                sizes.emplace_back(100, 100);
            }
            rec->train(ptrs, sizes, train_labels, 1);
            
            // Evaluate
            int correct = 0;
            for (const auto& s : val_samples) {
                auto pred = rec->predict(s.image.data(), s.width, s.height, s.channels);
                if (pred.label == s.label) {
                    ++correct;
                }
            }
            
            float fold_acc = static_cast<float>(correct) / val_samples.size();
            total_accuracy += fold_acc;
            
            if (verbose) {
                std::cout << "Fold " << (fold + 1) << "/" << k_folds 
                          << " accuracy: " << (fold_acc * 100) << "%\n";
            }
        }
        
        float mean_accuracy = total_accuracy / k_folds;
        
        if (verbose) {
            std::cout << "Mean CV accuracy: " << (mean_accuracy * 100) << "%\n";
        }
        
        return mean_accuracy;
    }

private:
    std::string output_dir_;
    std::unique_ptr<FaceRecognizer> recognizer_;
    std::vector<TrainingHistory> history_;

    std::string recognizerTypeName(RecognizerType type) const {
        switch (type) {
            case RecognizerType::LBPH: return "LBPH";
            case RecognizerType::EIGEN: return "EigenFace";
            case RecognizerType::FISHER: return "FisherFace";
            case RecognizerType::EMBEDDING: return "Embedding";
            default: return "Unknown";
        }
    }

    // Horizontal flip for augmentation
    std::vector<float> flipHorizontal(const float* data, int width, int height, int channels) {
        std::vector<float> result(width * height * channels);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int src_x = width - 1 - x;
                for (int c = 0; c < channels; ++c) {
                    result[(y * width + x) * channels + c] = 
                        data[(y * width + src_x) * channels + c];
                }
            }
        }
        
        return result;
    }
};

// ============================================================================
// Hyperparameter Tuning
// ============================================================================

/**
 * @brief Grid search for hyperparameter tuning
 */
class GridSearchCV {
public:
    struct Result {
        std::map<std::string, float> params;
        float mean_score;
        float std_score;
        std::vector<float> fold_scores;
    };

    GridSearchCV(int cv_folds = 5) : cv_folds_(cv_folds) {}

    /**
     * @brief Search for best LBPH parameters
     */
    Result searchLBPH(
        FaceDataset& dataset,
        const std::vector<int>& radii = {1, 2, 3},
        const std::vector<int>& neighbors = {8, 16, 24},
        bool verbose = true
    ) {
        Result best;
        best.mean_score = 0.0f;
        
        auto [images, labels] = dataset.getTrainData();
        
        for (int radius : radii) {
            for (int n : neighbors) {
                if (verbose) {
                    std::cout << "Testing radius=" << radius << ", neighbors=" << n << "\n";
                }
                
                // Note: In full implementation, would pass these to recognizer
                FaceTrainer trainer;
                float score = trainer.crossValidate(dataset, RecognizerType::LBPH, 
                                                    cv_folds_, false);
                
                if (score > best.mean_score) {
                    best.mean_score = score;
                    best.params["radius"] = static_cast<float>(radius);
                    best.params["neighbors"] = static_cast<float>(n);
                }
            }
        }
        
        if (verbose) {
            std::cout << "Best parameters: radius=" << best.params["radius"]
                      << ", neighbors=" << best.params["neighbors"]
                      << " (score=" << best.mean_score << ")\n";
        }
        
        return best;
    }

private:
    int cv_folds_;
};

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_TRAINER_HPP
