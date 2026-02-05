// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/ml.hpp - Machine Learning Algorithms
 * 
 * This header provides classical ML algorithms:
 * - Classification (SVM, KNN, Decision Tree, Random Forest, etc.)
 * - Regression (Linear, Ridge, Lasso, etc.)
 * - Clustering (KMeans, DBSCAN, etc.)
 * - Dimensionality Reduction (PCA, LDA, t-SNE, UMAP)
 * - Ensemble Methods (Bagging, Boosting, etc.)
 */

#ifndef NEUROVA_ML_HPP
#define NEUROVA_ML_HPP

#include "core.hpp"
#include <string>
#include <map>

namespace neurova {
namespace ml {

// ============================================================================
// Base Estimator Classes
// ============================================================================

class Estimator {
public:
    virtual ~Estimator() = default;
    virtual void fit(const Tensor& X, const Tensor& y) = 0;
    virtual Tensor predict(const Tensor& X) const = 0;
    virtual double score(const Tensor& X, const Tensor& y) const;
};

class Transformer {
public:
    virtual ~Transformer() = default;
    virtual void fit(const Tensor& X) = 0;
    virtual Tensor transform(const Tensor& X) const = 0;
    virtual Tensor fit_transform(const Tensor& X);
    virtual Tensor inverse_transform(const Tensor& X) const;
};

class Classifier : public Estimator {
public:
    virtual Tensor predict_proba(const Tensor& X) const;
    virtual Tensor predict_log_proba(const Tensor& X) const;
};

class Regressor : public Estimator {};

class Clusterer {
public:
    virtual ~Clusterer() = default;
    virtual void fit(const Tensor& X) = 0;
    virtual Tensor predict(const Tensor& X) const = 0;
    virtual Tensor fit_predict(const Tensor& X);
};

// ============================================================================
// Linear Models
// ============================================================================

class LinearRegression : public Regressor {
public:
    LinearRegression(bool fit_intercept = true);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor coef() const { return coef_; }
    double intercept() const { return intercept_; }

private:
    bool fit_intercept_;
    Tensor coef_;
    double intercept_;
};

class Ridge : public Regressor {
public:
    Ridge(double alpha = 1.0, bool fit_intercept = true, 
          int max_iter = 1000, double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor coef() const { return coef_; }
    double intercept() const { return intercept_; }

private:
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    Tensor coef_;
    double intercept_;
};

class Lasso : public Regressor {
public:
    Lasso(double alpha = 1.0, bool fit_intercept = true,
          int max_iter = 1000, double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor coef() const { return coef_; }
    double intercept() const { return intercept_; }

private:
    double alpha_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    Tensor coef_;
    double intercept_;
};

class ElasticNet : public Regressor {
public:
    ElasticNet(double alpha = 1.0, double l1_ratio = 0.5, 
               bool fit_intercept = true, int max_iter = 1000, 
               double tol = 1e-4);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;

private:
    double alpha_, l1_ratio_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    Tensor coef_;
    double intercept_;
};

class LogisticRegression : public Classifier {
public:
    LogisticRegression(double C = 1.0, const std::string& penalty = "l2",
                       bool fit_intercept = true, int max_iter = 100,
                       double tol = 1e-4, const std::string& solver = "lbfgs");
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    
    Tensor coef() const { return coef_; }
    double intercept() const { return intercept_; }

private:
    double C_;
    std::string penalty_;
    bool fit_intercept_;
    int max_iter_;
    double tol_;
    std::string solver_;
    Tensor coef_;
    double intercept_;
    std::vector<int> classes_;
};

// ============================================================================
// Support Vector Machines
// ============================================================================

enum class SVMKernel {
    LINEAR,
    POLY,
    RBF,
    SIGMOID
};

class SVC : public Classifier {
public:
    SVC(double C = 1.0, SVMKernel kernel = SVMKernel::RBF,
        int degree = 3, double gamma = 0.0, double coef0 = 0.0,
        double tol = 1e-3, int max_iter = -1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    Tensor decision_function(const Tensor& X) const;

private:
    double C_;
    SVMKernel kernel_;
    int degree_;
    double gamma_, coef0_;
    double tol_;
    int max_iter_;
    
    Tensor support_vectors_;
    Tensor dual_coef_;
    Tensor intercept_;
    std::vector<int> support_;
    std::vector<int> classes_;
    
    Tensor compute_kernel(const Tensor& X1, const Tensor& X2) const;
};

class SVR : public Regressor {
public:
    SVR(double C = 1.0, double epsilon = 0.1, 
        SVMKernel kernel = SVMKernel::RBF,
        int degree = 3, double gamma = 0.0, double coef0 = 0.0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;

private:
    double C_, epsilon_;
    SVMKernel kernel_;
    int degree_;
    double gamma_, coef0_;
    
    Tensor support_vectors_;
    Tensor dual_coef_;
    double intercept_;
};

// ============================================================================
// K-Nearest Neighbors
// ============================================================================

enum class KNNWeight {
    UNIFORM,
    DISTANCE
};

enum class KNNMetric {
    EUCLIDEAN,
    MANHATTAN,
    CHEBYSHEV,
    MINKOWSKI,
    COSINE
};

class KNeighborsClassifier : public Classifier {
public:
    KNeighborsClassifier(int n_neighbors = 5, 
                         KNNWeight weights = KNNWeight::UNIFORM,
                         KNNMetric metric = KNNMetric::EUCLIDEAN,
                         double p = 2.0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    
    std::pair<Tensor, Tensor> kneighbors(const Tensor& X, int n_neighbors = -1) const;

private:
    int n_neighbors_;
    KNNWeight weights_;
    KNNMetric metric_;
    double p_;
    
    Tensor X_train_;
    Tensor y_train_;
    std::vector<int> classes_;
    
    Tensor compute_distances(const Tensor& X) const;
};

class KNeighborsRegressor : public Regressor {
public:
    KNeighborsRegressor(int n_neighbors = 5,
                        KNNWeight weights = KNNWeight::UNIFORM,
                        KNNMetric metric = KNNMetric::EUCLIDEAN);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;

private:
    int n_neighbors_;
    KNNWeight weights_;
    KNNMetric metric_;
    
    Tensor X_train_;
    Tensor y_train_;
};

// ============================================================================
// Decision Trees
// ============================================================================

enum class TreeCriterion {
    GINI,       // Classification
    ENTROPY,    // Classification
    MSE,        // Regression
    MAE         // Regression
};

struct TreeNode {
    int feature_index;
    double threshold;
    double value;           // For leaf nodes
    std::vector<double> class_probs;  // For classification
    int left_child;
    int right_child;
    bool is_leaf;
};

class DecisionTreeClassifier : public Classifier {
public:
    DecisionTreeClassifier(TreeCriterion criterion = TreeCriterion::GINI,
                           int max_depth = -1, int min_samples_split = 2,
                           int min_samples_leaf = 1, int max_features = -1,
                           int random_state = -1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    
    Tensor feature_importances() const;

private:
    TreeCriterion criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    int random_state_;
    
    std::vector<TreeNode> nodes_;
    std::vector<int> classes_;
    Tensor feature_importances_;
    
    int build_tree(const Tensor& X, const Tensor& y, int depth);
    std::pair<int, double> find_best_split(const Tensor& X, const Tensor& y);
    double compute_criterion(const Tensor& y);
};

class DecisionTreeRegressor : public Regressor {
public:
    DecisionTreeRegressor(TreeCriterion criterion = TreeCriterion::MSE,
                          int max_depth = -1, int min_samples_split = 2,
                          int min_samples_leaf = 1, int max_features = -1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor feature_importances() const;

private:
    TreeCriterion criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    
    std::vector<TreeNode> nodes_;
    Tensor feature_importances_;
};

// ============================================================================
// Ensemble Methods
// ============================================================================

class RandomForestClassifier : public Classifier {
public:
    RandomForestClassifier(int n_estimators = 100,
                           TreeCriterion criterion = TreeCriterion::GINI,
                           int max_depth = -1, int min_samples_split = 2,
                           int min_samples_leaf = 1, int max_features = -1,
                           bool bootstrap = true, int n_jobs = -1,
                           int random_state = -1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    
    Tensor feature_importances() const;

private:
    int n_estimators_;
    TreeCriterion criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    bool bootstrap_;
    int n_jobs_;
    int random_state_;
    
    std::vector<DecisionTreeClassifier> trees_;
    std::vector<int> classes_;
};

class RandomForestRegressor : public Regressor {
public:
    RandomForestRegressor(int n_estimators = 100,
                          TreeCriterion criterion = TreeCriterion::MSE,
                          int max_depth = -1, int min_samples_split = 2,
                          int min_samples_leaf = 1, int max_features = -1,
                          bool bootstrap = true, int n_jobs = -1);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor feature_importances() const;

private:
    int n_estimators_;
    TreeCriterion criterion_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    int max_features_;
    bool bootstrap_;
    int n_jobs_;
    
    std::vector<DecisionTreeRegressor> trees_;
};

class GradientBoostingClassifier : public Classifier {
public:
    GradientBoostingClassifier(int n_estimators = 100,
                               double learning_rate = 0.1,
                               int max_depth = 3, int min_samples_split = 2,
                               int min_samples_leaf = 1,
                               double subsample = 1.0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    Tensor decision_function(const Tensor& X) const;

private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double subsample_;
    
    std::vector<std::vector<DecisionTreeRegressor>> trees_;
    std::vector<int> classes_;
    double init_score_;
};

class GradientBoostingRegressor : public Regressor {
public:
    GradientBoostingRegressor(int n_estimators = 100,
                              double learning_rate = 0.1,
                              int max_depth = 3, int min_samples_split = 2,
                              int min_samples_leaf = 1,
                              double subsample = 1.0,
                              const std::string& loss = "squared_error");
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;

private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    double subsample_;
    std::string loss_;
    
    std::vector<DecisionTreeRegressor> trees_;
    double init_score_;
};

class AdaBoostClassifier : public Classifier {
public:
    AdaBoostClassifier(int n_estimators = 50, double learning_rate = 1.0,
                       const std::string& algorithm = "SAMME.R");
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;

private:
    int n_estimators_;
    double learning_rate_;
    std::string algorithm_;
    
    std::vector<DecisionTreeClassifier> trees_;
    std::vector<double> weights_;
    std::vector<int> classes_;
};

class BaggingClassifier : public Classifier {
public:
    BaggingClassifier(int n_estimators = 10, int max_samples = -1,
                      int max_features = -1, bool bootstrap = true,
                      bool bootstrap_features = false);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;

private:
    int n_estimators_;
    int max_samples_;
    int max_features_;
    bool bootstrap_;
    bool bootstrap_features_;
    
    std::vector<DecisionTreeClassifier> trees_;
    std::vector<std::vector<size_t>> feature_indices_;
};

// ============================================================================
// Clustering
// ============================================================================

class KMeans : public Clusterer {
public:
    KMeans(int n_clusters = 8, int max_iter = 300, int n_init = 10,
           double tol = 1e-4, const std::string& init = "k-means++",
           int random_state = -1);
    
    void fit(const Tensor& X) override;
    Tensor predict(const Tensor& X) const override;
    Tensor transform(const Tensor& X) const;
    
    Tensor cluster_centers() const { return centers_; }
    Tensor labels() const { return labels_; }
    double inertia() const { return inertia_; }

private:
    int n_clusters_;
    int max_iter_;
    int n_init_;
    double tol_;
    std::string init_;
    int random_state_;
    
    Tensor centers_;
    Tensor labels_;
    double inertia_;
    
    Tensor init_centroids(const Tensor& X);
};

class MiniBatchKMeans : public Clusterer {
public:
    MiniBatchKMeans(int n_clusters = 8, int batch_size = 1024,
                    int max_iter = 100, double tol = 0.0,
                    int max_no_improvement = 10);
    
    void fit(const Tensor& X) override;
    void partial_fit(const Tensor& X);
    Tensor predict(const Tensor& X) const override;
    
    Tensor cluster_centers() const { return centers_; }

private:
    int n_clusters_;
    int batch_size_;
    int max_iter_;
    double tol_;
    int max_no_improvement_;
    
    Tensor centers_;
    std::vector<int> counts_;
};

class DBSCAN : public Clusterer {
public:
    DBSCAN(double eps = 0.5, int min_samples = 5, 
           KNNMetric metric = KNNMetric::EUCLIDEAN);
    
    void fit(const Tensor& X) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor labels() const { return labels_; }
    std::vector<std::vector<int>> core_sample_indices() const;

private:
    double eps_;
    int min_samples_;
    KNNMetric metric_;
    
    Tensor X_train_;
    Tensor labels_;
    std::vector<bool> core_samples_;
};

class AgglomerativeClustering : public Clusterer {
public:
    AgglomerativeClustering(int n_clusters = 2,
                            const std::string& linkage = "ward",
                            KNNMetric metric = KNNMetric::EUCLIDEAN);
    
    void fit(const Tensor& X) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor labels() const { return labels_; }

private:
    int n_clusters_;
    std::string linkage_;
    KNNMetric metric_;
    
    Tensor labels_;
};

class SpectralClustering : public Clusterer {
public:
    SpectralClustering(int n_clusters = 8, int n_components = -1,
                       double gamma = 1.0, const std::string& affinity = "rbf");
    
    void fit(const Tensor& X) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor labels() const { return labels_; }

private:
    int n_clusters_;
    int n_components_;
    double gamma_;
    std::string affinity_;
    
    Tensor labels_;
};

class MeanShift : public Clusterer {
public:
    MeanShift(double bandwidth = 0.0, int max_iter = 300);
    
    void fit(const Tensor& X) override;
    Tensor predict(const Tensor& X) const override;
    
    Tensor cluster_centers() const { return centers_; }
    Tensor labels() const { return labels_; }

private:
    double bandwidth_;
    int max_iter_;
    
    Tensor centers_;
    Tensor labels_;
    
    double estimate_bandwidth(const Tensor& X);
};

// ============================================================================
// Dimensionality Reduction
// ============================================================================

class PCA : public Transformer {
public:
    PCA(int n_components = -1, bool whiten = false, double svd_solver = 0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;
    
    Tensor components() const { return components_; }
    Tensor explained_variance() const { return explained_variance_; }
    Tensor explained_variance_ratio() const { return explained_variance_ratio_; }
    Tensor singular_values() const { return singular_values_; }
    Tensor mean() const { return mean_; }

private:
    int n_components_;
    bool whiten_;
    
    Tensor components_;
    Tensor explained_variance_;
    Tensor explained_variance_ratio_;
    Tensor singular_values_;
    Tensor mean_;
};

class IncrementalPCA : public Transformer {
public:
    IncrementalPCA(int n_components = -1, int batch_size = -1);
    
    void fit(const Tensor& X) override;
    void partial_fit(const Tensor& X);
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    int n_components_;
    int batch_size_;
    int n_samples_seen_;
    
    Tensor components_;
    Tensor mean_;
    Tensor var_;
    Tensor singular_values_;
};

class KernelPCA : public Transformer {
public:
    KernelPCA(int n_components = -1, SVMKernel kernel = SVMKernel::RBF,
              double gamma = 0.0, int degree = 3, double coef0 = 1.0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    int n_components_;
    SVMKernel kernel_;
    double gamma_;
    int degree_;
    double coef0_;
    
    Tensor X_fit_;
    Tensor alphas_;
    Tensor lambdas_;
};

class LDA : public Transformer {
public:
    LDA(int n_components = -1);
    
    void fit(const Tensor& X) override;
    void fit(const Tensor& X, const Tensor& y);
    Tensor transform(const Tensor& X) const override;
    Tensor predict(const Tensor& X) const;

private:
    int n_components_;
    
    Tensor scalings_;
    Tensor means_;
    Tensor priors_;
    std::vector<int> classes_;
};

class TSNE : public Transformer {
public:
    TSNE(int n_components = 2, double perplexity = 30.0, 
         double learning_rate = 200.0, int n_iter = 1000,
         int random_state = -1, double early_exaggeration = 12.0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor fit_transform(const Tensor& X) override;

private:
    int n_components_;
    double perplexity_;
    double learning_rate_;
    int n_iter_;
    int random_state_;
    double early_exaggeration_;
    
    Tensor embedding_;
    
    Tensor compute_pairwise_affinities(const Tensor& X);
};

class UMAP : public Transformer {
public:
    UMAP(int n_components = 2, int n_neighbors = 15,
         double min_dist = 0.1, double spread = 1.0,
         KNNMetric metric = KNNMetric::EUCLIDEAN,
         int n_epochs = -1, double learning_rate = 1.0,
         int random_state = -1);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor fit_transform(const Tensor& X) override;

private:
    int n_components_;
    int n_neighbors_;
    double min_dist_;
    double spread_;
    KNNMetric metric_;
    int n_epochs_;
    double learning_rate_;
    int random_state_;
    
    Tensor embedding_;
    Tensor graph_;
};

class TruncatedSVD : public Transformer {
public:
    TruncatedSVD(int n_components = 2, int n_iter = 5, int random_state = -1);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    int n_components_;
    int n_iter_;
    int random_state_;
    
    Tensor components_;
    Tensor singular_values_;
    Tensor explained_variance_;
};

class NMF : public Transformer {
public:
    NMF(int n_components = -1, int max_iter = 200, double tol = 1e-4,
        int random_state = -1, const std::string& init = "nndsvd");
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;
    
    Tensor components() const { return components_; }

private:
    int n_components_;
    int max_iter_;
    double tol_;
    int random_state_;
    std::string init_;
    
    Tensor components_;
};

// ============================================================================
// Gaussian Processes
// ============================================================================

class GaussianProcessClassifier : public Classifier {
public:
    GaussianProcessClassifier(SVMKernel kernel = SVMKernel::RBF,
                              int max_iter_predict = 100, int n_restarts = 0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;

private:
    SVMKernel kernel_;
    int max_iter_predict_;
    int n_restarts_;
    
    Tensor X_train_;
    Tensor y_train_;
    Tensor K_;
    std::vector<int> classes_;
};

class GaussianProcessRegressor : public Regressor {
public:
    GaussianProcessRegressor(SVMKernel kernel = SVMKernel::RBF,
                             double alpha = 1e-10, bool normalize_y = false);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    std::pair<Tensor, Tensor> predict_with_std(const Tensor& X) const;

private:
    SVMKernel kernel_;
    double alpha_;
    bool normalize_y_;
    
    Tensor X_train_;
    Tensor y_train_;
    Tensor alpha_vec_;
    Tensor L_;
    double y_mean_;
};

// ============================================================================
// Naive Bayes
// ============================================================================

class GaussianNB : public Classifier {
public:
    GaussianNB(double var_smoothing = 1e-9);
    
    void fit(const Tensor& X, const Tensor& y) override;
    void partial_fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;
    Tensor predict_log_proba(const Tensor& X) const override;

private:
    double var_smoothing_;
    
    Tensor theta_;       // Mean of each feature per class
    Tensor var_;         // Variance of each feature per class
    Tensor class_prior_; // Prior probability of each class
    std::vector<int> classes_;
};

class MultinomialNB : public Classifier {
public:
    MultinomialNB(double alpha = 1.0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    void partial_fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;

private:
    double alpha_;
    
    Tensor feature_log_prob_;
    Tensor class_log_prior_;
    std::vector<int> classes_;
};

class BernoulliNB : public Classifier {
public:
    BernoulliNB(double alpha = 1.0, double binarize = 0.0);
    
    void fit(const Tensor& X, const Tensor& y) override;
    Tensor predict(const Tensor& X) const override;
    Tensor predict_proba(const Tensor& X) const override;

private:
    double alpha_;
    double binarize_;
    
    Tensor feature_log_prob_;
    Tensor class_log_prior_;
    std::vector<int> classes_;
};

// ============================================================================
// Preprocessing
// ============================================================================

class StandardScaler : public Transformer {
public:
    StandardScaler(bool with_mean = true, bool with_std = true);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;
    
    Tensor mean() const { return mean_; }
    Tensor scale() const { return scale_; }

private:
    bool with_mean_;
    bool with_std_;
    Tensor mean_;
    Tensor scale_;
};

class MinMaxScaler : public Transformer {
public:
    MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    double feature_min_;
    double feature_max_;
    Tensor data_min_;
    Tensor data_max_;
    Tensor scale_;
};

class MaxAbsScaler : public Transformer {
public:
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    Tensor max_abs_;
};

class RobustScaler : public Transformer {
public:
    RobustScaler(bool with_centering = true, bool with_scaling = true,
                 double quantile_min = 25.0, double quantile_max = 75.0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    bool with_centering_;
    bool with_scaling_;
    double quantile_min_;
    double quantile_max_;
    Tensor center_;
    Tensor scale_;
};

class Normalizer : public Transformer {
public:
    Normalizer(const std::string& norm = "l2");
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;

private:
    std::string norm_;
};

class Binarizer : public Transformer {
public:
    Binarizer(double threshold = 0.0);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;

private:
    double threshold_;
};

class PolynomialFeatures : public Transformer {
public:
    PolynomialFeatures(int degree = 2, bool interaction_only = false,
                       bool include_bias = true);
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;

private:
    int degree_;
    bool interaction_only_;
    bool include_bias_;
    size_t n_input_features_;
    size_t n_output_features_;
};

class LabelEncoder {
public:
    void fit(const Tensor& y);
    Tensor transform(const Tensor& y) const;
    Tensor inverse_transform(const Tensor& y) const;
    Tensor fit_transform(const Tensor& y);
    
    std::vector<int> classes() const { return classes_; }

private:
    std::vector<int> classes_;
    std::map<int, int> class_to_idx_;
};

class OneHotEncoder : public Transformer {
public:
    OneHotEncoder(bool sparse = false, const std::string& handle_unknown = "error");
    
    void fit(const Tensor& X) override;
    Tensor transform(const Tensor& X) const override;
    Tensor inverse_transform(const Tensor& X) const override;

private:
    bool sparse_;
    std::string handle_unknown_;
    std::vector<std::vector<int>> categories_;
};

// ============================================================================
// Model Selection
// ============================================================================

std::tuple<Tensor, Tensor, Tensor, Tensor> train_test_split(
    const Tensor& X, const Tensor& y, double test_size = 0.25,
    int random_state = -1, bool shuffle = true, bool stratify = false);

// K-Fold cross validation
class KFold {
public:
    KFold(int n_splits = 5, bool shuffle = false, int random_state = -1);
    
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> 
    split(const Tensor& X) const;

private:
    int n_splits_;
    bool shuffle_;
    int random_state_;
};

class StratifiedKFold {
public:
    StratifiedKFold(int n_splits = 5, bool shuffle = false, int random_state = -1);
    
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
    split(const Tensor& X, const Tensor& y) const;

private:
    int n_splits_;
    bool shuffle_;
    int random_state_;
};

// Cross validation score
Tensor cross_val_score(Estimator& estimator, const Tensor& X, const Tensor& y,
                       int cv = 5, const std::string& scoring = "accuracy");

// Grid search
class GridSearchCV {
public:
    GridSearchCV(Estimator& estimator, 
                 const std::map<std::string, std::vector<double>>& param_grid,
                 int cv = 5, const std::string& scoring = "accuracy",
                 int n_jobs = -1);
    
    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    
    std::map<std::string, double> best_params() const { return best_params_; }
    double best_score() const { return best_score_; }

private:
    Estimator& estimator_;
    std::map<std::string, std::vector<double>> param_grid_;
    int cv_;
    std::string scoring_;
    int n_jobs_;
    
    std::map<std::string, double> best_params_;
    double best_score_;
};

// ============================================================================
// Metrics
// ============================================================================

namespace metrics {

// Classification metrics
double accuracy_score(const Tensor& y_true, const Tensor& y_pred);
double precision_score(const Tensor& y_true, const Tensor& y_pred, 
                       const std::string& average = "binary");
double recall_score(const Tensor& y_true, const Tensor& y_pred,
                    const std::string& average = "binary");
double f1_score(const Tensor& y_true, const Tensor& y_pred,
                const std::string& average = "binary");
Tensor confusion_matrix(const Tensor& y_true, const Tensor& y_pred);
std::string classification_report(const Tensor& y_true, const Tensor& y_pred);

double roc_auc_score(const Tensor& y_true, const Tensor& y_score);
std::pair<Tensor, Tensor> roc_curve(const Tensor& y_true, const Tensor& y_score);
double average_precision_score(const Tensor& y_true, const Tensor& y_score);
std::pair<Tensor, Tensor> precision_recall_curve(const Tensor& y_true, 
                                                  const Tensor& y_score);
double log_loss(const Tensor& y_true, const Tensor& y_pred);
double hinge_loss(const Tensor& y_true, const Tensor& y_pred);

// Regression metrics
double mean_squared_error(const Tensor& y_true, const Tensor& y_pred);
double mean_absolute_error(const Tensor& y_true, const Tensor& y_pred);
double r2_score(const Tensor& y_true, const Tensor& y_pred);
double explained_variance_score(const Tensor& y_true, const Tensor& y_pred);
double max_error(const Tensor& y_true, const Tensor& y_pred);
double mean_absolute_percentage_error(const Tensor& y_true, const Tensor& y_pred);

// Clustering metrics
double silhouette_score(const Tensor& X, const Tensor& labels);
double calinski_harabasz_score(const Tensor& X, const Tensor& labels);
double davies_bouldin_score(const Tensor& X, const Tensor& labels);
double adjusted_rand_score(const Tensor& labels_true, const Tensor& labels_pred);
double normalized_mutual_info_score(const Tensor& labels_true, const Tensor& labels_pred);
double homogeneity_score(const Tensor& labels_true, const Tensor& labels_pred);
double completeness_score(const Tensor& labels_true, const Tensor& labels_pred);
double v_measure_score(const Tensor& labels_true, const Tensor& labels_pred);

// Distance metrics
Tensor pairwise_distances(const Tensor& X, const Tensor& Y = Tensor(), 
                          KNNMetric metric = KNNMetric::EUCLIDEAN);
Tensor cosine_similarity(const Tensor& X, const Tensor& Y = Tensor());
Tensor euclidean_distances(const Tensor& X, const Tensor& Y = Tensor());
Tensor manhattan_distances(const Tensor& X, const Tensor& Y = Tensor());

} // namespace metrics

} // namespace ml
} // namespace neurova

#endif // NEUROVA_ML_HPP
