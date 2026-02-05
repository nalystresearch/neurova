# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Machine Learning Module for Neurova

This module provides comprehensive machine learning algorithms implemented
with NumPy only.

Included:
- Preprocessing: StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder
- Advanced Scaling: RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
- Imputation: SimpleImputer, KNNImputer
- Feature Selection: SelectKBest, RFE, VarianceThreshold
- Classification: KNN, Logistic Regression, Naive Bayes, Decision Trees
- Ensemble: Random Forest, Gradient Boosting, AdaBoost, Bagging
- SVM: Support Vector Classification/Regression, Kernel Ridge
- Gaussian Processes: GP Regression & Classification
- Clustering: K-Means, DBSCAN, Hierarchical
- Dimensionality Reduction: PCA, LDA
- Statistical Tests: t-tests, ANOVA, chi-square, KS test
- Metrics: All standard classification & regression metrics
- Model Selection: train_test_split, cross_validate, GridSearchCV
"""

from neurova.ml.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    LabelEncoder,
    OneHotEncoder,
    polynomial_features,
)

from neurova.ml.imputation import (
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    SimpleImputer,
    KNNImputer,
)

from neurova.ml.feature_selection import (
    SelectKBest,
    SelectPercentile,
    RFE,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
)

from neurova.ml.classification import (
    KNearestNeighbors,
    LogisticRegression,
    NaiveBayes,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from neurova.ml.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)

from neurova.ml.svm import (
    SVC,
    SVR,
    KernelRidge,
)

from neurova.ml.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessClassifier,
)

from neurova.ml.stats import (
    ttest_ind,
    ttest_1samp,
    ttest_rel,
    f_oneway,
    chi2_contingency,
    kstest,
    TTestResult,
    FTestResult,
    ChiSquareResult,
)

from neurova.ml.clustering import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift,
    SpectralClustering,
    OPTICS,
    GaussianMixture,
    MiniBatchKMeans,
    Birch,
)

from neurova.ml.dimensionality import (
    PCA,
    LDA,
    TSNE,
    Isomap,
    MDS,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    TruncatedSVD,
    FactorAnalysis,
    NMF,
    KernelPCA,
)

from neurova.ml.regression import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    PolynomialFeatures,
    BayesianRidge,
    HuberRegressor,
    QuantileRegressor,
    Lars,
    OrthogonalMatchingPursuit,
)

from neurova.ml.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from neurova.ml.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    GroupKFold,
    LeaveOneOut,
    LeaveOneGroupOut,
    ShuffleSplit,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

from neurova.ml.pipeline import (
    Pipeline,
    ColumnTransformer,
    FeatureUnion,
    FunctionTransformer,
    make_pipeline,
    make_union,
)

__all__ = [
    # preprocessing
    'StandardScaler',
    'MinMaxScaler',
    'Normalizer',
    'LabelEncoder',
    'OneHotEncoder',
    'polynomial_features',
    # advanced Scaling & Transforms
    'RobustScaler',
    'MaxAbsScaler',
    'QuantileTransformer',
    'PowerTransformer',
    # imputation
    'SimpleImputer',
    'KNNImputer',
    # feature Selection
    'SelectKBest',
    'SelectPercentile',
    'RFE',
    'VarianceThreshold',
    'f_classif',
    'f_regression',
    'mutual_info_classif',
    # classification
    'KNearestNeighbors',
    'LogisticRegression',
    'NaiveBayes',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    # ensemble Methods
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'AdaBoostClassifier',
    'BaggingClassifier',
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'AdaBoostRegressor',
    'BaggingRegressor',
    # sVM
    'SVC',
    'SVR',
    'KernelRidge',
    # gaussian Processes
    'GaussianProcessRegressor',
    'GaussianProcessClassifier',
    # statistical Tests
    'ttest_ind',
    'ttest_1samp',
    'ttest_rel',
    'f_oneway',
    'chi2_contingency',
    'kstest',
    'TTestResult',
    'FTestResult',
    'ChiSquareResult',
    # clustering
    'KMeans',
    'DBSCAN',
    'AgglomerativeClustering',
    'MeanShift',
    'SpectralClustering',
    'OPTICS',
    'GaussianMixture',
    'MiniBatchKMeans',
    'Birch',
    # dimensionality Reduction
    'PCA',
    'LDA',
    'TSNE',
    'Isomap',
    'MDS',
    'LocallyLinearEmbedding',
    'SpectralEmbedding',
    'TruncatedSVD',
    'FactorAnalysis',
    'NMF',
    'KernelPCA',
    # regression
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'PolynomialFeatures',
    'BayesianRidge',
    'HuberRegressor',
    'QuantileRegressor',
    'Lars',
    'OrthogonalMatchingPursuit',
    # metrics
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    # model Selection
    'train_test_split',
    'cross_validate',
    'GridSearchCV',
    'RandomizedSearchCV',
    'KFold',
    'StratifiedKFold',
    'GroupKFold',
    'LeaveOneOut',
    'LeaveOneGroupOut',
    'ShuffleSplit',
    'StratifiedShuffleSplit',
    'TimeSeriesSplit',
    # Pipeline
    'Pipeline',
    'ColumnTransformer',
    'FeatureUnion',
    'FunctionTransformer',
    'make_pipeline',
    'make_union',
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.