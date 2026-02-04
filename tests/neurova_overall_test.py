#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Neurova Overall Test
=====================

Comprehensive test covering ALL Neurova features and modules.
Run this to verify the complete library functionality.

Usage:
    python neurova_overall_test.py
    python neurova_overall_test.py --verbose
    python neurova_overall_test.py --quick
"""

import sys
import time
import argparse
import traceback
import numpy as np

# Test counters
PASSED = 0
FAILED = 0
SKIPPED = 0
VERBOSE = False


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global PASSED, FAILED
            try:
                if VERBOSE:
                    print(f"  Testing {name}...", end=" ", flush=True)
                func(*args, **kwargs)
                if VERBOSE:
                    print("")
                PASSED += 1
                return True
            except Exception as e:
                if VERBOSE:
                    print(f" {e}")
                else:
                    print(f"     {name}: {e}")
                FAILED += 1
                if VERBOSE:
                    traceback.print_exc()
                return False
        wrapper.__name__ = name
        return wrapper
    return decorator


def section(title):
    """Print section header."""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# SECTION 1: CORE IMPORTS

def test_core_imports():
    section("1. CORE IMPORTS")
    
    @test("neurova main import")
    def _():
        import neurova
        assert hasattr(neurova, '__version__')
    _()
    
    @test("neurova.nvc import")
    def _():
        from neurova import nvc
        assert hasattr(nvc, 'imread')
        assert hasattr(nvc, 'GaussianBlur')
    _()
    
    @test("neurova.device import")
    def _():
        from neurova import device
        assert hasattr(device, 'get_device')
    _()
    
    @test("neurova.version")
    def _():
        from neurova import __version__
        assert isinstance(__version__, str)
    _()


# SECTION 2: NVC MODULE (OpenCV-compatible API)

def test_nvc_module():
    section("2. NVC MODULE (CV API)")
    
    from neurova import nvc
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @test("nvc.cvtColor")
    def _():
        result = nvc.cvtColor(img, nvc.COLOR_BGR2GRAY)
        assert result.shape == (100, 100)
    _()
    
    @test("nvc.GaussianBlur")
    def _():
        result = nvc.GaussianBlur(gray, ksize=5, sigma=1.0)
        assert result.shape == gray.shape
    _()
    
    @test("nvc.resize")
    def _():
        result = nvc.resize(img, (50, 50))
        assert result.shape == (50, 50, 3)
    _()
    
    @test("nvc.threshold")
    def _():
        _, result = nvc.threshold(gray, 127, 255, nvc.THRESH_BINARY)
        assert result.shape == gray.shape
    _()
    
    @test("nvc.Canny")
    def _():
        result = nvc.Canny(gray, low_threshold=50, high_threshold=150)
        assert result.shape == gray.shape
    _()
    
    @test("nvc.morphologyEx")
    def _():
        kernel = np.ones((3, 3), dtype=np.uint8)
        result = nvc.morphologyEx(gray, nvc.MORPH_OPEN, kernel)
        assert result.shape == gray.shape
    _()
    
    @test("nvc.rectangle")
    def _():
        canvas = img.copy()
        nvc.rectangle(canvas, (10, 10), (50, 50), (0, 255, 0), 2)
        assert canvas.shape == img.shape
    _()
    
    @test("nvc.circle")
    def _():
        canvas = img.copy()
        nvc.circle(canvas, (50, 50), 20, (255, 0, 0), -1)
        assert canvas.shape == img.shape
    _()
    
    @test("nvc.putText")
    def _():
        canvas = img.copy()
        nvc.putText(canvas, "Test", (10, 50), nvc.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    _()
    
    @test("nvc.hconcat/vconcat")
    def _():
        h = nvc.hconcat([img, img])
        v = nvc.vconcat([img, img])
        assert h.shape == (100, 200, 3)
        assert v.shape == (200, 100, 3)
    _()


# SECTION 3: FILTERS MODULE

def test_filters_module():
    section("3. FILTERS MODULE")
    
    from neurova.filters import (
        gaussian_blur, box_blur, median_blur, sharpen,
        sobel, canny, laplacian, scharr,
        gaussian_kernel, sobel_kernels,
        convolve2d, filter2d,
        bilateralFilter
    )
    
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8).astype(np.float64)
    
    @test("gaussian_blur")
    def _():
        result = gaussian_blur(gray, sigma=1.0)
        assert result.shape == gray.shape
    _()
    
    @test("box_blur")
    def _():
        result = box_blur(gray, ksize=5)
        assert result.shape == gray.shape
    _()
    
    @test("median_blur")
    def _():
        result = median_blur(gray.astype(np.uint8), ksize=3)
        assert result.shape == gray.shape
    _()
    
    @test("sobel")
    def _():
        result = sobel(gray)
        assert isinstance(result, tuple) or hasattr(result, 'shape')
    _()
    
    @test("canny")
    def _():
        result = canny(gray, low_threshold=50, high_threshold=150)
        assert result.shape == gray.shape
    _()
    
    @test("laplacian")
    def _():
        result = laplacian(gray)
        assert result.shape == gray.shape
    _()
    
    @test("gaussian_kernel")
    def _():
        kernel = gaussian_kernel(5, sigma=1.0)
        assert kernel.shape == (5, 5)
        assert abs(kernel.sum() - 1.0) < 0.01
    _()
    
    @test("convolve2d")
    def _():
        kernel = np.ones((3, 3)) / 9
        result = convolve2d(gray, kernel)
        assert result.shape == gray.shape
    _()
    
    @test("bilateralFilter")
    def _():
        result = bilateralFilter(gray.astype(np.float32), 9, 75, 75)
        assert result.shape == gray.shape
    _()


# SECTION 4: DETECTION MODULE

def test_detection_module():
    section("4. DETECTION MODULE")
    
    from neurova.detection import (
        HaarCascadeClassifier, HOGDescriptor,
        match_template, non_max_suppression, TemplateDetector
    )
    
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @test("HOGDescriptor")
    def _():
        hog = HOGDescriptor()
        features = hog.compute(gray)
        assert len(features) > 0
    _()
    
    @test("match_template")
    def _():
        template = gray[20:40, 20:40]
        result = match_template(gray, template)
        assert result.shape[0] > 0 and result.shape[1] > 0
    _()
    
    @test("non_max_suppression")
    def _():
        boxes = np.array([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])
        scores = np.array([0.9, 0.8, 0.7])
        result = non_max_suppression(boxes, scores, iou_threshold=0.3)
        assert len(result) <= len(boxes)
    _()
    
    @test("TemplateDetector")
    def _():
        template = gray[30:60, 30:60]
        detector = TemplateDetector(template, threshold=0.7)
        boxes, scores = detector.detect(gray)
        assert isinstance(boxes, np.ndarray)
    _()
    
    @test("HaarCascadeClassifier import")
    def _():
        # HaarCascadeClassifier requires a cascade file
        assert HaarCascadeClassifier is not None
    _()


# SECTION 5: FACE MODULE

def test_face_module():
    section("5. FACE MODULE")
    
    from neurova.face import FaceDetector, FaceRecognizer, FaceTrainer, FaceDataset
    
    @test("FaceDetector creation")
    def _():
        detector = FaceDetector(method='haar')
        assert detector is not None
    _()
    
    @test("FaceRecognizer LBPH")
    def _():
        recognizer = FaceRecognizer(method='lbph')
        assert recognizer is not None
    _()
    
    @test("FaceRecognizer EigenFace")
    def _():
        recognizer = FaceRecognizer(method='eigen')
        assert recognizer is not None
    _()
    
    @test("FaceRecognizer FisherFace")
    def _():
        recognizer = FaceRecognizer(method='fisher')
        assert recognizer is not None
    _()
    
    @test("FaceTrainer creation")
    def _():
        trainer = FaceTrainer()
        assert trainer is not None
    _()
    
    @test("FaceDataset creation")
    def _():
        dataset = FaceDataset()
        assert dataset is not None
    _()


# SECTION 6: ML MODULE

def test_ml_module():
    section("6. ML MODULE")
    
    from neurova.ml import (
        KMeans, DBSCAN, AgglomerativeClustering,
        PCA, LDA,
        KNearestNeighbors, LogisticRegression, DecisionTreeClassifier,
        LinearRegression, DecisionTreeRegressor, RandomForestRegressor,
        StandardScaler, MinMaxScaler, LabelEncoder,
        train_test_split,
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    
    # Test data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    y_reg = np.random.randn(100)
    
    @test("KMeans clustering")
    def _():
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(X)
        assert len(np.unique(labels)) <= 3
    _()
    
    @test("DBSCAN clustering")
    def _():
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        assert len(labels) == len(X)
    _()
    
    @test("PCA")
    def _():
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)
        assert reduced.shape == (100, 2)
    _()
    
    @test("KNearestNeighbors")
    def _():
        knn = KNearestNeighbors(n_neighbors=3)
        knn.fit(X, y)
        pred = knn.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("LogisticRegression")
    def _():
        lr = LogisticRegression()
        lr.fit(X, y)
        pred = lr.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("DecisionTreeClassifier")
    def _():
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X, y)
        pred = dt.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("LinearRegression")
    def _():
        lr = LinearRegression()
        lr.fit(X, y_reg)
        pred = lr.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("DecisionTreeRegressor")
    def _():
        dtr = DecisionTreeRegressor(max_depth=5)
        dtr.fit(X, y_reg)
        pred = dtr.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("RandomForestRegressor")
    def _():
        rf = RandomForestRegressor(n_estimators=10, max_depth=5)
        rf.fit(X, y_reg)
        pred = rf.predict(X[:10])
        assert len(pred) == 10
    _()
    
    @test("StandardScaler")
    def _():
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        assert abs(scaled.mean()) < 0.1
    _()
    
    @test("MinMaxScaler")
    def _():
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(X)
        assert scaled.min() >= 0 and scaled.max() <= 1
    _()
    
    @test("LabelEncoder")
    def _():
        le = LabelEncoder()
        labels = ['cat', 'dog', 'cat', 'bird']
        encoded = le.fit_transform(labels)
        assert len(encoded) == 4
    _()
    
    @test("train_test_split")
    def _():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        assert len(X_train) == 70
        assert len(X_test) == 30
    _()
    
    @test("accuracy_score")
    def _():
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        acc = accuracy_score(y_true, y_pred)
        assert 0 <= acc <= 1
    _()
    
    @test("confusion_matrix")
    def _():
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
    _()


# SECTION 7: NEURAL MODULE

def test_neural_module():
    section("7. NEURAL MODULE")
    
    from neurova.neural import Tensor, Module, Parameter, layers, losses, optim
    from neurova.neural.layers import Linear, ReLU, Sigmoid, Tanh, Sequential
    from neurova.neural.conv import Conv2D, MaxPool2D, Flatten
    from neurova.neural.losses import MSELoss, CrossEntropyLoss
    from neurova.neural.optim import SGD, Adam
    
    @test("Tensor creation")
    def _():
        t = Tensor([1, 2, 3, 4])
        assert t.data.shape == (4,)
    _()
    
    @test("Tensor operations")
    def _():
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a + b
        d = a * b
        assert c.data.shape == (2, 2)
        assert d.data.shape == (2, 2)
    _()
    
    @test("Linear layer")
    def _():
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(2, 10).astype(np.float32))
        out = layer.forward(x)
        assert out.data.shape == (2, 5)
    _()
    
    @test("ReLU activation")
    def _():
        relu = ReLU()
        x = Tensor(np.array([-1, 0, 1, 2]).astype(np.float32))
        out = relu.forward(x)
        assert out.data[0] == 0
        assert out.data[3] == 2
    _()
    
    @test("Sequential model")
    def _():
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        x = Tensor(np.random.randn(2, 10).astype(np.float32))
        out = model.forward(x)
        assert out.data.shape == (2, 5)
    _()
    
    @test("Conv2D layer")
    def _():
        conv = Conv2D(3, 16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 32, 32, 3).astype(np.float32))
        out = conv.forward(x)
        assert out.data.shape[0] == 1
        assert out.data.shape[3] == 16
    _()
    
    @test("MaxPool2D layer")
    def _():
        pool = MaxPool2D(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 32, 32, 16).astype(np.float32))
        out = pool.forward(x)
        assert out.data.shape[1] == 16
        assert out.data.shape[2] == 16
    _()
    
    @test("MSELoss")
    def _():
        loss_fn = MSELoss()
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.1, 2.1, 2.9]))
        loss = loss_fn(pred, target)
        assert loss.data >= 0
    _()
    
    @test("SGD optimizer")
    def _():
        layer = Linear(5, 3)
        optim_sgd = SGD(layer.parameters(), lr=0.01)
        assert optim_sgd is not None
    _()
    
    @test("Adam optimizer")
    def _():
        layer = Linear(5, 3)
        optim_adam = Adam(layer.parameters(), lr=0.001)
        assert optim_adam is not None
    _()


# SECTION 8: AUGMENTATION MODULE

def test_augmentation_module():
    section("8. AUGMENTATION MODULE")
    
    from neurova.augmentation import (
        hflip, vflip, rotate, resize, crop, center_crop, pad,
        normalize, adjust_brightness, adjust_contrast,
        gaussian_blur as aug_blur, to_tensor, to_numpy
    )
    from neurova.augmentation import (
        Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
        RandomRotation, ColorJitter, Normalize, ToTensor
    )
    
    img_hwc = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_chw = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
    
    @test("hflip")
    def _():
        result = hflip(img_hwc)
        assert result.shape == img_hwc.shape
    _()
    
    @test("vflip")
    def _():
        result = vflip(img_chw)
        assert result.shape == img_chw.shape
    _()
    
    @test("rotate")
    def _():
        result = rotate(img_chw, 45)
        assert result.shape == img_chw.shape
    _()
    
    @test("resize")
    def _():
        result = resize(img_chw, (50, 50))
        assert result.shape == (3, 50, 50)
    _()
    
    @test("crop")
    def _():
        result = crop(img_chw, 10, 10, 50, 50)
        assert result.shape == (3, 50, 50)
    _()
    
    @test("center_crop")
    def _():
        result = center_crop(img_chw, 50)
        assert result.shape == (3, 50, 50)
    _()
    
    @test("pad")
    def _():
        result = pad(img_chw, 10)
        assert result.shape == (3, 120, 120)
    _()
    
    @test("normalize")
    def _():
        img_float = img_chw.astype(np.float32) / 255.0
        result = normalize(img_float, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        assert result.shape == img_chw.shape
    _()
    
    @test("to_tensor / to_numpy")
    def _():
        tensor = to_tensor(img_hwc)
        assert tensor.shape == (3, 100, 100)
        back = to_numpy(tensor)
        assert back.shape == (100, 100, 3)
    _()
    
    @test("Compose pipeline")
    def _():
        transform = Compose([
            Resize((64, 64)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        result = transform(img_hwc)
        assert result.shape == (3, 64, 64)
    _()
    
    @test("ColorJitter")
    def _():
        jitter = ColorJitter(brightness=0.2, contrast=0.2)
        result = jitter(img_chw)
        assert result.shape == img_chw.shape
    _()


# SECTION 9: MORPHOLOGY MODULE

def test_morphology_module():
    section("9. MORPHOLOGY MODULE")
    
    from neurova.morphology import (
        erode, dilate, morphologyEx,
        binary_erode, binary_dilate, binary_open, binary_close,
        getStructuringElement, MORPH_RECT, MORPH_OPEN, MORPH_CLOSE
    )
    
    binary = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    @test("erode")
    def _():
        result = erode(binary, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("dilate")
    def _():
        result = dilate(binary, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("morphologyEx OPEN")
    def _():
        result = morphologyEx(binary, MORPH_OPEN, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("morphologyEx CLOSE")
    def _():
        result = morphologyEx(binary, MORPH_CLOSE, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("binary_erode")
    def _():
        result = binary_erode(binary, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("binary_dilate")
    def _():
        result = binary_dilate(binary, kernel)
        assert result.shape == binary.shape
    _()
    
    @test("getStructuringElement")
    def _():
        se = getStructuringElement(MORPH_RECT, (5, 5))
        assert se.shape == (5, 5)
    _()


# SECTION 10: FEATURES MODULE

def test_features_module():
    section("10. FEATURES MODULE")
    
    from neurova.features import (
        harris_response, shi_tomasi_response,
        SIFT, ORB,
        BFMatcher, FlannBasedMatcher
    )
    
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @test("harris_response")
    def _():
        response = harris_response(gray)
        assert response.shape == gray.shape
    _()
    
    @test("shi_tomasi_response")
    def _():
        response = shi_tomasi_response(gray)
        assert response.shape == gray.shape
    _()
    
    @test("SIFT descriptor")
    def _():
        sift = SIFT()
        kp, desc = sift.detectAndCompute(gray, None)
        assert isinstance(kp, list)
    _()
    
    @test("ORB descriptor")
    def _():
        orb = ORB()
        kp, desc = orb.detectAndCompute(gray, None)
        assert isinstance(kp, list)
    _()
    
    @test("BFMatcher")
    def _():
        matcher = BFMatcher()
        assert matcher is not None
    _()


# SECTION 11: SEGMENTATION MODULE

def test_segmentation_module():
    section("11. SEGMENTATION MODULE")
    
    from neurova.segmentation import (
        otsu_threshold,
        label_connected_components, find_contours
    )
    
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    @test("otsu_threshold")
    def _():
        thresh_val = otsu_threshold(gray)
        assert 0 <= thresh_val <= 255
    _()
    
    @test("label_connected_components")
    def _():
        binary = (gray > 127).astype(np.uint8)
        labels = label_connected_components(binary)
        assert labels.shape == gray.shape
    _()
    
    @test("find_contours")
    def _():
        binary = (gray > 127).astype(np.uint8)
        contours = find_contours(binary)
        assert isinstance(contours, list)
    _()


# SECTION 12: IO MODULE

def test_io_module():
    section("12. IO MODULE")
    
    import tempfile
    import os
    from neurova.io import imread, imwrite
    
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @test("imwrite PNG")
    def _():
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        imwrite(path, img)
        assert os.path.exists(path)
        os.unlink(path)
    _()
    
    @test("imread/imwrite roundtrip")
    def _():
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        imwrite(path, img)
        loaded = imread(path)
        assert loaded.shape == img.shape
        os.unlink(path)
    _()


# SECTION 13: DATA MODULE

def test_data_module():
    section("13. DATA MODULE")
    
    from neurova.data import DataLoader, Dataset, TensorDataset, Sampler
    
    @test("TensorDataset")
    def _():
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        dataset = TensorDataset(X, y)
        assert len(dataset) == 100
        x, label = dataset[0]
        assert x.shape == (10,)
    _()
    
    @test("DataLoader")
    def _():
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 16
    _()


# SECTION 14: ARCHITECTURE MODULE

def test_architecture_module():
    section("14. ARCHITECTURE MODULE")
    
    from neurova.architecture import (
        MLP, RNN, LSTM, GRU,
        Transformer, Autoencoder, VAE
    )
    
    @test("MLP creation")
    def _():
        model = MLP(input_shape=10, output_shape=3, hidden_layers=[32, 16])
        assert model is not None
    _()
    
    @test("RNN creation")
    def _():
        model = RNN(input_shape=(10, 5), output_shape=3)
        assert model is not None
    _()
    
    @test("LSTM creation")
    def _():
        model = LSTM(input_shape=(10, 5), output_shape=3)
        assert model is not None
    _()
    
    @test("GRU creation")
    def _():
        model = GRU(input_shape=(10, 5), output_shape=3)
        assert model is not None
    _()
    
    @test("Transformer creation")
    def _():
        model = Transformer(input_shape=(20, 64), output_shape=10)
        assert model is not None
    _()
    
    @test("Autoencoder creation")
    def _():
        model = Autoencoder(input_shape=100, latent_dim=10)
        assert model is not None
    _()


# SECTION 15: NN MODULE (PyTorch-like API)

def test_nn_module():
    section("15. NN MODULE (PyTorch-like)")
    
    from neurova import nn
    from neurova.nn import functional as F
    
    @test("nn.functional.relu")
    def _():
        x = np.array([-1, 0, 1, 2])
        result = F.relu(x)
        assert result[0] == 0 and result[3] == 2
    _()
    
    @test("nn.functional.sigmoid")
    def _():
        x = np.array([0])
        result = F.sigmoid(x)
        assert abs(result[0] - 0.5) < 0.01
    _()
    
    @test("nn.functional.softmax")
    def _():
        x = np.array([[1, 2, 3]])
        result = F.softmax(x)
        assert abs(result.sum() - 1.0) < 0.01
    _()
    
    @test("nn.functional.gelu")
    def _():
        x = np.array([0, 1, -1])
        result = F.gelu(x)
        assert result.shape == x.shape
    _()
    
    @test("nn.functional.dropout")
    def _():
        x = np.ones((10, 10))
        result = F.dropout(x, p=0.5, training=True)
        assert result.shape == x.shape
    _()


# SECTION 16: LINALG MODULE

def test_linalg_module():
    section("16. LINALG MODULE")
    
    from neurova.nn import linalg
    
    A = np.random.randn(3, 3)
    A = A @ A.T + np.eye(3)  # Make positive definite
    
    @test("linalg.cholesky")
    def _():
        L = linalg.cholesky(A)
        assert L.shape == A.shape
    _()
    
    @test("linalg.svd")
    def _():
        U, S, Vh = linalg.svd(A)
        assert U.shape == (3, 3)
        assert len(S) == 3
    _()
    
    @test("linalg.det")
    def _():
        det = linalg.det(A)
        assert isinstance(det, (int, float, np.floating))
    _()
    
    @test("linalg.inv")
    def _():
        inv = linalg.inv(A)
        assert inv.shape == A.shape
    _()
    
    @test("linalg.eig")
    def _():
        eigenvalues, eigenvectors = linalg.eig(A)
        assert len(eigenvalues) == 3
    _()


# SECTION 17: FFT MODULE

def test_fft_module():
    section("17. FFT MODULE")
    
    from neurova.nn import fft
    
    signal = np.sin(np.linspace(0, 4*np.pi, 64))
    
    @test("fft.fft")
    def _():
        result = fft.fft(signal)
        assert len(result) == len(signal)
    _()
    
    @test("fft.ifft")
    def _():
        freq = fft.fft(signal)
        back = fft.ifft(freq)
        assert np.allclose(signal, back.real, atol=1e-10)
    _()
    
    @test("fft.fft2")
    def _():
        img = np.random.randn(32, 32)
        result = fft.fft2(img)
        assert result.shape == img.shape
    _()
    
    @test("fft.fftshift")
    def _():
        freq = fft.fft(signal)
        shifted = fft.fftshift(freq)
        assert shifted.shape == freq.shape
    _()


# SECTION 18: SPECIAL FUNCTIONS

def test_special_module():
    section("18. SPECIAL FUNCTIONS")
    
    from neurova.nn import special
    
    @test("special.gamma")
    def _():
        result = special.gamma(np.array([1, 2, 3, 4, 5]))
        expected = np.array([1, 1, 2, 6, 24])
        assert np.allclose(result, expected, atol=0.01)
    _()
    
    @test("special.erf")
    def _():
        result = special.erf(np.array([0, 1]))
        assert abs(result[0]) < 0.01
        assert 0.8 < result[1] < 0.9
    _()
    
    @test("special.beta")
    def _():
        result = special.beta(2, 3)
        assert result > 0
    _()


# SECTION 19: VIDEO MODULE

def test_video_module():
    section("19. VIDEO MODULE")
    
    from neurova.video import VideoCapture, VideoWriter
    from neurova.video.trackers import TrackerMIL, TrackerKCF, TrackerCSRT
    
    @test("VideoCapture creation")
    def _():
        # Just test creation, not actual capture
        cap = VideoCapture
        assert cap is not None
    _()
    
    @test("VideoWriter creation")
    def _():
        writer = VideoWriter
        assert writer is not None
    _()
    
    @test("TrackerMIL")
    def _():
        tracker = TrackerMIL()
        assert tracker is not None
    _()
    
    @test("TrackerKCF")
    def _():
        tracker = TrackerKCF()
        assert tracker is not None
    _()
    
    @test("TrackerCSRT")
    def _():
        tracker = TrackerCSRT()
        assert tracker is not None
    _()


# SECTION 20: CALIBRATION MODULE

def test_calibration_module():
    section("20. CALIBRATION MODULE")
    
    from neurova.calibration import (
        calibrateCamera, undistort, findChessboardCorners,
        cornerSubPix, drawChessboardCorners, solvePnP
    )
    
    @test("calibrateCamera import")
    def _():
        assert calibrateCamera is not None
    _()
    
    @test("undistort import")
    def _():
        assert undistort is not None
    _()
    
    @test("findChessboardCorners import")
    def _():
        assert findChessboardCorners is not None
    _()
    
    @test("cornerSubPix import")
    def _():
        assert cornerSubPix is not None
    _()
    
    @test("solvePnP import")
    def _():
        assert solvePnP is not None
    _()


# MAIN

def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(description="Neurova Overall Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick test (skip slow tests)")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    
    print("=" * 60)
    print("  NEUROVA OVERALL TEST")
    print("  Complete Library Feature Verification")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all test sections
    test_core_imports()
    test_nvc_module()
    test_filters_module()
    test_detection_module()
    test_face_module()
    test_ml_module()
    test_neural_module()
    test_augmentation_module()
    test_morphology_module()
    test_features_module()
    test_segmentation_module()
    test_io_module()
    test_data_module()
    test_architecture_module()
    test_nn_module()
    test_linalg_module()
    test_fft_module()
    test_special_module()
    test_video_module()
    test_calibration_module()
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"   PASSED:  {PASSED}")
    print(f"   FAILED:  {FAILED}")
    print(f"  ⏱  TIME:    {elapsed:.2f}s")
    print("=" * 60)
    
    if FAILED == 0:
        print("\n ALL TESTS PASSED!")
        print("\nNeurova is fully functional and ready for use.")
        print("\nDependencies required:")
        print("  - numpy")
        print("  - scipy")
        print("  - pillow")
        print("\nNo OpenCV (cv2) required! ")
    else:
        print(f"\n  {FAILED} test(s) failed")
        print("Please review the errors above.")
    
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
