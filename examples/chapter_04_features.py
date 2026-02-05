# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 4: Feature Extraction


This chapter covers:
- Histogram features
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- SIFT-like features
- Corner detection (Harris, Shi-Tomasi)
- Blob detection

Author: Neurova Team
"""

import numpy as np

print("")
print("Chapter 4: Feature Extraction")
print("")

import neurova as nv
from neurova import features, datasets, core

# load sample image from neurova for feature extraction demos
try:
    rgb_sample = datasets.load_sample_image('building')
    if rgb_sample.shape[2] == 4:  # BGRA to BGR
        rgb_sample = rgb_sample[:, :, :3]
    image = core.rgb2gray(rgb_sample).astype(np.uint8)
# resize if too large
    if image.shape[0] > 200 or image.shape[1] > 200:
        from neurova import transform
        image = transform.resize(image, (200, 200)).astype(np.uint8)
    print(f"Loaded 'building' sample image from Neurova")
except:
# fallback to structured test image
    np.random.seed(42)
    image = np.zeros((200, 200), dtype=np.uint8)
# add rectangles for structure
    image[30:70, 30:70] = 200
    image[100:160, 80:150] = 150
    image[50:100, 120:180] = 180
# add some gradient
    for i in range(200):
        image[:, i] = np.clip(image[:, i] + i // 4, 0, 255)
    print(f"Using synthetic structured image")

print(f"Sample image: shape={image.shape}")

# 4.1 histogram features
print(f"\n4.1 Histogram Features")

# compute histogram
hist, bins = np.histogram(image, bins=256, range=(0, 256))
print(f"    Histogram: {len(hist)} bins")
print(f"    Most common intensity: {np.argmax(hist)}")
print(f"    Mean intensity: {np.mean(image):.2f}")
print(f"    Std intensity: {np.std(image):.2f}")

# histogram equalization
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    return cdf_normalized[img].astype(np.uint8)

equalized = histogram_equalization(image)
print(f"    Equalized image: mean={np.mean(equalized):.2f}, std={np.std(equalized):.2f}")

# 4.2 hog features (histogram of oriented gradients)
print(f"\n4.2 HOG Features")

from neurova.detection.hog import HOGDescriptor

# create hog descriptor
hog = HOGDescriptor(
    winSize=(64, 128),
    blockSize=(16, 16),
    blockStride=(8, 8),
    cellSize=(8, 8),
    nbins=9
)

# get descriptor size
desc_size = hog.getDescriptorSize()
print(f"    HOG descriptor size: {desc_size}")
print(f"    Window size: {hog.winSize}")
print(f"    Block size: {hog.blockSize}")
print(f"    Cell size: {hog.cellSize}")
print(f"    Number of bins: {hog.nbins}")

# compute hog for a window
window = np.random.randint(0, 255, (128, 64), dtype=np.uint8)
hog_features = hog.compute(window)
if hog_features is not None:
    print(f"    HOG features shape: {hog_features.shape}")
else:
    print(f"    HOG features: computed internally")

# 4.3 lbp features (local binary patterns)
print(f"\n4.3 LBP Features")

def compute_lbp(image, radius=1, n_points=8):
    """Compute Local Binary Pattern."""
    h, w = image.shape
    lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = image[i, j]
            code = 0
            # 8-connected neighbors
            code |= (image[i-1, j-1] >= center) << 7
            code |= (image[i-1, j] >= center) << 6
            code |= (image[i-1, j+1] >= center) << 5
            code |= (image[i, j+1] >= center) << 4
            code |= (image[i+1, j+1] >= center) << 3
            code |= (image[i+1, j] >= center) << 2
            code |= (image[i+1, j-1] >= center) << 1
            code |= (image[i, j-1] >= center) << 0
            lbp[i-radius, j-radius] = code
    
    return lbp

lbp_image = compute_lbp(image)
print(f"    LBP image shape: {lbp_image.shape}")
print(f"    LBP values range: [{lbp_image.min()}, {lbp_image.max()}]")

# lbp histogram
lbp_hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
print(f"    LBP histogram: {len(lbp_hist)} bins")
print(f"    Most common pattern: {np.argmax(lbp_hist)}")

# 4.4 corner detection - harris
print(f"\n4.4 Harris Corner Detection")

def harris_corners(image, k=0.04, threshold=0.01):
    """Detect Harris corners."""
# compute gradients
    Ix = np.zeros_like(image, dtype=np.float64)
    Iy = np.zeros_like(image, dtype=np.float64)
    
    Ix[:, 1:-1] = (image[:, 2:].astype(float) - image[:, :-2].astype(float)) / 2
    Iy[1:-1, :] = (image[2:, :].astype(float) - image[:-2, :].astype(float)) / 2
    
# compute products
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
# gaussian window
    from scipy.ndimage import gaussian_filter
    Sxx = gaussian_filter(Ixx, sigma=1)
    Syy = gaussian_filter(Iyy, sigma=1)
    Sxy = gaussian_filter(Ixy, sigma=1)
    
# harris response
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k * trace ** 2
    
# threshold
    corners = R > threshold * R.max()
    
    return corners, R

try:
    corners, response = harris_corners(image)
    corner_count = np.sum(corners)
    print(f"    Harris corners detected: {corner_count}")
    print(f"    Response range: [{response.min():.2f}, {response.max():.2f}]")
except ImportError:
    print(f"    Harris corners: requires scipy")

# 4.5 gradient features
print(f"\n4.5 Gradient Features")

# compute gradients
gx = np.zeros_like(image, dtype=np.float32)
gy = np.zeros_like(image, dtype=np.float32)

gx[:, 1:-1] = (image[:, 2:].astype(float) - image[:, :-2].astype(float)) / 2
gy[1:-1, :] = (image[2:, :].astype(float) - image[:-2, :].astype(float)) / 2

# gradient magnitude and direction
magnitude = np.sqrt(gx**2 + gy**2)
direction = np.arctan2(gy, gx)

print(f"    Gradient magnitude: range=[{magnitude.min():.2f}, {magnitude.max():.2f}]")
print(f"    Gradient direction: range=[{direction.min():.2f}, {direction.max():.2f}] radians")
print(f"    Mean gradient strength: {np.mean(magnitude):.2f}")

# 4.6 statistical features
print(f"\n4.6 Statistical Features")

def extract_statistical_features(img):
    """Extract statistical features from image."""
    features = {
        'mean': np.mean(img),
        'std': np.std(img),
        'min': np.min(img),
        'max': np.max(img),
        'median': np.median(img),
        'skewness': float(np.mean(((img - np.mean(img)) / np.std(img)) ** 3)),
        'kurtosis': float(np.mean(((img - np.mean(img)) / np.std(img)) ** 4) - 3),
        'energy': float(np.sum(img.astype(float) ** 2)),
        'entropy': float(-np.sum(np.histogram(img, 256)[0] / img.size * 
                                 np.log2(np.histogram(img, 256)[0] / img.size + 1e-10))),
    }
    return features

stats = extract_statistical_features(image)
print(f"    Mean: {stats['mean']:.2f}")
print(f"    Std: {stats['std']:.2f}")
print(f"    Skewness: {stats['skewness']:.4f}")
print(f"    Kurtosis: {stats['kurtosis']:.4f}")
print(f"    Energy: {stats['energy']:.0f}")

# 4.7 texture features (glcm)
print(f"\n4.7 Texture Features (GLCM)")

def compute_glcm(image, distances=[1], angles=[0], levels=256):
    """Compute Gray Level Co-occurrence Matrix."""
    h, w = image.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    
    d = distances[0]
    angle = angles[0]
    
    dx = int(round(d * np.cos(angle)))
    dy = int(round(d * np.sin(angle)))
    
    for i in range(max(0, -dy), min(h, h - dy)):
        for j in range(max(0, -dx), min(w, w - dx)):
            glcm[image[i, j], image[i + dy, j + dx]] += 1
    
# normalize
    glcm /= glcm.sum() + 1e-10
    
    return glcm

# quantize image to fewer levels for glcm
quantized = (image // 16).astype(np.uint8)  # 16 levels
glcm = compute_glcm(quantized, levels=16)

# glcm features
contrast = np.sum(glcm * (np.arange(16)[:, None] - np.arange(16)[None, :]) ** 2)
energy = np.sum(glcm ** 2)
homogeneity = np.sum(glcm / (1 + np.abs(np.arange(16)[:, None] - np.arange(16)[None, :])))

print(f"    GLCM shape: {glcm.shape}")
print(f"    Contrast: {contrast:.4f}")
print(f"    Energy: {energy:.4f}")
print(f"    Homogeneity: {homogeneity:.4f}")

# 4.8 feature vector construction
print(f"\n4.8 Feature Vector Construction")

def extract_all_features(img):
    """Extract comprehensive feature vector from image."""
    features = []
    
# statistical features
    features.extend([
        np.mean(img), np.std(img), np.min(img), np.max(img), np.median(img)
    ])
    
    # Histogram features (binned)
    hist, _ = np.histogram(img, bins=16, range=(0, 256))
    features.extend(hist / hist.sum())
    
# gradient features
    gx = np.diff(img.astype(float), axis=1)
    gy = np.diff(img.astype(float), axis=0)
    features.extend([np.mean(np.abs(gx)), np.mean(np.abs(gy))])
    
    return np.array(features)

feature_vector = extract_all_features(image)
print(f"    Feature vector length: {len(feature_vector)}")
print(f"    Feature vector (first 10): {feature_vector[:10]}")

# summary
print("\n" + "=" * 60)
print("Chapter 4 Summary:")
print("   Computed histogram features")
print("   Extracted HOG features")
print("   Computed LBP texture features")
print("   Detected Harris corners")
print("   Computed gradient magnitude and direction")
print("   Extracted statistical features")
print("   Computed GLCM texture features")
print("   Constructed comprehensive feature vectors")
print("")
