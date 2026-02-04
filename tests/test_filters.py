# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

import numpy as np

import neurova as nv


def test_gaussian_kernel_sums_to_one():
    k = nv.filters.kernels.gaussian_kernel(5, sigma=1.0)
    assert k.shape == (5, 5)
    assert np.isclose(k.sum(), 1.0, atol=1e-12)


def test_box_blur_on_impulse_center():
    img = np.zeros((7, 7), dtype=np.float64)
    img[3, 3] = 1.0

    out = nv.filters.blur.box_blur(img, 3, border_mode="constant", constant_value=0.0)

    # the 3x3 neighborhood should each get 1/9.
    assert np.isclose(out[3, 3], 1.0 / 9.0)
    assert np.isclose(out[2:5, 2:5].sum(), 1.0)


def test_sobel_on_horizontal_ramp():
    # horizontal ramp -> non-zero gx, near-zero gy.
    img = np.tile(np.arange(0, 16, dtype=np.float64), (16, 1))
    gx, gy = nv.filters.edges.sobel(img)
    assert gx.shape == img.shape
    assert gy.shape == img.shape

    # ignore borders. gy should be near 0.
    assert np.abs(gy[1:-1, 1:-1]).max() < 1e-9
    assert np.abs(gx[1:-1, 1:-1]).mean() > 0.0


def test_canny_edges_on_simple_square():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 255

    edges = nv.filters.edges.canny(img, low_threshold=20.0, high_threshold=40.0, sigma=1.0)
    assert edges.dtype == np.uint8
    assert edges.shape == img.shape

    # edges should exist.
    assert int(edges.sum()) > 0
