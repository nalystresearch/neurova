# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

import numpy as np

import neurova as nv


def test_otsu_threshold_bimodal():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, :32] = 30
    img[:, 32:] = 220

    t = nv.segmentation.threshold.otsu_threshold(img)
    assert 0.0 <= t <= 255.0

    used_t, out = nv.segmentation.threshold.threshold(
        img,
        0.0,
        max_value=255,
        method=nv.core.ThresholdMethod.OTSU,
    )
    assert used_t == t
    assert out.dtype == np.uint8
    assert int(out[:, :32].max()) == 0
    assert int(out[:, 32:].min()) == 255


def test_threshold_binary():
    img = np.array([[0, 50, 100, 150, 200]], dtype=np.uint8)
    t, out = nv.segmentation.threshold.threshold(img, 100, max_value=255, method=nv.core.ThresholdMethod.BINARY)
    assert t == 100.0
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0, 0, 255, 255]]
