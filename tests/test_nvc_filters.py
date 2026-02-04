# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Self-test for Neurova computer vision primitives.

This script is intentionally dependency-light and can be run with:
  python3 test_cv.py

It uses basic assert checks rather than pytest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import neurova as nv


def _banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    _banner("NEUROVA CV PRIMITIVES - SELF TEST")

    print("Test 1: Kernels")
    k = nv.filters.gaussian_kernel(5, sigma=1.0)
    assert k.shape == (5, 5)
    assert np.isclose(k.sum(), 1.0, atol=1e-12)
    print("  gaussian_kernel: OK")

    print("Test 2: Blur")
    img = np.zeros((7, 7), dtype=np.float64)
    img[3, 3] = 1.0
    out = nv.filters.box_blur(img, 3, border_mode="constant", constant_value=0.0)
    assert np.isclose(out[3, 3], 1.0 / 9.0)
    assert np.isclose(out[2:5, 2:5].sum(), 1.0)
    print("  box_blur: OK")

    print("Test 3: Edges")
    ramp = np.tile(np.arange(0, 16, dtype=np.float64), (16, 1))
    gx, gy = nv.filters.sobel(ramp)
    assert gx.shape == ramp.shape and gy.shape == ramp.shape
    assert np.abs(gy[1:-1, 1:-1]).max() < 1e-9
    assert np.abs(gx[1:-1, 1:-1]).mean() > 0.0

    sq = np.zeros((64, 64), dtype=np.uint8)
    sq[16:48, 16:48] = 255
    edges = nv.filters.canny(sq, low_threshold=20.0, high_threshold=40.0, sigma=1.0)
    assert edges.dtype == np.uint8
    assert int(edges.sum()) > 0
    print("  sobel: OK")
    print("  canny: OK")

    print("Test 4: Thresholding")
    bimodal = np.zeros((64, 64), dtype=np.uint8)
    bimodal[:, :32] = 30
    bimodal[:, 32:] = 220
    t = nv.segmentation.otsu_threshold(bimodal)
    used_t, th = nv.segmentation.apply_threshold(
        bimodal,
        0.0,
        max_value=255,
        method=nv.core.ThresholdMethod.OTSU,
    )
    assert used_t == t
    assert int(th[:, :32].max()) == 0
    assert int(th[:, 32:].min()) == 255
    print("  otsu_threshold + threshold(OTSU): OK")

    print("Test 5: Morphology")
    noise = np.zeros((9, 9), dtype=np.uint8)
    noise[4, 4] = 255
    se = nv.morphology.structuring_element(nv.core.KernelShape.RECT, 3)
    opened = nv.morphology.binary_open(noise, se)
    assert int(opened.sum()) == 0
    print("  binary_open: OK")

    print("Test 6: Transform")
    img = np.zeros((32, 48), dtype=np.uint8)
    img[8:24, 16:32] = 255
    rot = nv.transform.rotate(img, 15.0, keep_size=True)
    assert rot.shape == img.shape
    warp = nv.transform.warp_affine(
        img,
        np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 5.0]], dtype=np.float64),
        (img.shape[1], img.shape[0]),
    )
    assert warp.shape == img.shape
    print("  rotate + warp_affine: OK")

    print("Test 7: Features")
    sq = np.zeros((64, 64), dtype=np.uint8)
    sq[16:48, 16:48] = 255
    corners = nv.features.detect_corners(sq, method="shi_tomasi", max_corners=20, min_distance=10)
    assert corners.shape[1] == 2
    assert corners.shape[0] >= 4
    kps = nv.features.detect_keypoints(sq, method="harris", max_keypoints=50, min_distance=10)
    assert len(kps) > 0
    print("  detect_corners + detect_keypoints: OK")

    print("Test 8: Drawing")
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    out = nv.utils.draw_rectangle(canvas, (5, 5), (26, 20), color=(255, 0, 0), thickness=2)
    assert out.shape == canvas.shape
    assert int(out.sum()) > 0
    out2 = nv.utils.draw_keypoints(out, kps[:10])
    assert out2.shape == canvas.shape
    print("  draw_rectangle + draw_keypoints: OK")

    print("Test 9: I/O round-trip")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rt.png"
        img = (np.random.rand(20, 21, 3) * 255).astype(np.uint8)
        ok = nv.io.imwrite(p, img)
        assert ok is True
        out = nv.io.read_image(p).data
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        assert int(np.abs(out.astype(int) - img.astype(int)).max()) == 0

        # optional Pillow-based JPEG path (skip if Pillow isn't installed).
        try:
            import PIL  # noqa: F401

            pj = Path(td) / "rt.jpg"
            nv.io.write_image(pj, img)
            outj = nv.io.read_image(pj).data
            assert outj.shape[0] == img.shape[0]
            assert outj.shape[1] == img.shape[1]
        except Exception:
            pass
    print("  write_image/read_image (png): OK")

    _banner("ALL CV TESTS PASSED")


if __name__ == "__main__":
    main()
