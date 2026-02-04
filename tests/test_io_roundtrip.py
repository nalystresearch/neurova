# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

from pathlib import Path

import numpy as np

import neurova as nv


def test_png_roundtrip(tmp_path: Path):
    img = (np.random.rand(20, 21, 3) * 255).astype(np.uint8)
    p = tmp_path / "rt.png"

    ok = nv.io.imwrite(p, img)
    assert ok is True

    out = nv.io.read_image(p).data
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert int(np.abs(out.astype(int) - img.astype(int)).max()) == 0
