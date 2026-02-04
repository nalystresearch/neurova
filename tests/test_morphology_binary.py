# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

import numpy as np

import neurova as nv


def test_binary_erode_and_dilate_rect():
    img = np.zeros((7, 7), dtype=np.uint8)
    img[2:5, 2:5] = 255

    k = nv.morphology.structuring_element(nv.core.KernelShape.RECT, 3)

    er = nv.morphology.binary_erode(img, k)
    di = nv.morphology.binary_dilate(img, k)

    assert er.shape == img.shape
    assert di.shape == img.shape

    # erosion shrinks the square.
    assert int(er.sum()) < int(img.sum())
    # dilation expands it.
    assert int(di.sum()) > int(img.sum())


def test_binary_open_removes_isolated_pixel():
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 255

    k = nv.morphology.structuring_element(nv.core.KernelShape.RECT, 3)
    opened = nv.morphology.binary_open(img, k)
    assert int(opened.sum()) == 0
