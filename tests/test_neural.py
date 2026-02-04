# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Self-test for Neurova neural primitives.

Run:
  python3 test_neural.py

This is intentionally dependency-light and avoids pytest.
"""

from __future__ import annotations

import math

import numpy as np

import neurova as nv


def _banner(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def test_linear_regression() -> None:
    print("Test 1: Linear regression (MSE + SGD)")
    rng = np.random.default_rng(0)

    # y = 3x + 2 with noise
    X = rng.normal(size=(256, 1)).astype(np.float32)
    y = (3.0 * X[:, 0] + 2.0 + 0.05 * rng.normal(size=(256,))).astype(np.float32)

    model = nv.neural.layers.Sequential(
        nv.neural.layers.Linear(1, 1, seed=0),
    )
    opt = nv.neural.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
    loss_fn = nv.neural.losses.MSELoss()

    x_t = nv.neural.tensor(X, requires_grad=False)

    prev = None
    for step in range(200):
        opt.zero_grad()
        pred = model(x_t).reshape(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step={step:3d} loss={float(loss.data):.6f}")
        if prev is not None:
            assert float(loss.data) <= prev + 1e-4
        prev = float(loss.data)

    layer0 = model.layers[0]
    assert isinstance(layer0, nv.neural.layers.Linear)
    w = layer0.weight.data[0, 0]
    assert layer0.bias is not None
    b = layer0.bias.data[0]
    print(f"  learned w={w:.3f} b={b:.3f}")
    assert abs(float(w) - 3.0) < 0.2
    assert abs(float(b) - 2.0) < 0.2
    print("  OK")


def test_classification() -> None:
    print("Test 2: Simple classification (CrossEntropy + Adam)")
    rng = np.random.default_rng(1)

    # two Gaussian blobs in 2D
    n = 400
    x0 = rng.normal(loc=(-1.0, -1.0), scale=0.6, size=(n // 2, 2)).astype(np.float32)
    x1 = rng.normal(loc=(+1.0, +1.0), scale=0.6, size=(n // 2, 2)).astype(np.float32)
    X = np.vstack([x0, x1])
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)

    perm = rng.permutation(n)
    X = X[perm]
    y = y[perm]

    model = nv.neural.layers.Sequential(
        nv.neural.layers.Linear(2, 16, seed=2),
        nv.neural.layers.ReLU(),
        nv.neural.layers.Linear(16, 2, seed=3),
    )

    opt = nv.neural.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nv.neural.losses.CrossEntropyLoss()

    x_t = nv.neural.tensor(X, requires_grad=False)

    for step in range(150):
        opt.zero_grad()
        logits = model(x_t)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step={step:3d} loss={float(loss.data):.6f}")

    logits = model(x_t).data
    pred = np.argmax(logits, axis=1)
    acc = (pred == y).mean()
    print(f"  accuracy={acc:.3f}")
    assert acc > 0.9
    print("  OK")


def main() -> None:
    _banner("NEUROVA NEURAL - SELF TEST")
    test_linear_regression()
    test_classification()
    print("=" * 60)
    print("ALL NEURAL TESTS PASSED")


if __name__ == "__main__":
    main()
