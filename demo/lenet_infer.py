from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


FASHION_MNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def _im2col_nopad_stride1(x_chw: np.ndarray, kernel_h: int, kernel_w: int) -> np.ndarray:
    # x_chw: (C, H, W)
    channels, height, width = x_chw.shape
    out_h = 1 + (height - kernel_h)
    out_w = 1 + (width - kernel_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel larger than input")

    s_c, s_h, s_w = x_chw.strides
    view = np.lib.stride_tricks.as_strided(
        x_chw,
        shape=(channels, out_h, out_w, kernel_h, kernel_w),
        strides=(s_c, s_h, s_w, s_h, s_w),
        writeable=False,
    )
    # (out_h*out_w, C*kernel_h*kernel_w)
    return view.transpose(1, 2, 0, 3, 4).reshape(out_h * out_w, channels * kernel_h * kernel_w)


def _conv_forward(x_vec: np.ndarray, in_c: int, in_h: int, in_w: int, out_c: int, k: int, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    # x_vec is channel-blocked vector (C*H*W,), each channel stored in row-major raster order.
    x_chw = x_vec.reshape(in_c, in_h, in_w)
    cols = _im2col_nopad_stride1(x_chw, k, k)  # (H_out*W_out, in_c*k*k)
    out = cols @ w  # (H_out*W_out, out_c)
    out += b.reshape(1, out_c)
    return out.astype(np.float32, copy=False).reshape(-1, order="F")


def _maxpool_forward(x_vec: np.ndarray, c: int, h: int, w: int, pool: int, stride: int) -> tuple[np.ndarray, int, int]:
    x = x_vec.reshape(c, h, w)
    out_h = 1 + int(np.ceil((h - pool) / stride))
    out_w = 1 + int(np.ceil((w - pool) / stride))
    out = np.full((c, out_h, out_w), np.finfo(np.float32).min, dtype=np.float32)

    for ch in range(c):
        for i_out in range(out_h):
            for j_out in range(out_w):
                start_i = i_out * stride
                start_j = j_out * stride
                end_i = min(start_i + pool, h)
                end_j = min(start_j + pool, w)
                if start_i >= h or start_j >= w:
                    continue
                out[ch, i_out, j_out] = np.max(x[ch, start_i:end_i, start_j:end_j])

    return out.reshape(-1), out_h, out_w


@dataclass(frozen=True)
class _Params:
    conv1_w: np.ndarray
    conv1_b: np.ndarray
    conv2_w: np.ndarray
    conv2_b: np.ndarray
    fc3_w: np.ndarray
    fc3_b: np.ndarray
    fc4_w: np.ndarray
    fc4_b: np.ndarray


def _load_weights(weights_path: Path) -> _Params:
    data = weights_path.read_bytes()
    off = 0

    def read_i32() -> int:
        nonlocal off
        (val,) = struct.unpack_from("<i", data, off)
        off += 4
        return int(val)

    def read_f32(n: int) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(data, dtype="<f4", count=n, offset=off)
        off += 4 * n
        return np.array(arr, dtype=np.float32, copy=True)

    n_layers = read_i32()
    layer_params: list[np.ndarray] = []
    for _ in range(n_layers):
        size = read_i32()
        layer_params.append(read_f32(size))

    if n_layers < 9:
        raise ValueError(f"Unexpected weights format: n_layers={n_layers}")

    conv1 = layer_params[0]
    conv2 = layer_params[3]
    fc3 = layer_params[6]
    fc4 = layer_params[8]

    if conv1.size != (49 * 4 + 4):
        raise ValueError(f"Unexpected conv1 param size: {conv1.size}")
    if conv2.size != (196 * 16 + 16):
        raise ValueError(f"Unexpected conv2 param size: {conv2.size}")
    if fc3.size != (1296 * 32 + 32):
        raise ValueError(f"Unexpected fc3 param size: {fc3.size}")
    if fc4.size != (32 * 10 + 10):
        raise ValueError(f"Unexpected fc4 param size: {fc4.size}")

    conv1_w = conv1[: 49 * 4].reshape((49, 4), order="F")
    conv1_b = conv1[49 * 4 :]
    conv2_w = conv2[: 196 * 16].reshape((196, 16), order="F")
    conv2_b = conv2[196 * 16 :]
    fc3_w = fc3[: 1296 * 32].reshape((1296, 32), order="F")
    fc3_b = fc3[1296 * 32 :]
    fc4_w = fc4[: 32 * 10].reshape((32, 10), order="F")
    fc4_b = fc4[32 * 10 :]

    return _Params(
        conv1_w=conv1_w,
        conv1_b=conv1_b,
        conv2_w=conv2_w,
        conv2_b=conv2_b,
        fc3_w=fc3_w,
        fc3_b=fc3_b,
        fc4_w=fc4_w,
        fc4_b=fc4_b,
    )


class FashionMnistLenet86:
    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path)
        self.params = _load_weights(self.weights_path)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("L").resize((86, 86), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)

        # Canvas drawings tend to be black strokes on white background; Fashion-MNIST is white foreground on black.
        if float(arr.mean()) > 127.0:
            arr = 255.0 - arr

        return arr.reshape(-1).astype(np.float32, copy=False)

    def forward_probs(self, x_vec: np.ndarray) -> np.ndarray:
        p = self.params

        x = _conv_forward(x_vec, in_c=1, in_h=86, in_w=86, out_c=4, k=7, w=p.conv1_w, b=p.conv1_b)
        x = _relu(x)
        x, h, w = _maxpool_forward(x, c=4, h=80, w=80, pool=2, stride=2)

        x = _conv_forward(x, in_c=4, in_h=h, in_w=w, out_c=16, k=7, w=p.conv2_w, b=p.conv2_b)
        x = _relu(x)
        x, h, w = _maxpool_forward(x, c=16, h=34, w=34, pool=4, stride=4)

        x = (p.fc3_w.T @ x.reshape(-1, 1)).reshape(-1) + p.fc3_b
        x = _relu(x)
        x = (p.fc4_w.T @ x.reshape(-1, 1)).reshape(-1) + p.fc4_b
        return _softmax(x)

    def predict(self, img: Image.Image) -> dict:
        x = self.preprocess(img)
        probs = self.forward_probs(x)
        idx = int(np.argmax(probs))
        return {
            "pred_index": idx,
            "pred_label": FASHION_MNIST_LABELS[idx],
            "probs": [float(p) for p in probs],
            "labels": FASHION_MNIST_LABELS,
        }


__all__ = ["FashionMnistLenet86", "FASHION_MNIST_LABELS"]

