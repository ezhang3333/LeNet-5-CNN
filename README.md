# LeNet-5 CNN (Digits)

This repo is centered around the `app/` folder:

- `app/`: local web UI + Python inference (loads `weights.bin` by default)
- `app/cpp/`: CPU trainer (`lenet_cpu`) that can export `weights.bin` in the same binary format the Python demo loads
- `cnn/`: legacy / GPU + CUDA work kept for later

## Run the demo

```powershell
$env:LENET_WEIGHTS="weights.bin"   # optional (defaults to weights.bin, falls back to weights-86.bin)
python -m app.run_demo
```

Open `http://127.0.0.1:8000`.

## Train weights (CPU, C++)

Requires CMake + a C++ compiler:

```powershell
cmake -S . -B build
cmake --build build --config Release
build\Release\lenet_cpu.exe --data <path-to-mnist-idx-files> --out weights.bin --epochs 1
```

Notes:
- `--data` should point to a folder containing the 4 uncompressed MNIST IDX files:
  `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.
- If you put MNIST under `cnn/datasets/`, `lenet_cpu` will auto-detect it (or set `MNIST_DIR` / `LENET_DATA`).
