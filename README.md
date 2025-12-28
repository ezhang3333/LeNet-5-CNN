# LeNet-5 CNN

LeNet-5 CNN trained on the MNIST Digits Dataset.

## Requirements

- Python 3.10+
- numpy, pillow
- CMake + a C++17 compiler

## Run app on local host

From the repo root:

```powershell
python -m pip install numpy pillow

$env:LENET_WEIGHTS="weights.bin"
python -m app.run_demo
```

Open `http://127.0.0.1:8000`.

- More details in `app/README.md`.

## MNIST data

If you are interested in training on the dataset yourself, download the MNIST IDX files 
and put the 4 uncompressed files in `cnn/datasets/`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Train model

```powershell
cmake -S . -B build
cmake --build build --config Release

build\Release\lenet_cpu.exe --data cnn\datasets --out weights.bin --epochs 1
```
