# LeNet-5 Local Demo (MNIST Digits)

This is a tiny local web demo that lets you draw on a canvas or upload an image and run your LeNet-5 inference for digits (0-9).

## Run

From the repo root:

```powershell
python -m app.run_demo
```

Then open `http://127.0.0.1:8000`.

## Notes

- The server loads weights from `LENET_WEIGHTS` (default: `weights.bin` in the repo root).
- To train and export `weights.bin`, build and run the CPU trainer:
  - Configure/build: `cmake -S . -B build` then `cmake --build build --config Release`
  - Train: `build\\Release\\lenet_cpu.exe --data <path-to-mnist-idx-files> --out weights.bin --epochs 1`
