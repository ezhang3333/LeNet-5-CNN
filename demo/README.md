# LeNet-5 Local Demo (Draw or Upload)

This is a tiny local web demo that lets you draw on a canvas or upload an image and run your LeNet-5 (Fashion-MNIST-86) inference.

## Run

From the repo root:

```powershell
python -m demo.run_demo
```

Then open `http://127.0.0.1:8000`.

## Notes

- The model loads `weights-86.bin` from the repo root.
- The model is trained for Fashion-MNIST classes (not digits).
