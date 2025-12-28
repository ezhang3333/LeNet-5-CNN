# LeNet-5 CNN

LeNet-5 CNN trained on the MNIST Digits Dataset 

## Run app

```powershell
$env:LENET_WEIGHTS="weights.bin
python -m app.run_demo
```

Open `http://127.0.0.1:8000`.

## Train model

```powershell
cmake -S . -B build
cmake --build build --config Release
build\app\cpp\lenet_cpu.exe --out weights.bin --epochs 1
```
