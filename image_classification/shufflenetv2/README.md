# shufflenetv2

## Input --> Output

![](https://pytorch.org/assets/images/shufflenet_v2_1.png)

## Convert 

onnx -->  onnx-sim --> onnx2ncnn --> ncnnoptimize --> ncnn

```python
import os

# 1. download
# download_url = https://github.com/onnx/models/tree/main/vision/classification/shufflenet
os.system("")

# 2. onnx --> onnxsim
os.system("python -m onnxsim shufflenet-v2-12.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```

## Example project


## Reference

- [hub/pytorch_vision_shufflenet_v2/](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/)
- [shufflenet](https://github.com/onnx/models/tree/main/vision/classification/shufflenet)


