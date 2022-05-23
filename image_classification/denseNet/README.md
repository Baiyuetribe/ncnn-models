# densenet121

## Input --> Output

![](https://pytorch.org/assets/images/densenet1.png)

## Convert 

pt --> onnx --> onnx-sim --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.eval()
x = torch.rand(1, 3, 224, 224)  # 最小224*224
# 1. pt ---> onnx
torch_out = torch.onnx._export(model, x, "densenet121.onnx", export_params=True)

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim densenet121.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```

## Example project


## Reference

- [pytorch_vision_densenet](https://pytorch.org/hub/pytorch_vision_densenet/)


