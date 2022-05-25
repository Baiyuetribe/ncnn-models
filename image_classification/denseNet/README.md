# densenet121

## Input --> Output

![](https://pytorch.org/assets/images/densenet1.png)

## Convert 

pt --> torchscript  --> pnnx --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.eval()
x = torch.rand(1, 3, 224, 224)
# 方法1： pnnx
# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")   # 2022年5月25日起，pnnx默认自动量化，不需要再次optmize

# # 方法2： onnx
# # 1. pt ---> onnx
# torch_out = torch.onnx._export(model, x, "densenet121.onnx", export_params=True)

# # 2. onnx --> onnxsim
# os.system("python3 -m onnxsim densenet121.onnx sim.onnx")

# # 3. onnx --> ncnn
# os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# # 4. ncnn --> optmize ---> ncnn
# os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16

# 两种转换都成功
```

## Example project


## Reference

- [pytorch_vision_densenet](https://pytorch.org/hub/pytorch_vision_densenet/)


