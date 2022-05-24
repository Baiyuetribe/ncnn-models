# mobilenet_v2

## Input --> Output

![](https://pytorch.org/assets/images/mobilenet_v2_2.png)

## Convert 

pt --> onnx --> onnx-sim --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()    # 下面两种方式都可以成功，推理结果也一样
# # inference
# x = torch.randn(1, 3, 224, 224)
# # 1. pt ---> onnx
# torch_out = torch.onnx._export(model, x, "mobilenet_v2.onnx", export_params=True)

# # 2. onnx --> onnxsim
# os.system("python3 -m onnxsim mobilenet_v2.onnx sim.onnx")

# # 3. onnx --> ncnn
# os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# # 4. ncnn --> optmize ---> ncnn
# os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16


# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 224, 224), strict=False)    # 最低要求224
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]  device=cpu")    # 可能错误

# 3. ncnn ---> optmize ----> ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```
两种转换都可以成功，且推理结果一样

## Example project


## Reference

- [pytorch_vision_mobilenet_v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)


