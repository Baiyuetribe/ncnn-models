# hybridnets

## Input --> Output

![](https://github.com/datvuthanh/HybridNets/raw/1aa5dd3783e3760d88e372079ec5f07907e9406e/images/hybridnets.jpg)
![](https://github.com/datvuthanh/HybridNets/raw/1aa5dd3783e3760d88e372079ec5f07907e9406e/images/full_video.gif)

## Convert 

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
model.eval()
# # inference
# x = torch.randn(1, 3, 640, 384)
# # 1. pt ---> onnx
# torch_out = torch.onnx._export(model, x, "hybridnets.onnx", export_params=True, opset_version=11)

# # 2. onnx --> onnxsim
# os.system("python3 -m onnxsim hybridnets.onnx sim.onnx")

# # 3. onnx --> ncnn
# os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# # 4. ncnn --> optmize ---> ncnn
# os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16


# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 640, 384), strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,640,384]  device=cpu")    # 可能错误

# 3. ncnn ---> optmize ----> ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```

两种转换均已失败告终，报错内容如下：
```
onnx模式下报错：
Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
pnnx模式下：
[ CPUFloatType{1,46035,4} ]) of traced region did not have observable data dependence with trace inputs; this probably indicates your program cannot be understood by the tracer.
```

## Example project

  
## Reference

- [datvuthanh/HybridNets](https://github.com/datvuthanh/HybridNets)


