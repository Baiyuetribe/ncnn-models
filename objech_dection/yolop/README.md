# yolop

## Input --> Output

![](https://github.com/hustvl/YOLOP/raw/main/pictures/detect.png)
![](https://github.com/hustvl/YOLOP/raw/main/pictures/da.png)

## Convert 

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()
# inference
x = torch.randn(1, 3, 640, 640)
# 1. pt ---> onnx
torch_out = torch.onnx._export(model, x, "yolop.onnx", export_params=True)

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim yolop.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16

# ==== pnnx方法
# # 1. pt --> torchscript
# traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 640, 640), strict=False)
# traced_script_module.save("ts.pt")

# # 2. ts --> pnnx --> ncnn
# os.system("pnnx ts.pt inputshape=[1,3,640,640]  device=cpu")    # 可能错误

# # 3. ncnn ---> optmize ----> ncnn
# os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```

两种转换均已失败告终，报错内容如下：
```
onnx模式下报错：
RuntimeError: step!=1 is currently not supported
pnnx模式下：
:List inputs to traced functions must have consistent element type. Found Tuple[Tensor, List[Tensor]] and Tensor
```

## Example project

- [Android: EdVince/YOLOP-NCNN](https://github.com/EdVince/YOLOP-NCNN)
- [Desktop: EdVince/YOLOP-NCNN](https://github.com/EdVince/YOLOP-NCNN)
  
## Reference

- [详细记录u版YOLOv5目标检测ncnn实现（第二版）](https://zhuanlan.zhihu.com/p/471357671)
- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)


