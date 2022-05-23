import os
import torch
import torchvision
import torch.onnx
# 0. pt模型下载及初始化
model = torchvision.models.resnet18()
# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 224, 224)
# 1. pt ---> onnx
torch_out = torch.onnx._export(model, x, "resnet18.onnx", export_params=True)

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim resnet18.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
