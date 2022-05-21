import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
# model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
model.eval()

# 1. pt-->torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512))
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,512,512]")
# os.system("pnnx ts.pt")

# 3. ncnn ---> optmize---->ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
