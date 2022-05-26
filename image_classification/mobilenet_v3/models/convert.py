import os
import torch
import timm
# 0. pt模型下载及初始化
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)  # 224起步
# # 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# # 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")
