import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")  # or "resnet50"
model.cpu()
model.eval()
# model(src, *rec, downsample_ratio=0.25) # src can be [B, C, H, W] or [B, T, C, H, W] RGB input is normalized to 0~1 range.
# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,512,512] device=cpu")    # 可能错误


# 运行后c++报错：原因暂时不清楚
# ignore pnnx.Expression pnnx_expr_256 param expr=1
# create_custom_layer pnnx.Expression
# fuse_convolution_activation conv_12 hswish_83
