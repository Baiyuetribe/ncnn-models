import torch
import timm
import os

model = timm.create_model('cait_xxs36_384', pretrained=True)
model.eval()

x = torch.rand(1, 3, 384, 384)
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,384,384]")

# 模型文件已支持如下：
#     "cait_m36_384",
#     "cait_m48_448",
#     "cait_s24_224",
#     "cait_s24_384",
#     "cait_s36_384",
#     "cait_xs24_384",
#     "cait_xxs24_224",
#     "cait_xxs24_384",
#     "cait_xxs36_224",
#     "cait_xxs36_384",
