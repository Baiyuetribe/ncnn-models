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

# 两种方式都报错
