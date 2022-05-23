import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()
x = torch.rand(1, 3, 224, 224)
# 1. pt ---> onnx
torch_out = torch.onnx._export(model, x, "deeplabv3_resnet50.onnx", export_params=True)

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim deeplabv3_resnet50.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16


# 方法2 ：torchscript 方法报错：提示pnnx layer不存在
# # 1. pt --> torchscript
# traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 224, 224), strict=False)
# traced_script_module.save("ts.pt")

# # 2. ts --> pnnx --> ncnn
# os.system("pnnx ts.pt inputshape=[1,3,224,224]  device=cpu")    # 可能错误

# # 3. ncnn ---> optmize ----> ncnn
# os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
