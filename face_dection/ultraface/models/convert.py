import os
import torch
# 0. pt模型下载及初始化
# 1. download onnx model # Windows下手动执行
os.system("wget https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx -O version-RFB-320.onnx")

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim version-RFB-320.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
