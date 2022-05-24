import os
import torch
# 0. pt模型下载及初始化
# 1. download onnx model
os.system("wget https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx -O nanodet-plus-m_416.onnx")

# 2. onnx --> onnxsim
os.system("python3 -m onnxsim nanodet-plus-m_416.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
