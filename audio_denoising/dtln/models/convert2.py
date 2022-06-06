import os

# 2. onnx --> onnxsim
os.system("python -m onnxsim model_p2.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16

# 从原始项目中的onnx模型进行转换，原始项目pnnx模式报错，意思pnnx算子不支持aten:square
