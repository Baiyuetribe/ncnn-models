import torch
import model.detector   # 导入模型
import os

# model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True)
model = model.detector.Detector(80, 3, True, True)
model.load_state_dict(torch.load(r"modelzoo\coco2017-0.241078ap-model.pth"))
# sets the module in eval node
model.eval()


# # 1. pt --> torchscript
# traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 352, 352))
# traced_script_module.save("ts.pt")
# # 2. ts --> pnnx --> ncnn
# os.system("pnnx ts.pt inputshape=[1,3,352,352]")

torch.onnx.export(model,  # model being run
                  torch.randn(1, 3, 352, 352),                 # model input (or a tuple for multiple inputs)
                  "out.onnx",               # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True)  # whether to execute constant folding for optimization

# 2. onnx --> onnxsim
os.system("python -m onnxsim out.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16

# 转换方法，将上述文件放到项目目录下，然后运行命令就可以成功。
# torchscript转换成功，但是运行失败，onnx方法转换成功
