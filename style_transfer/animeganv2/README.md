# AnimeGanV3

## Input --> Output

![](https://user-images.githubusercontent.com/26464535/142294796-54394a4a-a566-47a1-b9ab-4e715b901442.gif)

## Convert 

pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model.eval()

# 1. pt-->torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512))
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,512,512]")

# 3. ncnn ---> optmize---->ncnn
os.system("ncnnoptimize ts.ncnn.param ts.ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```


## Reference

- [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)


