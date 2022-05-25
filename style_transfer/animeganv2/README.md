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
# model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
model.eval()

# 1. pt-->torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 512, 512))
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,512,512]")    # 2022年5月25日起，pnnx默认自动量化，不需要再次optmize
```

## Example project

- [Android: nihui/ncnn-android-styletransfer](https://github.com/nihui/ncnn-android-styletransfer)
- [Desktop: baiyue/animegan_gui](https://github.com/Baiyuetribe/paper2gui/blob/main/Style%20Transfer/animegan_gui.md)  

## Reference

- [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)


