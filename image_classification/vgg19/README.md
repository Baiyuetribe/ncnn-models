# vgg

## Input --> Output

![](https://pytorch.org/assets/images/vgg.png)

## Convert 

pt -->  torchscript --> pnnx --> ncnn

```python
import os
import torch
# 0. pt模型下载及初始化
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)  # vgg11_bn vgg13 vgg13_bn vgg16_bn vgg19_bn vgg19
model.eval()
x = torch.rand(1, 3, 224, 224)  # 最低224*224起步
# 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, torch.randn(1, 3, 224, 224), strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")
```

## Example project


## Reference

- [hub/pytorch_vision_vgg](https://pytorch.org/hub/pytorch_vision_vgg/)


