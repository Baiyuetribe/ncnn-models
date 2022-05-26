# res2net

## Input --> Output

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_4.32.52_PM.png)

## Convert 

pt -->  torchscript --> pnnx --> ncnn

```python
import os
import torch
import timm
# 0. pt模型下载及初始化
model = timm.create_model('res2next50', pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)  # 224起步
# # 1. pt --> torchscript
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# # 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")
```

## Example project


## Reference

- [pytorch-image-models/models/res2next/](https://rwightman.github.io/pytorch-image-models/models/res2next/)


