# efficientnet

## Input --> Output

## Convert 

pt --> torchscript--> pnnx --> ncnn

```python
import torch
import timm
import os

model = timm.create_model('cait_xxs36_384', pretrained=True)
model.eval()

x = torch.rand(1, 3, 384, 384)
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,384,384]")
```

## Example project


## Reference

- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


