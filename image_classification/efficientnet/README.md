# efficientnet

## Input --> Output

![](https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png)

## Convert 

pt --> torchscript--> pnnx --> ncnn

```python
import torch
import timm # 优秀的预训练模型库
import os

model = timm.create_model('efficientnet_b0', pretrained=True)
model.eval()

x = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")
```

## Example project


## Reference

- [qubvel/efficientnet](https://github.com/qubvel/efficientnet)
- [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


