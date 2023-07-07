# StableDiffuson

## Input --> Output

![](https://user-images.githubusercontent.com/43716063/213969098-433f39eb-7171-43c8-aae4-dd2de91a0b4c.png)

## 出入信息

### 输入尺寸

[1,4,-1,-1],[1],[1,77,768]

## 模型转换

```python
import os
import torch
from diffusers import StableDiffusionPipeline

# config
device = "cuda"
from_model = "../diffusers-model"
to_model = "model"
height, width = 512, 512

# check
assert height % 8 == 0 and width % 8 == 0
height, width = height // 8, width // 8
os.makedirs(to_model, exist_ok=True)

# load model
pipe = StableDiffusionPipeline.from_pretrained(from_model, torch_dtype=torch.float32)
pipe = pipe.to(device)

# jit unet
unet = torch.jit.trace(pipe.unet, (torch.rand(1,4,height,width).to(device),torch.rand(1).to(device),torch.rand(1,77,768).to(device)))
unet.save(os.path.join(to_model,"unet.pt"))

## 4. fp16量化
pnnx unet.pt inputshape=[1,4,64,64],[1],[1,77,768]

```

## c++实现

参见 <https://github.com/EdVince/Stable-Diffusion-NCNN>

## Example project

## Reference

- [EdVince/Stable-Diffusion-NCNN](https://github.com/EdVince/Stable-Diffusion-NCNN)
- [Tencent/ncnn](https://github.com/Tencent/ncnn)
