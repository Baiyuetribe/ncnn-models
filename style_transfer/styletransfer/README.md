# fast_neural_style

## Input --> Output

![](https://github.com/pytorch/examples/raw/main/fast_neural_style/images/content-images/amber.jpg)
![](https://github.com/pytorch/examples/raw/main/fast_neural_style/images/style-images/mosaic.jpg)
![](https://github.com/pytorch/examples/blob/main/fast_neural_style/images/style-images/candy.jpg)
![](https://github.com/pytorch/examples/raw/main/fast_neural_style/images/style-images/rain-princess-cropped.jpg)

## Convert 

onnx --> simpleonnx --> onnx2ncnn --> ncnnOptimize --> ncnn

```
import os
# 1. download models
# download_url = https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style/model
os.system("")

# 2. onnx --> onnxsim
os.system("python -m onnxsim candy-9.onnx sim.onnx")

# 3. onnx --> ncnn
os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# 4. ncnn --> optmize ---> ncnn
os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```


## Reference

- [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)
- [style_transfer/fast_neural_style](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style)


