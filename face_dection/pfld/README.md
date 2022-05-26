# pfld

## Input --> Output

![](https://github.com/guoqiangqi/PFLD/raw/master/data/sample_imgs/ucgif_20190809185908.gif)

## Convert

pt --> onnx --> onnx-sim --> ncnnOptimize --> ncnn

```python
# import os
# import torch
# # 0. pt模型下载及初始化
# # 1. download onnx model # Windows下手动执行
# os.system("wget xxxx.onnx -O xxx.onnx")

# # 2. onnx --> onnxsim
# os.system("python3 -m onnxsim xxx.onnx sim.onnx")

# # 3. onnx --> ncnn
# os.system("onnx2ncnn sim.onnx ncnn.param ncnn.bin")

# # 4. ncnn --> optmize ---> ncnn
# os.system("ncnnoptimize ncnn.param ncnn.bin opt.param opt.bin 1")  # 数字0 代表fp32 ；1代表fp16
```

## Example project

- [abyssss52/PFLD_ncnn_test](https://github.com/abyssss52/PFLD_ncnn_test)


## Reference

- [polarisZhao/PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)


