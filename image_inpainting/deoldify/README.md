# AnimeGanV3

## Input --> Output

![](input.png)
![](output.jpg)

## Convert 

### 方法1:
pt --> TorchScript --> pnnx --> ncnnOptimize --> ncnn

```python
# 欢迎pr
```
### 方法2：
pt ---> onnx ---> onnxsim ---> onnx2ncnn ---> 手动修复reshape等 ---> ncnnOptimize --> ncnn

实现案例：https://zhuanlan.zhihu.com/p/350332071

## fix

已修复模型-23303=1,0 5=1;移除C代码；简化demo运行

## Reference

- [jantic/DeOldify](https://github.com/jantic/DeOldify)
- [KeepGoing2019HaHa/AI-application](https://github.com/KeepGoing2019HaHa/AI-application)


