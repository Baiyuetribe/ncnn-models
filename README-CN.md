##  <p align="center"> ncnn models </p>

The collection of pre-trained AI models, and how they were converted, deployed.

![](docs/images/logo.png)

### 关于

ncnn框架借助vulkan实现了全平台部署，我们通过pytorch、tensorflow、飞桨等预训练模型，然后转换成ncnn通用模型，实现Windows、mac、linux、安卓、ios、WebAssembly 以及uni-app的最终部署。然而模型转换并不是一键的，需要手动处理。为了拓展ncnn的边界应用，我们建立此仓库，欢迎提交接收任何转换成功或失败的案例。

### 如何参与贡献？

fork代码，然后按照如下格式提交，最好是c++20 的最小demo，不要内嵌并发。

yolov5  # 项目名称
- models/xx.bin or xx.param # model
- input.png # 输入
- out.png # 输出
- README.md # 模型推理介绍、转换步骤、相关案例
- convert.py # 从pytorch等模型转换具体复现代码

灵感来自[ailia](https://github.com/axinc-ai/ailia-models),鉴于接收失败案例为主，因此失败案例优先排序，成功案例分类排序。

### 一些代表

- [nihui](https://github.com/nihui) ncnn作者
- [飞哥](https://github.com/feigechuanshu) ncnn安卓系列
- [baiyue](https://github.com/Baiyuetribe/paper2gui) ncnn在PC桌面GUI系列
- [670***@qq.com](https://ext.dcloud.net.cn/plugin?id=5243) ncnn在uni-app中的应用
- [nihui](https://github.com/nihui/ncnn_on_esp32) ncnn在嵌入式设备上的应用
- [nihui](https://github.com/nihui/ncnn-webassembly-yolov5) ncnn在wasm实现案例

### 跨设备的意义

> 维护一次代码，多设备可用。核心就是积极反内耗，让跨端部署更加简单，让开发者更加专注模型优化或数据集整理。

- 借助vulkan，无需繁重的cuda驱动
- 借助wasm实现任意设备的模型部署
- 借助uni-app可绕过java实现多端部署