# MAT: Mask-Aware Transformer for Large Hole Image Inpainting

## Input --> Output

![](https://github.com/fenglinglwb/MAT/raw/main/figures/teasing.png)
![](https://github.com/fenglinglwb/MAT/raw/main/figures/sota.png)

## Convert 

pt --> TorchScript --> pnnx --> ncnn

```bash
# 欢迎pr
个人运行依赖环境报错：
ImportError: DLL load failed while importing upfirdn2d_plugin: 找不到指定的模块。
warnings.warn('Failed to build CUDApython generate_image.py --network pretrained/CelebA-HQ_512.pkl --dpath test_sets/CelebA-HQ/images --mpath test_sets/CelebA-HQ/masks --outdir
```

## fix



## Reference

- [fenglinglwb/MAT](https://github.com/fenglinglwb/MAT)


