import torch
import timm  # 优秀的预训练模型库
import os

model = timm.create_model('efficientnet_b0', pretrained=True)   # efficientnet_b0~...
model.eval()

x = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, x, strict=False)
traced_script_module.save("ts.pt")

# 2. ts --> pnnx --> ncnn
os.system("pnnx ts.pt inputshape=[1,3,224,224]")

# 支持下列所有模型
# "efficientnet_b0",
# "efficientnet_b1",
# "efficientnet_b1_pruned",
# "efficientnet_b2",
# "efficientnet_b2_pruned",
# "efficientnet_b3",
# "efficientnet_b3_pruned",
# "efficientnet_b4",
# "efficientnet_el",
# "efficientnet_el_pruned",
# "efficientnet_em",
# "efficientnet_es",
# "efficientnet_es_pruned",
# "efficientnet_lite0",
