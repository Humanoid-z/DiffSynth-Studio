import torch
from safetensors import safe_open

from diffsynth.models.sd_unet import SDUNetStateDictConverter

ip_ckpt = 'models/Adapter/ip-adapter-plus_sd15.safetensors'
state_dict = {"image_proj": {}, "ip_adapter": {}}
with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith("image_proj."):
            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
        elif key.startswith("ip_adapter."):
            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
# print(state_dict["image_proj"].keys())
# print(state_dict["ip_adapter"].keys())
# print(len(state_dict["ip_adapter"].keys()))     # 16 layers

# ckpt = 'models/stable_diffusion/aingdiffusion_v12.safetensors'
# state_dict = {}
# with safe_open(ckpt, framework="pt", device="cpu") as f:
#     for key in f.keys():
#         state_dict[key] = f.get_tensor(key)
# print(state_dict.keys())

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

unet = UNet2DConditionModel()
# ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
# for n,_ in unet.named_parameters():
#     print(n)
key_id = 1
new_state_dict = {}
for name in unet.attn_processors.keys():
    if name.endswith("attn2.processor"):
        new_name = name[:-10]
        # print(f"{key_id}.to_k_ip.weight")
        new_state_dict[new_name+'.to_k_ip.weight'] = state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]
        new_state_dict[new_name + '.to_v_ip.weight'] = state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]
        key_id += 2
# print(new_state_dict.keys())

print(SDUNetStateDictConverter().from_diffusers(new_state_dict).keys())

