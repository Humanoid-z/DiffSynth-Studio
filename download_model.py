import requests
import os

def download_model(url, file_path):
  proxies = {
    'http': 'http://localhost:7890',
    'https': 'https://localhost:7890',
  }
  model_file = requests.get(url, allow_redirects=True,verify=False)
  with open(file_path, "wb") as f:
    f.write(model_file.content)
  print(file_path+' downloaded')

# download_model("https://civitai.com/api/download/models/229575", "models/stable_diffusion/aingdiffusion_v12.safetensors")
# download_model("https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt", "models/AnimateDiff/mm_sd_v15_v2.ckpt")
# download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth", "models/ControlNet/control_v11p_sd15_lineart.pth")
# download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth", "models/ControlNet/control_v11f1e_sd15_tile.pth")
# # download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth", "models/ControlNet/control_v11f1p_sd15_depth.pth")
# # download_model("https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth", "models/ControlNet/control_v11p_sd15_softedge.pth")
# # download_model("https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt", "models/Annotators/dpt_hybrid-midas-501f0c75.pt")
# # download_model("https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth", "models/Annotators/ControlNetHED.pth")
# download_model("https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth", "models/Annotators/sk_model.pth")
# download_model("https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth", "models/Annotators/sk_model2.pth")
# download_model("https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16", "models/textual_inversion/verybadimagenegative_v1.3.pt")


# download_model("https://civitai.com/api/download/models/66246", "models/lora/bocchi_style_offset.safetensors")
download_model("http://hf-mirror.com/h94/IP-Adapter/blob/main/models/image_encoder/model.safetensors", "models/CLIPImageEncoder/model.safetensors")
download_model("http://hf-mirror.com/h94/IP-Adapter/blob/main/models/image_encoder/config.json", "models/CLIPImageEncoder/config.json")