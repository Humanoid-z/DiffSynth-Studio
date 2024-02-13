import sys
# sys.path.append('/DataDisk1/wenlingfeng/DiffSynth-Studio')
sys.path.append('D:\GitResp\DiffSynth-Studio')
from PIL import Image
from diffsynth import ModelManager, SDImagePipeline, ControlNetConfigUnit
import torch


# Download models
# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575?type=Model&format=SafeTensor&size=full&fp=fp16)
# `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
# `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
# `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
# `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_textual_inversions("models/textual_inversion")
model_manager.load_models([
    # "models/stable_diffusion/aingdiffusion_v12.safetensors",
    # "models/stable_diffusion/diffusion_pytorch_model.fp16.safetensors",
    "models/stable_diffusion/v1-5-pruned-emaonly.safetensors",

    # "models/ControlNet/control_v11f1e_sd15_tile.pth",
    "models/ControlNet/control_v11p_sd15_lineart.pth",
    # "models/ControlNet/control_v11p_sd15_openpose.pth",
    "models/Adapter/ip-adapter-plus_sd15.safetensors"
    # "models/lora/keqing_lion_optimizer_dim64_loraModel_5e-3noise_token1_4-3-2023.safetensors"
])
# for lineart_scale in torch.linspace(0.1,1.0,10):
pipe = SDImagePipeline.from_model_manager(
    model_manager,
    [
        # ControlNetConfigUnit(
        #     processor_id="tile",
        #     model_path=rf"models/ControlNet/control_v11f1e_sd15_tile.pth",
        #     scale=0.6
        # ),
        # ControlNetConfigUnit(
        #     processor_id="openpose",
        #     model_path=rf"models/ControlNet/control_v11p_sd15_openpose.pth",
        #     scale=0.4
        # ),
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path=rf"models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=0.5
        ),
    ]
)
pose_image = Image.open('data/examples/pose.JPG')
ip_adapter_image = Image.open('data/examples/example.jpg')
prompt = "best quality, a beautiful anime girl"
# prompt = "best quality, solo"
# prompt = "anime girl, long hair, bangs, purple hair, sidelocks, blunt bangs, bright pupils, half updo, BREAK shirt, dress, jacket, white shirt, open clothes, hood, white dress, hood down, BREAK looking at viewer, upper body, sky, nature,"
negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# negative_prompt = "worst quality, low quality"

# 或许不应用tile
# lineart_scale 不用或0.1~0.2
torch.manual_seed(0)
ip_scale = 0.45
openpose_scale = 0.4  # in 0.3~0.5
cfg = 7.5 # [6,7,9,11]
# for cfg in range(1,15,1):
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=cfg, clip_skip=1,
    ip_adapter_image=ip_adapter_image,
    controlnet_image=pose_image.resize((1024, 1024)),
    height=1024, width=1024, num_inference_steps=80,
    # denoising_strength=0.7,
)
image.save(f"origin_cfg={cfg}-openpose_scale={openpose_scale}-ip_scale={ip_scale}.jpg")
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     cfg_scale=7.5, clip_skip=1,
#     height=512, width=512, num_inference_steps=80,
# )
# image.save("512.jpg")

# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     cfg_scale=7.5, clip_skip=1,
#     input_image=image.resize((1024, 1024)), controlnet_image=image.resize((1024, 1024)),
#     height=1024, width=1024, num_inference_steps=40, denoising_strength=0.7,
# )
# image.save("1024.jpg")

# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     cfg_scale=7.5, clip_skip=1,
#     input_image=image.resize((2048, 2048)), controlnet_image=image.resize((2048, 2048)),
#     height=2048, width=2048, num_inference_steps=20, denoising_strength=0.7,
# )
# image.save("2048.jpg")
#
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     cfg_scale=7.5, clip_skip=1,
#     input_image=image.resize((4096, 4096)), controlnet_image=image.resize((4096, 4096)),
#     height=4096, width=4096, num_inference_steps=10, denoising_strength=0.5,
#     tiled=True, tile_size=128, tile_stride=64
# )
# image.save("4096.jpg")
