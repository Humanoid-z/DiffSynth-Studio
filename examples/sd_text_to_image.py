import sys
sys.path.append('/DataDisk1/wenlingfeng/DiffSynth-Studio')
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
    "models/stable_diffusion/aingdiffusion_v12.safetensors",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
    # "models/ControlNet/control_v11p_sd15_lineart.pth",
    "models/ControlNet/control_v11p_sd15_openpose.pth",
    "models/lora/keqing_lion_optimizer_dim64_loraModel_5e-3noise_token1_4-3-2023.safetensors"
],[1.])
pipe = SDImagePipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="tile",
            model_path=rf"models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="openpose",
            model_path=rf"models/ControlNet/control_v11p_sd15_openpose.pth",
            scale=0.5
        ),
    ]
)
input_image = Image.open('data/examples/example.jpeg')
prompt = "keqing (piercing thunderbolt) (genshin impact), keqing (genshin impact), pantyhose, hair bun, purple hair, gloves, twintails, long hair, purple eyes, diamond-shaped pupils, bare shoulders, hair ornament, black pantyhose, cone hair bun, detached sleeves,dress, jewelry, medium breasts, earrings, bangs, frills, purple dress, black gloves, braid, skirt, masterpiece, best quality"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist,"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=7.5, clip_skip=1,
    # input_image=input_image.resize((1024, 1024)),
    controlnet_image=input_image.resize((1024, 1024)),
    height=1024, width=1024, num_inference_steps=80,
    # denoising_strength=0.7,
)
image.save("1024.jpg")
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
