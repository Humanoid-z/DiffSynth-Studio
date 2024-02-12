import sys
sys.path.append('/DataDisk1/wenlingfeng/DiffSynth-Studio')
from diffsynth import SDVideoPipelineRunner


# Download models
# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575)
# `models/AnimateDiff/mm_sd_v15_v2.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)
# `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
# `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
# `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
# `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)
# `models/textual_inversion/verybadimagenegative_v1.3.pt`: [link](https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16)

# The original video in the examples is https://www.bilibili.com/video/BV1iG411a7sQ/.

config = {
    "models": {
        "model_list": [
            "models/stable_diffusion/aingdiffusion_v12.safetensors",
            "models/AnimateDiff/mm_sd_v15_v2.ckpt",
            "models/ControlNet/control_v11p_sd15_openpose.pth",
            # "models/ControlNet/control_v11f1e_sd15_tile.pth",   #补全细节的controlnet
            "models/ControlNet/control_v11p_sd15_lineart.pth",
            # "models/lora/keqing_lion_optimizer_dim64_loraModel_5e-3noise_token1_4-3-2023.safetensors"
        #     https://civitai.com/models/15699/keqing-or-genshin-impact-or-3in1-lora-and-locon
        ],
        "textual_inversion_folder": "models/textual_inversion",
        "device": "cuda:3",
        "lora_alphas": [1.],
        "controlnet_units": [
            {
                "processor_id": "openpose",
                "model_path": "models/ControlNet/control_v11p_sd15_openpose.pth",
                "scale": 0.5
            },
            # {
            #     "processor_id": "tile",
            #     "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
            #     "scale": 0.3
            # },
            {
                "processor_id": "lineart",
                "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                "scale": 0.5
            }
        ]
    },
    # "smoother_configs": [
    #     {
    #         "processor_type": "FastBlend",
    #         "config": {}
    #     }
    # ],
    "data": {
        "input_frames": {
            "video_file": "data/examples/diffutoon/input_video.mp4",
            "image_folder": None,
            "height": 1024,
            "width": 512,
            "start_frame_id": 0,    # [start_frame_id,end_frame_id)
            "end_frame_id": 30
        },
        "controlnet_frames": [
            {
                "video_file": "data/examples/diffutoon/input_video.mp4",
                "image_folder": None,
                "height": 1024,
                "width": 512,
                "start_frame_id": 0,
                "end_frame_id": 30
            },
            {
                "video_file": "data/examples/diffutoon/input_video.mp4",
                "image_folder": None,
                "height": 1024,
                "width": 512,
                "start_frame_id": 0,
                "end_frame_id": 30
            }
        ],
        "output_folder": "data/examples/diffutoon/output",
        "fps": 30
    },
    "pipeline": {
        "seed": 0,
        "pipeline_inputs": {
            "prompt": "keqing (genshin impact), keqing is dancing, smile, solo, best quality",
            # best quality, perfect anime illustration, light, a girl is dancing, smile, solo
            "negative_prompt": "verybadimagenegative_v1.3",
            "cfg_scale": 7.0,   #7.0
            "clip_skip": 2,
            "denoising_strength": 1.0,
            "num_inference_steps": 10,
            "animatediff_batch_size": 16,
            "animatediff_stride": 8,
            "unet_batch_size": 1,
            "controlnet_batch_size": 1,
            "cross_frame_attention": False,
            # The following parameters will be overwritten. You don't need to modify them.
            "input_frames": [],
            "num_frames": 30,
            "width": 1536,
            "height": 1536,
            "controlnet_frames": []
        }
    }
}
import time
t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
config['data']['output_folder'] = f"data/examples/diffutoon/{t}"
runner = SDVideoPipelineRunner()
runner.run(config)
