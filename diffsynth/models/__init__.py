import torch, os
from diffusers.models import ImageProjection
from diffusers.models.embeddings import IPAdapterFullImageProjection, IPAdapterPlusImageProjection
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .sd_text_encoder import SDTextEncoder
from .sd_unet import SDUNet
from .sd_vae_encoder import SDVAEEncoder
from .sd_vae_decoder import SDVAEDecoder
from .sd_lora import SDLoRA

from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sdxl_unet import SDXLUNet
from .sdxl_vae_decoder import SDXLVAEDecoder
from .sdxl_vae_encoder import SDXLVAEEncoder

from .sd_controlnet import SDControlNet

from .sd_motion import SDMotionModel


class ModelManager:
    def __init__(self, torch_dtype=torch.float16, device="cuda"):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = {}
        self.model_path = {}
        self.textual_inversion_dict = {}

    def is_RIFE(self, state_dict):
        param_name = "block_tea.convblock3.0.1.weight"
        return param_name in state_dict or ("module." + param_name) in state_dict

    def is_beautiful_prompt(self, state_dict):
        param_name = "transformer.h.9.self_attention.query_key_value.weight"
        return param_name in state_dict

    def is_stabe_diffusion_xl(self, state_dict):
        param_name = "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight"
        return param_name in state_dict

    def is_stable_diffusion(self, state_dict):
        if self.is_stabe_diffusion_xl(state_dict):
            return False
        param_name = "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight"
        return param_name in state_dict
    
    def is_controlnet(self, state_dict,file_path=None):
        if file_path == 'models/ControlNet/control_v11p_sd15_softedge.fp16.safetensors':
            return True
        param_name = "control_model.time_embed.0.weight"
        return param_name in state_dict
    
    def is_animatediff(self, state_dict):
        param_name = "mid_block.motion_modules.0.temporal_transformer.proj_out.weight"
        return param_name in state_dict
    
    def is_sd_lora(self, state_dict):
        param_name = "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight"
        return param_name in state_dict
    
    def is_translator(self, state_dict):
        param_name = "model.encoder.layers.5.self_attn_layer_norm.weight"
        return param_name in state_dict and len(state_dict) == 254
    def is_ip_adapter(self, state_dict):
        param_name = "ip_adapter.1.to_k_ip.weight"
        return param_name in state_dict
    def load_stable_diffusion(self, state_dict, components=None, file_path=""):
        component_dict = {
            "text_encoder": SDTextEncoder,
            "unet": SDUNet,
            "vae_decoder": SDVAEDecoder,
            "vae_encoder": SDVAEEncoder,
            "refiner": SDXLUNet,
        }
        if components is None:
            components = ["text_encoder", "unet", "vae_decoder", "vae_encoder"]
        for component in components:
            if component == "text_encoder":
                # Add additional token embeddings to text encoder
                token_embeddings = [state_dict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]]
                for keyword in self.textual_inversion_dict:
                    _, embeddings = self.textual_inversion_dict[keyword]
                    token_embeddings.append(embeddings.to(dtype=token_embeddings[0].dtype))
                token_embeddings = torch.concat(token_embeddings, dim=0)
                state_dict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = token_embeddings
                self.model[component] = component_dict[component](vocab_size=token_embeddings.shape[0])
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
                self.model[component].to(self.torch_dtype).to(self.device)
            else:
                self.model[component] = component_dict[component]()
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict),strict=False)
                self.model[component].to(self.torch_dtype).to(self.device)
            self.model_path[component] = file_path

    def load_stable_diffusion_xl(self, state_dict, components=None, file_path=""):
        component_dict = {
            "text_encoder": SDXLTextEncoder,
            "text_encoder_2": SDXLTextEncoder2,
            "unet": SDXLUNet,
            "vae_decoder": SDXLVAEDecoder,
            "vae_encoder": SDXLVAEEncoder,
        }
        if components is None:
            components = ["text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder"]
        for component in components:
            self.model[component] = component_dict[component]()
            self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
            if component in ["vae_decoder", "vae_encoder"]:
                # These two model will output nan when float16 is enabled.
                # The precision problem happens in the last three resnet blocks.
                # I do not know how to solve this problem.
                self.model[component].to(torch.float32).to(self.device)
            else:
                self.model[component].to(self.torch_dtype).to(self.device)
            self.model_path[component] = file_path

    def load_controlnet(self, state_dict, file_path=""):
        component = "controlnet"
        if component not in self.model:
            self.model[component] = []
            self.model_path[component] = []
        model = SDControlNet()
        if file_path == 'models/ControlNet/control_v11p_sd15_softedge.fp16.safetensors':
            model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        else:
            model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component].append(model)
        self.model_path[component].append(file_path)

    def load_animatediff(self, state_dict, file_path=""):
        component = "motion_modules"
        model = SDMotionModel()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_beautiful_prompt(self, state_dict, file_path=""):
        component = "beautiful_prompt"
        from transformers import AutoModelForCausalLM
        model_folder = os.path.dirname(file_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_folder, state_dict=state_dict, local_files_only=True, torch_dtype=self.torch_dtype
        ).to(self.device).eval()
        self.model[component] = model
        self.model_path[component] = file_path

    def load_RIFE(self, state_dict, file_path=""):
        component = "RIFE"
        from ..extensions.RIFE import IFNet
        model = IFNet().eval()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(torch.float32).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_sd_lora(self, state_dict, alpha):
        SDLoRA().add_lora_to_text_encoder(self.model["text_encoder"], state_dict, alpha=alpha, device=self.device)
        SDLoRA().add_lora_to_unet(self.model["unet"], state_dict, alpha=alpha, device=self.device)

    def load_translator(self, state_dict, file_path=""):
        # This model is lightweight, we do not place it on GPU.
        component = "translator"
        from transformers import AutoModelForSeq2SeqLM
        model_folder = os.path.dirname(file_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_folder).eval()
        self.model[component] = model
        self.model_path[component] = file_path

    def load_CLIPImageProcessor(self):
        component = "clip_image_processor"
        clip_image_processor = CLIPImageProcessor()
        self.model[component] = clip_image_processor
    def load_CLIPImageEncoder(self):
        component = "image_encoder"
        image_encoder = CLIPVisionModelWithProjection.from_pretrained('./models/CLIPImageEncoder/').to(
            self.device, dtype=self.torch_dtype
        )
        self.model[component] = image_encoder
        print('CLIPVisionModelWithProjection loaded')
    # diffusers/src/diffusers/loaders/unet.py _convert_ip_adapter_image_proj_to_diffusers(self, state_dict):
    def load_image_projector(self, state_dict):
        component = "image_projector"
        updated_state_dict = {}
        image_projection = None

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            image_projection = ImageProjection(
                cross_attention_dim=cross_attention_dim,
                image_embed_dim=clip_embeddings_dim,
                num_image_text_embeds=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            image_projection = IPAdapterFullImageProjection(
                cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

            image_projection = IPAdapterPlusImageProjection(
                embed_dims=embed_dims,
                output_dims=output_dims,
                hidden_dims=hidden_dims,
                heads=heads,
                num_queries=num_image_text_embeds,
            )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")
                diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
                diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
                diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    updated_state_dict[diffusers_name] = value

        image_projection.load_state_dict(updated_state_dict)
        self.model[component] = image_projection.to(self.torch_dtype).to(self.device).eval()
        print('image_projector loaded')
    def load_IP_Adapter(self,state_dict):
        IP_Adapter2Diffuser_map = {'1.to_k_ip.weight': 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '1.to_v_ip.weight': 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '3.to_k_ip.weight': 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '3.to_v_ip.weight': 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '5.to_k_ip.weight': 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '5.to_v_ip.weight': 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '7.to_k_ip.weight': 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '7.to_v_ip.weight': 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '9.to_k_ip.weight': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '9.to_v_ip.weight': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '11.to_k_ip.weight': 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '11.to_v_ip.weight': 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '13.to_k_ip.weight': 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '13.to_v_ip.weight': 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '15.to_k_ip.weight': 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '15.to_v_ip.weight': 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '17.to_k_ip.weight': 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k_ip.weight',
         '17.to_v_ip.weight': 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v_ip.weight',
         '19.to_k_ip.weight': 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '19.to_v_ip.weight': 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '21.to_k_ip.weight': 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '21.to_v_ip.weight': 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '23.to_k_ip.weight': 'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k_ip.weight',
         '23.to_v_ip.weight': 'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v_ip.weight',
         '25.to_k_ip.weight': 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '25.to_v_ip.weight': 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight',
         '27.to_k_ip.weight': 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k_ip.weight',
         '27.to_v_ip.weight': 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v_ip.weight',
         '29.to_k_ip.weight': 'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k_ip.weight',
         '29.to_v_ip.weight': 'up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v_ip.weight',
         '31.to_k_ip.weight': 'mid_block.attentions.0.transformer_blocks.0.attn2.to_k_ip.weight',
         '31.to_v_ip.weight': 'mid_block.attentions.0.transformer_blocks.0.attn2.to_v_ip.weight'}
        new_state_dict = {IP_Adapter2Diffuser_map[k]:state_dict[k] for k in state_dict.keys()}
        self.model["unet"].load_state_dict(self.model["unet"].state_dict_converter().from_diffusers(new_state_dict),strict=False)
        self.model["unet"].to(self.torch_dtype).to(self.device)
        self.model["ip_adapter"] = None
        print('ip_adapter loaded')


    def search_for_embeddings(self, state_dict):
        embeddings = []
        for k in state_dict:
            if isinstance(state_dict[k], torch.Tensor):
                embeddings.append(state_dict[k])
            elif isinstance(state_dict[k], dict):
                embeddings += self.search_for_embeddings(state_dict[k])
        return embeddings

    def load_textual_inversions(self, folder):
        # Store additional tokens here
        self.textual_inversion_dict = {}

        # Load every textual inversion file
        for file_name in os.listdir(folder):
            if file_name.endswith(".txt"):
                continue
            keyword = os.path.splitext(file_name)[0]
            state_dict = load_state_dict(os.path.join(folder, file_name))

            # Search for embeddings
            for embeddings in self.search_for_embeddings(state_dict):
                if len(embeddings.shape) == 2 and embeddings.shape[1] == 768:
                    tokens = [f"{keyword}_{i}" for i in range(embeddings.shape[0])]
                    self.textual_inversion_dict[keyword] = (tokens, embeddings)
                    break
        
    def load_model(self, file_path, components=None, lora_alphas=[]):
        state_dict = load_state_dict(file_path, torch_dtype=self.torch_dtype)
        if self.is_animatediff(state_dict):
            self.load_animatediff(state_dict, file_path=file_path)
        elif self.is_controlnet(state_dict,file_path):
            self.load_controlnet(state_dict, file_path=file_path)
        elif self.is_stabe_diffusion_xl(state_dict):
            self.load_stable_diffusion_xl(state_dict, components=components, file_path=file_path)
        elif self.is_stable_diffusion(state_dict):
            self.load_stable_diffusion(state_dict, components=components, file_path=file_path)
        elif self.is_sd_lora(state_dict):
            self.load_sd_lora(state_dict, alpha=lora_alphas.pop(0))
            print('load_sd_lora: ',file_path)
        elif self.is_beautiful_prompt(state_dict):
            self.load_beautiful_prompt(state_dict, file_path=file_path)
        elif self.is_RIFE(state_dict):
            self.load_RIFE(state_dict, file_path=file_path)
        elif self.is_translator(state_dict):
            self.load_translator(state_dict, file_path=file_path)
        elif self.is_ip_adapter(state_dict):
            new_state_dict = {"image_proj": {}, "ip_adapter": {}}
            for key in state_dict.keys():
                if key.startswith("image_proj."):
                    new_state_dict["image_proj"][key.replace("image_proj.", "")] = state_dict[key]
                elif key.startswith("ip_adapter."):
                    new_state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = state_dict[key]
            self.load_IP_Adapter(new_state_dict['ip_adapter'])
            self.load_image_projector(new_state_dict['image_proj'])

    def load_models(self, file_path_list, lora_alphas=[]):
        for file_path in file_path_list:
            self.load_model(file_path, lora_alphas=lora_alphas)
        if 'ip_adapter' in self.model:
            self.load_CLIPImageEncoder()
            self.load_CLIPImageProcessor()
            #self.load_image_projector() in load ip_adapter


    def to(self, device):
        for component in self.model:
            if self.model[component] is None or isinstance(self.model[component],CLIPImageProcessor):
                continue
            if isinstance(self.model[component], list):
                for model in self.model[component]:
                    model.to(device)
            else:
                self.model[component].to(device)
        torch.cuda.empty_cache()

    def get_model_with_model_path(self, model_path):    # only for fetch_controlnet_models
        for component in self.model_path:
            if isinstance(self.model_path[component], str):
                if os.path.samefile(self.model_path[component], model_path):
                    return self.model[component]
            elif isinstance(self.model_path[component], list):
                for i, model_path_ in enumerate(self.model_path[component]):
                    if os.path.samefile(model_path_, model_path):
                        return self.model[component][i]
        raise ValueError(f"Please load model {model_path} before you use it.")
    
    def __getattr__(self, __name):
        if __name in self.model:
            return self.model[__name]
        else:
            return super.__getattribute__(__name)


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu")
    if torch_dtype is not None:
        state_dict = {i: state_dict[i].to(torch_dtype) for i in state_dict}
    return state_dict


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-6:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-6:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)
