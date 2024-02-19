from typing_extensions import Literal, TypeAlias
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from controlnet_aux.processor import (
        CannyDetector, MidasDetector, HEDdetector, LineartDetector, LineartAnimeDetector, OpenposeDetector,DWposeDetector
    )

Processor_id: TypeAlias = Literal[
    "canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "tile"
]

class Annotator:
    def __init__(self, processor_id: Processor_id, model_path="models/Annotators", detect_resolution=None):
        if processor_id == "canny":
            self.processor = CannyDetector()
        elif processor_id == "depth":
            self.processor = MidasDetector.from_pretrained(model_path).to("cuda")
        elif processor_id == "softedge":
            self.processor = HEDdetector.from_pretrained(model_path).to("cuda")
        elif processor_id == "lineart":
            self.processor = LineartDetector.from_pretrained(model_path).to("cuda")
        elif processor_id == "lineart_anime":
            self.processor = LineartAnimeDetector.from_pretrained(model_path).to("cuda")
        elif processor_id == "openpose":
            self.processor = OpenposeDetector.from_pretrained(model_path).to("cuda")
        elif processor_id == "dwpose":
            # det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
            # det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
            # pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
            # pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
            det_config = model_path+'/dwpose/yolox_l_8xb8-300e_coco.py'
            det_ckpt = model_path+'/dwpose/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
            pose_config = model_path+'/dwpose/dwpose-l_384x288.py'
            pose_ckpt = model_path+'/dwpose/dw-ll_ucoco_384.pth'
            self.processor = DWposeDetector(det_config,det_ckpt,pose_config,pose_ckpt).to("cuda")
        elif processor_id == "tile":
            self.processor = None
        else:
            raise ValueError(f"Unsupported processor_id: {processor_id}")
        print('load ',processor_id)
        self.processor_id = processor_id
        self.detect_resolution = detect_resolution

    def __call__(self, image):
        width, height = image.size
        if self.processor_id == "openpose":
            kwargs = {
                "include_body": True,
                "include_hand": True,
                "include_face": True
            }
        else:
            kwargs = {}
        if self.processor is not None:
            detect_resolution = self.detect_resolution if self.detect_resolution is not None else min(width, height)
            image = self.processor(image, detect_resolution=detect_resolution, image_resolution=min(width, height), **kwargs)
        image = image.resize((width, height))
        return image

