from abc import ABC, abstractmethod
from PIL import Image

import torch
import torch.nn as nn

from ssl_watermarking import data_augmentation
from ssl_watermarking import utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img

from src.utils import DEFAULT_IMAGE_SHAPE


class BaseAttack(ABC):
    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
    ):
        self.name_model = name_model
        self.path_backbone = path_backbone
        self.path_norm_layer = path_norm_layer
        self.device = device
        self.model = self.create_wm_model()

        if transform == "none":
            self.transform = data_augmentation.DifferentiableDataAugmentation()
        elif transform == "all":
            self.transform = data_augmentation.All()

        self.D = self.model(
            torch.zeros((1, *DEFAULT_IMAGE_SHAPE)).to(self.device)
        ).size(-1)

    def create_wm_model(self) -> nn.Module:
        backbone = wm_utils.build_backbone(
            path=self.path_backbone, name=self.name_model
        )
        normlayer = wm_utils.load_normalization_layer(path=self.path_norm_layer)
        model_wm = wm_utils.NormLayerWrapper(backbone, normlayer)
        model_wm = model_wm.to(self.device, non_blocking=True)

        for param in model_wm.parameters():
            param.requires_grad = False

        model_wm.eval()
        return model_wm

    @torch.no_grad()
    def extract_emb(self, img: Image.Image) -> torch.Tensor:
        img_pt = wm_utils_img.default_transform(img).unsqueeze(0).to(self.device)
        emb = self.model(img_pt).detach().clone()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @abstractmethod
    def attack(self, img, carrier, *args, **kwargs):
        pass
