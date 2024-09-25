from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

import ssl_watermarking.encode as encode
import ssl_watermarking.utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img

from .base_attack import BaseAttack
from .encode import watermark_0bit_attack
from src.utils import SingleImageDataset


DEFAULT_0BIT_WM_PARAMS = DictConfig(
    {
        "optimizer": "Adam,lr=0.01",
        "scheduler": None,
        "epochs": 100,
        "target_psnr": 42.0,
        "lambda_w": 1.0,
        "lambda_i": 1.0,
        "verbose": 1,
    }
)


class Base0BitAttack(BaseAttack, ABC):
    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        target_fpr: float = 1e-6,
        params_wm: Optional[DictConfig] = None,
        use_cosine_sim: bool = False,
    ):
        super().__init__(name_model, path_backbone, path_norm_layer, device, transform)
        self.target_fpr = target_fpr
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=target_fpr)
        self.rho = 1 + np.tan(self.angle) ** 2
        self.batch_size = 1

        if params_wm is None:
            self.params_wm = DEFAULT_0BIT_WM_PARAMS
        else:
            self.params_wm = params_wm

        self.use_cosine_sim = use_cosine_sim

    def watermark_0bit(
        self, img: Image.Image, carrier: torch.Tensor, params_wm: DictConfig
    ) -> Image.Image:
        ds = SingleImageDataset(img, wm_utils_img.default_transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_wm_pt = encode.watermark_0bit(
            dl,
            carrier,
            self.angle,
            self.model,
            self.transform,
            params_wm,
        )[0]
        img_wm = wm_utils_img.unnormalize_img(img_wm_pt).squeeze(0)
        return TF.to_pil_image(img_wm)

    @torch.inference_mode()
    def evaluate_watermark_0bit(
        self, img_orig: Image.Image, img_wm: Image.Image, carrier: torch.Tensor
    ) -> Tuple[float, Dict, Dict]:
        img_orig_np = np.array(img_orig)
        img_wm_np = np.array(img_wm)

        psnr_value = psnr(img_orig_np, img_wm_np)

        def eval_image(img: Image.Image) -> Tuple[float, float, float]:
            img_pt = wm_utils_img.default_transform(img).unsqueeze(0).to(self.device)
            emb = self.model(img_pt)
            dot = emb @ carrier.T
            norm = torch.norm(emb, dim=-1)
            R = (self.rho * dot**2 - norm**2).item()
            cosine = torch.abs(dot / norm).item()
            log10_pvalue = np.log10(wm_utils.cosine_pvalue(cosine, self.D))
            return R, cosine, log10_pvalue

        R_orig, cosine_orig, log10_pvalue_orig = eval_image(img_orig)
        R_wm, cosine_wm, log10_pvalue_wm = eval_image(img_wm)

        return (
            psnr_value,
            {
                "R": R_orig,
                "cosine": cosine_orig,
                "log10_pvalue": log10_pvalue_orig,
                "decision": R_orig > 0,
            },
            {
                "R": R_wm,
                "cosine": cosine_wm,
                "log10_pvalue": log10_pvalue_wm,
                "decision": R_wm > 0,
            },
        )

    def watermark_with_estimated(
        self,
        img: Image.Image,
        carrier_est: torch.Tensor,
        carrier_gt: torch.Tensor,
        params_wm: DictConfig,
    ) -> Tuple[Image.Image, Dict]:
        ds = SingleImageDataset(img, wm_utils_img.default_transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_wm_pt, logs = watermark_0bit_attack(
            dl,
            carrier_est,
            carrier_gt,
            self.angle,
            self.model,
            self.transform,
            params_wm,
            return_logs=True,
            use_cosine_sim=self.use_cosine_sim,
        )
        img_wm_pt = img_wm_pt[0]
        img_wm = wm_utils_img.unnormalize_img(img_wm_pt).squeeze(0)
        return TF.to_pil_image(img_wm), logs

    @abstractmethod
    def attack(self, img, carrier, *args, **kwargs):
        pass
