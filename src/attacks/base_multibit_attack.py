from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from ssl_watermarking import encode
from ssl_watermarking import decode
from ssl_watermarking import utils_img as wm_utils_img
from ssl_watermarking import utils as wm_utils

from .base_attack import BaseAttack
from .encode import watermark_multibit_attack
from src.utils import SingleImageDataset


DEFAULT_MULTIBIT_WM_PARAMS = DictConfig(
    {
        "optimizer": "Adam,lr=0.01",
        "scheduler": None,
        "epochs": 100,
        "target_psnr": 42.0,
        "lambda_w": 5e4,
        "lambda_i": 1.0,
        "verbose": 1,
    }
)


class BaseMultibitAttack(BaseAttack, ABC):

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        use_cosine_sim: bool = False,
        params_wm: Optional[DictConfig] = None,
    ):
        super().__init__(name_model, path_backbone, path_norm_layer, device, transform)
        self.batch_size = 1

        if params_wm is None:
            self.params_wm = DEFAULT_MULTIBIT_WM_PARAMS
        else:
            self.params_wm = params_wm

        self.params_wm.batch_size = self.batch_size
        self.use_cosine_sim = use_cosine_sim

        self.target_fpr = 1e-6
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=self.target_fpr)

    def watermark_multibit(
        self,
        img: Image.Image,
        msg: torch.Tensor,
        carrier: torch.Tensor,
        params_wm: DictConfig,
    ) -> Image.Image:
        ds = SingleImageDataset(img, wm_utils_img.default_transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_wm_pt = encode.watermark_multibit(
            dl,
            msg,
            carrier,
            self.model,
            self.transform,
            params_wm,
        )[0]
        img_wm = wm_utils_img.unnormalize_img(img_wm_pt).squeeze(0)
        return TF.to_pil_image(img_wm)

    def watermark_with_estimated(
        self,
        img: Image.Image,
        carrier_est: torch.Tensor,
        carrier_gt: torch.Tensor,
        msg: torch.Tensor,
        params_wm: DictConfig,
    ):
        ds = SingleImageDataset(img, wm_utils_img.default_transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_wm_pt, logs = watermark_multibit_attack(
            dl,
            carrier_est,
            carrier_gt,
            msg,
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

    @torch.inference_mode()
    def evaluate_watermark_multibit(
        self,
        img_orig: Image.Image,
        img_wm: Image.Image,
        msg: torch.Tensor,
        carrier: torch.Tensor,
    ) -> Tuple[float, Dict, Dict]:
        img_orig_np = np.array(img_orig)
        img_wm_np = np.array(img_wm)

        psnr_value = psnr(img_orig_np, img_wm_np)

        def eval_image(img: Image.Image):
            decoded = decode.decode_multibit([img], carrier, self.model)[0]
            msg_decoded = decoded["msg"]
            bit_acc = (
                ~torch.logical_xor(msg.squeeze().cpu(), msg_decoded)
            ).sum().item() / msg.size(0)
            return bit_acc, msg_decoded

        bit_acc_orig, msg_decoded_orig = eval_image(img_orig)
        bit_acc_wm, msg_decoded_wm = eval_image(img_wm)

        return (
            psnr_value,
            {"msg_decoded": msg_decoded_orig, "bit_acc": bit_acc_orig},
            {"msg_decoded": msg_decoded_wm, "bit_acc": bit_acc_wm},
        )

    @abstractmethod
    def attack(self, img, carrier, msg, *args, **kwargs):
        pass
