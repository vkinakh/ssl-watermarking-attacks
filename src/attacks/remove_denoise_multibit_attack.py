from typing import Optional, Dict

import numpy as np
from omegaconf import DictConfig
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from ssl_watermarking import encode
from ssl_watermarking import utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img

from .base_multibit_attack import BaseMultibitAttack
from src.utils import SingleImageDataset, wiener2, UINT8_MAX


DEFAULT_PARAMS_WATERMARK = DictConfig(
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


class RemoveDenoiseMultibitAttack(BaseMultibitAttack):
    """Remove multibit watermarking by denoising the watermarked image and using the estimated embedding to remodulate
    to remove the watermark"""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        use_cosine_sim: bool = False,
        filter_size: int = 25,
        params_wm: Optional[DictConfig] = None,
        params_attack: Optional[DictConfig] = None,
    ):
        super().__init__(
            name_model,
            path_backbone,
            path_norm_layer,
            device,
            transform,
            use_cosine_sim,
            params_wm,
        )

        if params_attack is None:
            self.params_attack = DEFAULT_PARAMS_WATERMARK
        else:
            self.params_attack = params_attack

        self.params_attack.batch_size = self.batch_size

        self.target_fpr = 1e-6
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=self.target_fpr)

        self.filter_size = filter_size

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

    def denoise_image(self, img: Image.Image) -> Image.Image:
        denoise_params = {"local_mean_filter_size": self.filter_size, "noise_var": None}

        img_np = np.array(img)
        img_np = img_np / UINT8_MAX
        img_np = wiener2(img_np, **denoise_params)
        img_np = np.clip(img_np * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)
        return Image.fromarray(img_np)

    def attack(
        self, img: Image.Image, carrier: torch.Tensor, msg: torch.Tensor
    ) -> Dict:
        """Remove multibit watermarking by denoising the watermarked image and using the estimated embedding to remodulate
        to remove the watermark

        Args:
            img: image to embed the watermark and attack
            msg: watermark message
            carrier: watermark carrier

        Returns:
            Dict: Dictionary with the following keys:
            - img_wm (Image.Image): Watermarked image
            - img_denoised (Image.Image): Denoised watermarked image
            - img_removed (Image.Image): Watermarked image with the watermark removed
            - psnr_orig_wm (float): PSNR between the original image and the watermarked image
            - eval_orig (torch.Tensor): Embedding of the original image
            - eval_wm (torch.Tensor): Embedding of the watermarked image
            - psnr_wm_removed (float): PSNR between the watermarked image and the watermark removed image
            - eval_removed (torch.Tensor): Embedding of the watermark removed image
            - psnr_orig_denoised (float): PSNR between the original image and the denoised watermarked image
            - eval_denoised (torch.Tensor): Embedding of the denoised watermarked image
            - cosine_orig_wm (float): Cosine similarity between the original image and the watermarked image
            - cosine_orig_denoised (float): Cosine similarity between the original image and the denoised watermarked image
            - cosine_wm_removed (float): Cosine similarity between the watermarked image and the watermark removed image
            - logs (Dict): Logs of the attack
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_multibit(img, msg, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_multibit(
            img, img_wm, msg, carrier
        )

        # step 3: apply wiener2 filter to the watermarked image to remove the watermark
        img_denoised = self.denoise_image(img_wm)

        # step 4: evaluate denoised image
        psnr_orig_denoised, _, eval_denoised = self.evaluate_watermark_multibit(
            img, img_denoised, msg, carrier
        )

        # step 5: estimate embedding
        emb_denoised = self.extract_emb(img_denoised)

        # step 6: watermark the watermarked image with the estimated embedding
        img_wm_denoised, logs = self.watermark_with_estimated(
            img_wm, emb_denoised, carrier, msg, self.params_attack
        )

        # step 7: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_multibit(
            img_wm, img_wm_denoised, msg, carrier
        )

        # step 8: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_denoised = self.extract_emb(img_denoised)
        emb_removed = self.extract_emb(img_wm_denoised)

        cosine_orig_wm = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_denoised = torch.abs(emb @ emb_denoised.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_denoised": img_denoised,
            "img_removed": img_wm_denoised,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_removed": eval_removed,
            "psnr_orig_denoised": psnr_orig_denoised,
            "eval_denoised": eval_denoised,
            "cosine_wm_orig": cosine_orig_wm,
            "cosine_orig_denoised": cosine_orig_denoised,
            "cosine_wm_removed": cosine_wm_removed,
            "cosine_orig_removed": cosine_orig_removed,
            "logs": logs,
        }
