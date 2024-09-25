from typing import Optional, Dict

import numpy as np
from omegaconf import DictConfig
from PIL import Image

import torch

from .base_0bit_attack import Base0BitAttack, DEFAULT_0BIT_WM_PARAMS
from src.utils import UINT8_MAX, wiener2


class RemoveDenoise0BitAttack(Base0BitAttack):
    """Remove 0-bit watermarking by denoising the watermarked image and using the estimated embedding to remodulate
    to remove the watermark"""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        target_fpr: float = 1e-6,
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
            target_fpr,
            params_wm,
            use_cosine_sim,
        )

        if params_attack is None:
            self.params_attack = DEFAULT_0BIT_WM_PARAMS
        else:
            self.params_attack = params_attack

        self.filter_size = filter_size

    def denoise_image(self, img: Image.Image) -> Image.Image:
        denoise_params = {"local_mean_filter_size": self.filter_size, "noise_var": None}

        img_np = np.array(img)
        img_np = img_np / UINT8_MAX
        img_np = wiener2(img_np, **denoise_params)
        img_np = np.clip(img_np * UINT8_MAX, 0, UINT8_MAX).astype(np.uint8)
        return Image.fromarray(img_np)

    def attack(self, img: Image.Image, carrier: torch.Tensor, *args, **kwargs) -> Dict:
        """Remove 0-bit watermarking by denoising the watermarked image and using the estimated embedding to remodulate
        to remove the watermark

        Args:
            img: img to embed the watermark and attack
            carrier: watermark carrier to embed in the image

        Returns:
            Dict: Results of the attack with the following keys:
                - img_wm (Image.Image): Watermarked image
                - img_denoised (Image.Image): Denoised watermarked image
                - img_removed (Image.Image): Watermark removed image
                - psnr_orig_wm (float): PSNR between the original image and the watermarked image
                - eval_orig (Dict): Evaluation of the original image
                - eval_wm (Dict): Evaluation of the watermarked image
                - psnr_orig_denoised (float): PSNR between the original image and the denoised watermarked image
                - eval_denoised (Dict): Evaluation of the denoised watermarked image
                - psnr_wm_removed (float): PSNR between the watermarked image and the watermark removed image
                - eval_removed (Dict): Evaluation of the watermark removed image
                - logs (Dict): Logs of the
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_0bit(img, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_0bit(
            img, img_wm, carrier
        )

        # step 3: apply wiener2 filter to the watermarked image to remove the watermark
        img_denoised = self.denoise_image(img_wm)

        # step 4: evaluate denoised image
        psnr_orig_denoised, _, eval_denoised = self.evaluate_watermark_0bit(
            img, img_denoised, carrier
        )

        # step 4: estimate embedding
        emb_denoised = self.extract_emb(img_denoised)

        # step 5: watermark the watermarked image with the estimated embedding
        img_wm_denoised, logs = self.watermark_with_estimated(
            img_wm, emb_denoised, carrier, self.params_attack
        )

        # step 6: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_0bit(
            img_wm, img_wm_denoised, carrier
        )

        # step 7: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_denoised = self.extract_emb(img_wm_denoised)
        emb_removed = self.extract_emb(img_denoised)

        cosine_wm_orig = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_removed.T).item()
        cosine_orig_denoised = torch.abs(emb @ emb_denoised.T).item()
        cosine_wm_denoised = torch.abs(emb_wm @ emb_denoised.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()
        cosine_carrier_wm = torch.abs(carrier @ emb_wm.T).item()
        cosine_carrier_denoised = torch.abs(carrier @ emb_denoised.T).item()
        cosine_carrier_removed = torch.abs(carrier @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_denoised": img_denoised,
            "img_removed": img_wm_denoised,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_orig_denoised": psnr_orig_denoised,
            "eval_denoised": eval_denoised,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_removed": eval_removed,
            "cosine_wm_orig": cosine_wm_orig,
            "cosine_wm_denoised": cosine_wm_denoised,
            "cosine_orig_denoised": cosine_orig_denoised,
            "cosine_orig_removed": cosine_orig_removed,
            "cosine_wm_removed": cosine_wm_removed,
            "cosine_carrier_wm": cosine_carrier_wm,
            "cosine_carrier_denoised": cosine_carrier_denoised,
            "cosine_carrier_removed": cosine_carrier_removed,
            "logs": logs,
        }
