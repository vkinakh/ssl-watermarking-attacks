from typing import Optional, Dict

from omegaconf import DictConfig
from PIL import Image

import torch

from .base_0bit_attack import Base0BitAttack, DEFAULT_0BIT_WM_PARAMS


class RemoveOther0BitAttack(Base0BitAttack):
    """Remove 0-bit watermarking by embedding the watermark of another image and remodulate to remove the watermark"""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        target_fpr: float = 1e-6,
        use_cosine_sim: bool = False,
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

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        img_other: Image.Image,
        *args,
        **kwargs
    ) -> Dict:
        """Remove the watermark from the watermarked image by embedding the watermark of another image and remodule to
        remove the watermark

        Args:
            img: image to watermark and attack
            img_other: other image to extract embedding from
            carrier: watermark carrier

        Returns:
            Dict: dictionary with the following keys:
                - img_wm: watermarked image
                - img_removed: image with the watermark removed
                - psnr_orig_wm: PSNR between the original and watermarked image
                - eval_orig: evaluation of the original image
                - eval_wm: evaluation of the watermarked image
                - psnr_wm_removed: PSNR between the watermarked and watermark removed image
                - eval_removed: evaluation of the watermark removed image
                - logs: logs from the watermarking process
        """

        # step 1: watermark the image
        img_wm = self.watermark_0bit(img, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_0bit(
            img, img_wm, carrier
        )

        # step 3: new carrier - embedding of the other image
        emb_other = self.extract_emb(img_other)

        # step 4: watermark the watermarked images with the embedding of the other image
        img_wm_other, logs = self.watermark_with_estimated(
            img_wm, emb_other, carrier, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_0bit(
            img_wm, img_wm_other, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_other = self.extract_emb(img_other)
        emb_removed = self.extract_emb(img_wm_other)

        cosine_wm_orig = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_removed.T).item()
        cosine_orig_other = torch.abs(emb @ emb_other.T).item()
        cosine_wm_other = torch.abs(emb_wm @ emb_other.T).item()
        cosine_carrier_wm = torch.abs(carrier @ emb_wm.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()
        cosine_carrier_other = torch.abs(carrier @ emb_other.T).item()
        cosine_carrier_removed = torch.abs(carrier @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_removed": img_wm_other,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_removed": eval_removed,
            "cosine_wm_orig": cosine_wm_orig,
            "cosine_orig_removed": cosine_orig_removed,
            "cosine_orig_other": cosine_orig_other,
            "cosine_wm_other": cosine_wm_other,
            "cosine_carrier_wm": cosine_carrier_wm,
            "cosine_wm_removed": cosine_wm_removed,
            "cosine_carrier_other": cosine_carrier_other,
            "cosine_carrier_removed": cosine_carrier_removed,
            "logs": logs,
        }
