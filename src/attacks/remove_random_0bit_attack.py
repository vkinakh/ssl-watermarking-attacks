from typing import Optional, Dict

from omegaconf import DictConfig
from PIL import Image

import torch

from ssl_watermarking import utils as wm_utils

from .base_0bit_attack import Base0BitAttack, DEFAULT_0BIT_WM_PARAMS


class RemoveRandom0BitAttack(Base0BitAttack):
    """Remove 0-bit watermark from an image using a random carrier remodulation"""

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

    def attack(self, img: Image.Image, carrier: torch.Tensor, *args, **kwargs) -> Dict:
        """Remove 0-bit watermark from an image using a random carrier remodulation

        Args:
            img: image to watermark and attack
            carrier: watermark carrier

        Returns:
            Dictionary with the following keys:
                - img_wm: Watermarked image
                - img_removed: Image with watermark removed
                - psnr_orig_wm: PSNR between original image and watermarked image
                - eval_orig: Original watermark evaluation
                - eval_wm: Watermarked image evaluation
                - psnr_wm_removed: PSNR between watermarked image and image with watermark removed
                - eval_removed: Evaluation of image with watermark removed
                - logs: Attack logs
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_0bit(img, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_0bit(
            img, img_wm, carrier
        )

        # step 3: randomly generate a new carrier
        carrier_random = wm_utils.generate_carriers(1, self.D).to(self.device)

        # step 4: watermark the watermarked image with the new carrier
        img_wm_random, logs = self.watermark_with_estimated(
            img_wm, carrier_random, carrier, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_0bit(
            img_wm, img_wm_random, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_removed = self.extract_emb(img_wm_random)

        cosine_wm_orig = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_removed.T).item()
        cosine_orig_random = torch.abs(emb @ carrier_random.T).item()
        cosine_wm_random = torch.abs(emb_wm @ carrier_random.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()
        cosine_carrier_wm = torch.abs(carrier @ emb_wm.T).item()
        cosine_carrier_removed = torch.abs(carrier @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_removed": img_wm_random,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_removed": eval_removed,
            "cosine_wm_orig": cosine_wm_orig,
            "cosine_orig_removed": cosine_orig_removed,
            "cosine_orig_random": cosine_orig_random,
            "cosine_wm_random": cosine_wm_random,
            "cosine_wm_removed": cosine_wm_removed,
            "cosine_carrier_wm": cosine_carrier_wm,
            "cosine_carrier_removed": cosine_carrier_removed,
            "logs": logs,
        }
