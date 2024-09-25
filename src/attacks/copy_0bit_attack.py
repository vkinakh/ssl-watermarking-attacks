from typing import Optional, Dict

from omegaconf import DictConfig
from PIL import Image

import torch

from src.attacks.base_0bit_attack import Base0BitAttack


DEFAULT_ATTACK_PARAMS = DictConfig(
    {
        "optimizer": "Adam,lr=0.01",
        "scheduler": None,
        "epochs": 25,
        "target_psnr": 42.0,
        "lambda_w": 1.0,
        "lambda_i": 1.0,
        "verbose": 1,
    }
)


class Copy0BitAttack(Base0BitAttack):
    """Copy attack for 0-bit watermarking."""

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
            self.params_attack = DEFAULT_ATTACK_PARAMS
        else:
            self.params_attack = params_attack

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        img_to_copy: Image.Image,
        *args,
        **kwargs
    ) -> Dict:
        """Runs the copy attack: copy the watermark from one image to another.

        Args:
            img: image to watermark with the carrier
            carrier: watermark carrier to embed in the image
            img_to_copy: image to copy the watermark to

        Returns:
            Dictionary with the following
            - img_wm: watermarked image
            - img_copy: image with the watermark copied
            - psnr_orig_wm: PSNR between the original image and the watermarked image
            - eval_orig: evaluation of the original watermark
            - eval_wm: evaluation of the watermarked image
            - psnr_copy: PSNR between the original image and the copied watermark image
            - eval_copy: evaluation of the copied watermark
            - logs_copy: logs from the watermarking attack
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_0bit(img, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_0bit(
            img, img_wm, carrier
        )

        # step 3: extract the embedding from the watermarked image
        carrier_est = self.extract_emb(img_wm)

        # step 4: watermark the watermarked image with the estimated embedding
        img_wm_copy, logs_copy = self.watermark_with_estimated(
            img_to_copy, carrier_est, carrier, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_copy, _, eval_copy = self.evaluate_watermark_0bit(
            img_to_copy, img_wm_copy, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_copy = self.extract_emb(img_wm_copy)

        cosine_wm_orig = torch.abs(emb @ emb_wm.T).item()
        cosine_wm_copy = torch.abs(emb @ emb_copy.T).item()
        cosine_carrier_wm = torch.abs(carrier @ emb_wm.T).item()
        cosine_carrier_copy = torch.abs(carrier @ emb_copy.T).item()

        return {
            "img_wm": img_wm,
            "img_copy": img_wm_copy,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_copy": psnr_copy,
            "eval_copy": eval_copy,
            "logs": logs_copy,
            "cosine_wm_orig": cosine_wm_orig,
            "cosine_wm_copy": cosine_wm_copy,
            "cosine_carrier_wm": cosine_carrier_wm,
            "cosine_carrier_copy": cosine_carrier_copy,
        }
