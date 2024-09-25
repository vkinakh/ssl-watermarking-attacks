from typing import Dict, Optional

from omegaconf import DictConfig
from PIL import Image

import torch

from .base_multibit_attack import BaseMultibitAttack


DEFAULT_ATTACK_PARAMS = DictConfig(
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


class CopyMultibitAttack(BaseMultibitAttack):
    """Copy attack for multi-bit watermarking."""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
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
            use_cosine_sim,
            params_wm,
        )

        if params_attack is None:
            self.params_attack = DEFAULT_ATTACK_PARAMS
        else:
            self.params_attack = params_attack

        self.params_attack.batch_size = self.batch_size

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        msg: torch.Tensor,
        img_to_copy: Image.Image,
    ) -> Dict:
        """Runs the copy attack: copy the watermark from one image to another.

        Args:
            img: image to watermark with the carrier
            msg: message to embed in the image
            carrier: watermark carrier to embed in the image
            img_to_copy: image to copy the watermark to

        Returns:
            Dict: dictionary with the following keys
                - img_wm: watermarked image
                - img_copy: copied watermarked image
                - psnr_orig_wm: PSNR between the original and watermarked images
                - eval_orig: evaluation of the original image
                - eval_wm: evaluation of the watermarked image
                - psnr_copy: PSNR between the original and copied watermarked images
                - eval_copy: evaluation of the copied watermarked image
                - cosine_orig_wm: cosine similarity between the original and watermarked embeddings
                - cosine_orig_copy: cosine similarity between the original and copied watermarked embeddings
                - logs: logs from the watermarking attack
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_multibit(img, msg, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_multibit(
            img, img_wm, msg, carrier
        )

        # step 3: extract the embedding from the watermarked image
        carrier_est = self.extract_emb(img_wm)

        # step 4: watermark the watermarked image with the estimated embedding
        img_wm_copy, logs = self.watermark_with_estimated(
            img_to_copy, carrier_est, carrier, msg, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_copy, _, eval_copy = self.evaluate_watermark_multibit(
            img_to_copy, img_wm_copy, msg, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_wm_copy = self.extract_emb(img_wm_copy)

        cosine_wm_orig = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_copy = torch.abs(emb @ emb_wm_copy.T).item()
        cosine_wm_copy = torch.abs(emb_wm @ emb_wm_copy.T).item()

        return {
            "img_wm": img_wm,
            "img_copy": img_wm_copy,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_copy": psnr_copy,
            "eval_copy": eval_copy,
            "cosine_wm_orig": cosine_wm_orig,
            "cosine_wm_copy": cosine_wm_copy,
            "cosine_orig_copy": cosine_orig_copy,
            "logs": logs,
        }
