from typing import Dict

from omegaconf import DictConfig
from PIL import Image

import torch

from ssl_watermarking import utils as wm_utils

from .base_multibit_attack import BaseMultibitAttack


DEFAULT_PARAMS_ATTACK = DictConfig(
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


class RemoveOtherMultibitAttack(BaseMultibitAttack):
    """Remove multibit watermarking by embedding the watermark of another image and remodulate to remove the
    watermark"""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        use_cosine_sim: bool = False,
        params_wm: DictConfig = None,
        params_attack: DictConfig = None,
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
            self.params_attack = DEFAULT_PARAMS_ATTACK
        else:
            self.params_attack = params_attack

        self.params_attack.batch_size = self.batch_size

        self.target_fpr = 1e-6
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=self.target_fpr)

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        msg: torch.Tensor,
        img_other: Image.Image,
        *args,
        **kwargs
    ) -> Dict:
        """Remove the watermark from the watermarked image by embedding the watermark of another image and remodule to
        remove the watermark

        Args:
            img: image to watermark and attack
            img_other: other image to extract embedding from
            msg: message to embed
            carrier: watermark carrier

        Returns:
            Dict: dictionary with the following keys
                - img_wm: watermarked image
                - img_other: other image
                - img_removed: image with the watermark removed
                - psnr_orig_wm: PSNR between the original and watermarked image
                - eval_orig: evaluation of the original image
                - eval_wm: evaluation of the watermarked image
                - psnr_wm_removed: PSNR between the watermarked and watermark removed image
                - eval_removed: evaluation of the watermark removed image
                - logs: logs of the attack
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_multibit(img, msg, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_multibit(
            img, img_wm, msg, carrier
        )

        # step 3: new carrier as embedding of other image
        emb_other = self.extract_emb(img_other)

        # step 4: watermark with estimated carrier
        img_wm_other, logs = self.watermark_with_estimated(
            img_wm, emb_other, carrier, msg, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_multibit(
            img_wm, img_wm_other, msg, carrier
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
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_other": img_other,
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
            "cosine_wm_removed": cosine_wm_removed,
            "logs": logs,
        }
