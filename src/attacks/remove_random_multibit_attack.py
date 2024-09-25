from typing import Optional, Dict

from omegaconf import DictConfig
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from ssl_watermarking import encode
from ssl_watermarking import utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img

from .base_multibit_attack import BaseMultibitAttack
from src.utils import SingleImageDataset


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


class RemoveRandomMultibitAttack(BaseMultibitAttack):
    """Remove multibit watermark from an image using a random carrier remodulation"""

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
            self.params_attack = DEFAULT_PARAMS_ATTACK
        else:
            self.params_attack = params_attack

        self.params_attack.batch_size = self.batch_size

        self.target_fpr = 1e-6
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=self.target_fpr)

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

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        msg: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict:
        """Remove multibit watermark from an image using a random carrier remodulation

        Args:
            img: image to watermark and attack
            msg: watermark message
            carrier: watermark carrier

        Returns:
            Dictionary with the following keys:
                - img_wm: Watermarked image
                - img_removed: Image with watermark removed
                - psnr_orig_wm: PSNR between original image and watermarked image
                - eval_orig: Original watermark evaluation
                - eval_wm: Watermarked image evaluation
                - psnr_wm_removed: PSNR between watermarked image and image with watermark removed
                - eval_removed: Image with watermark removed evaluation
                - cosine_wm_removed: Cosine similarity between watermarked image and image with watermark removed
                - cosine_wm_orig: Cosine similarity between original image and watermarked image
                - logs: Attack logs
        """

        # step 1: watermark the image with provided carrier
        img_wm = self.watermark_multibit(img, msg, carrier, self.params_wm)

        # step 2: evaluate the watermark
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_multibit(
            img, img_wm, msg, carrier
        )

        # step 3: randomly generate a new carrier
        carrier_random = wm_utils.generate_carriers(1, self.D).to(self.device)

        # step 4: watermark the watermarked image with the new carrier
        img_wm_random, logs = self.watermark_with_estimated(
            img_wm, carrier_random, carrier, msg, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_multibit(
            img_wm, img_wm_random, msg, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_wm_random = self.extract_emb(img_wm_random)

        cosine = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_wm_random.T).item()
        cosine_wm_random = torch.abs(emb_wm @ emb_wm_random.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_wm_random.T).item()
        cosine_orig_random = torch.abs(emb @ carrier_random.T).item()

        return {
            "img_wm": img_wm,
            "img_removed": img_wm_random,
            "psnr_orig_wm": psnr_orig_wm,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_removed": eval_removed,
            "cosine_wm_random": cosine_wm_random,
            "cosine_wm_orig": cosine,
            "cosine_orig_removed": cosine_orig_removed,
            "cosine_wm_removed": cosine_wm_removed,
            "cosine_orig_random": cosine_orig_random,
            "logs": logs,
        }
