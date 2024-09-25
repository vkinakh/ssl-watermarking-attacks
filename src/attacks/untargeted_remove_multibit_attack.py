from typing import Dict

from omegaconf import DictConfig
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from ssl_watermarking import utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img

from .base_multibit_attack import BaseMultibitAttack
from .encode import watermark_multibit_untargeted_attack
from src.utils import SingleImageDataset


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


class UntargetedRemoveMultibitAttack(BaseMultibitAttack):
    """Untargeted remove multibit watermarking by embedding the watermark of another image and remodulate to remove the
    watermark"""

    def __init__(
        self,
        name_model: str,
        path_backbone: str,
        path_norm_layer: str,
        device: str = "cuda",
        transform: str = "none",
        params_wm: DictConfig = None,
        params_attack: DictConfig = None,
    ):
        super().__init__(
            name_model, path_backbone, path_norm_layer, device, transform, params_wm
        )

        if params_attack is None:
            self.params_attack = DEFAULT_PARAMS_WATERMARK
        else:
            self.params_attack = params_attack

        self.params_attack.batch_size = self.batch_size
        self.target_fpr = 1e-6
        self.angle = wm_utils.pvalue_angle(dim=self.D, k=1, proba=self.target_fpr)

    def remove_watermark(
        self,
        img: Image.Image,
        carrier_est: torch.Tensor,
        carrier_gt: torch.Tensor,
        msgs: torch.Tensor,
        params_wm: DictConfig,
    ):
        ds = SingleImageDataset(img, wm_utils_img.default_transform)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        img_wm_pt, logs = watermark_multibit_untargeted_attack(
            dl,
            carrier_est,
            carrier_gt,
            msgs,
            self.model,
            self.transform,
            params_wm,
            return_logs=True,
        )
        img_wm_pt = img_wm_pt[0]
        img_wm = wm_utils_img.unnormalize_img(img_wm_pt).squeeze(0)
        return TF.to_pil_image(img_wm), logs

    def attack(
        self,
        img: Image.Image,
        carrier: torch.Tensor,
        msg: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict:
        """Remove the watermark from the watermarked image by embedding the watermark of another image and remodule to
        remove the watermark

        Args:
            img: image to watermark and attack
            msg: message to embed
            carrier: watermark carrier

        Returns:
            Dict: dictionary with the following keys
                - img_wm: watermarked image
                - img_removed: image with the watermark removed
                - psnr_orig_wm: PSNR between the original and watermarked image
                - psnr_wm_removed: PSNR between the watermarked and watermark removed image
                - eval_orig: evaluation of the original image
                - eval_wm: evaluation of the watermarked image
                - eval_removed: evaluation of the watermark removed image
                - logs: logs of the attack
                - cosine_removed: cosine similarity between the estimated and removed carrier
        """

        # step 1:  watermark to image with provided message and carrier
        img_wm = self.watermark_multibit(img, msg, carrier, self.params_wm)

        # step 2:  remove watermark from image
        psnr_orig_wm, eval_orig, eval_wm = self.evaluate_watermark_multibit(
            img, img_wm, msg, carrier
        )

        # step 3: estimated carrier is the image embedding
        emb = self.extract_emb(img_wm)

        # step 4: remove watermark from image
        img_removed, logs = self.remove_watermark(
            img_wm, emb, carrier, msg, self.params_attack
        )

        # step 5: evaluate the watermark
        psnr_wm_removed, _, eval_removed = self.evaluate_watermark_multibit(
            img_wm, img_removed, msg, carrier
        )

        # step 6: compare embedding
        emb = self.extract_emb(img)
        emb_wm = self.extract_emb(img_wm)
        emb_removed = self.extract_emb(img_removed)

        cosine_orig_wm = torch.abs(emb @ emb_wm.T).item()
        cosine_orig_removed = torch.abs(emb @ emb_removed.T).item()
        cosine_wm_removed = torch.abs(emb_wm @ emb_removed.T).item()

        return {
            "img_wm": img_wm,
            "img_removed": img_removed,
            "psnr_orig_wm": psnr_orig_wm,
            "psnr_wm_removed": psnr_wm_removed,
            "eval_orig": eval_orig,
            "eval_wm": eval_wm,
            "eval_removed": eval_removed,
            "cosine_wm_orig": cosine_orig_wm,
            "cosine_orig_removed": cosine_orig_removed,
            "cosine_wm_removed": cosine_wm_removed,
            "logs": logs,
        }
