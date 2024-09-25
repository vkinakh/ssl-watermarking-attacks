import os
import sys

# add submodules to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from ssl_watermarking import utils as wm_utils
    from ssl_watermarking import utils_img as wm_utils_img
except ImportError as e:
    print(f"Error importing ssl_watermarking.utils: {e}")
    sys.exit(1)

sys.modules["utils"] = wm_utils
sys.modules["utils_img"] = wm_utils_img

import argparse
from typing import Dict, Literal
import random

from omegaconf import DictConfig
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ssl_watermarking import utils as wm_utils

from src.attacks import (
    Copy0BitAttack,
    CopyMultibitAttack,
    RemoveDenoise0BitAttack,
    RemoveDenoiseMultibitAttack,
    RemoveRandom0BitAttack,
    RemoveRandomMultibitAttack,
    RemoveOther0BitAttack,
    RemoveOtherMultibitAttack,
    UntargetedRemoveObitAttack,
    UntargetedRemoveMultibitAttack,
)
from src.utils import find_images_in_path, seed_everything

DEFAULT_MODEL = "resnet50"
DEFAULT_PATH_BACKBONE = "./models/dino_r50_plus.pth"
DEFAULT_PATH_NORM_LAYER = "./normlayers/out2048_yfcc_orig.pth"
ATTACK_TYPES = [
    "copy_0bit",
    "copy_multibit",
    "remove_denoise_0bit",
    "remove_denoise_multibit",
    "remove_other_0bit",
    "remove_other_multibit",
    "remove_random_0bit",
    "remove_random_multibit",
    "untargeted_remove_0bit",
    "untargeted_remove_multibit",
]

ATTACK_TYPE = Literal[
    "copy_0bit",
    "copy_multibit",
    "remove_denoise_0bit",
    "remove_denoise_multibit",
    "remove_other_0bit",
    "remove_other_multibit",
    "remove_random_0bit",
    "remove_random_multibit",
    "untargeted_remove_0bit",
    "untargeted_remove_multibit",
]


def get_parser():
    parser = argparse.ArgumentParser(description="Run watermark attacks")
    parser.add_argument("--attack", type=str, required=True, choices=ATTACK_TYPES)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of the encoder model to use for watermarking",
    )
    parser.add_argument(
        "--path_backbone",
        type=str,
        default=DEFAULT_PATH_BACKBONE,
        help="Path to the .pth file with the backbone weights",
    )
    parser.add_argument(
        "--path_norm_layer",
        type=str,
        default=DEFAULT_PATH_NORM_LAYER,
        help="Path to the .pth file with the norm layer weights",
    )
    parser.add_argument(
        "--path_images", type=str, required=True, help="Path to directory with images"
    )
    parser.add_argument(
        "--path_outputs", type=str, required=True, help="Path to .csv file with outputs"
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="none",
        choices=["none", "all"],
        help="Transformation to apply to the images during watermarking",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--psnr_wm",
        type=float,
        default=42.0,
        help="Target PSNR between original and watermarked images",
    )
    parser.add_argument(
        "--psnr_attack",
        type=float,
        default=42.0,
        help="Target PSNR between watermarked and attacked images",
    )
    parser.add_argument(
        "--lambda_w", type=float, default=5e4, help="Weight for the attack loss"
    )
    parser.add_argument(
        "--lambda_i", type=float, default=1.0, help="Weight for the identity loss"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train the attack"
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=1e-6,
        help="Target FPR for watermark and attack. Only for 0-bit watermarking",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=10,
        help="Number of bits in the watermark. Only for multibit watermarking",
    )
    parser.add_argument(
        "--use_cosine_sim", action="store_true", help="Use cosine similarity for attack"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam,lr=0.01", help="Optimizer for the attack"
    )
    parser.add_argument(
        "--scheduler", type=str, default=None, help="Scheduler for the attack"
    )
    parser.add_argument(
        "--wiener_filter_size",
        type=int,
        default=25,
        help="Size of the Wiener filter in denoise remove attack",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducibility"
    )

    return parser


def verify_args(args):
    if "0bit" in args.attack and args.target_fpr is None:
        raise ValueError("Target FPR is required for 0-bit watermarking attacks")

    if "multibit" in args.attack and args.num_bits is None:
        raise ValueError(
            "Number of bits is required for multi-bit watermarking attacks"
        )

    if "denoise" in args.attack and args.wiener_filter_size is None:
        raise ValueError("Wiener filter size is required for denoise attacks")

    if not args.path_outputs.endswith(".csv"):
        raise ValueError("Output path must be a .csv file")

    if args.psnr_wm < 0 or args.psnr_attack < 0:
        raise ValueError("PSNR values must be positive")

    if args.lambda_w < 0 or args.lambda_i < 0:
        raise ValueError("Lambda values must be positive")

    if args.epochs < 0:
        raise ValueError("Number of epochs must be positive")

    if args.target_fpr < 0:
        raise ValueError("Target FPR must be positive")

    if args.num_bits < 0:
        raise ValueError("Number of bits must be positive")

    # check if backbones and norm layers exist
    if not os.path.exists(args.path_backbone):
        raise ValueError("Backbone file does not exist")

    if not os.path.exists(args.path_norm_layer):
        raise ValueError("Norm layer file does not exist")


def parse_results(result: Dict, attack_type: ATTACK_TYPE) -> Dict:
    curr_result = {
        "psnr_orig_wm": result["psnr_orig_wm"],
        "cosine_wm_orig": result["cosine_wm_orig"],
    }

    if "0bit" in attack_type:
        curr_result = {
            **curr_result,
            "R_orig": result["eval_orig"]["R"],
            "cosine_orig": result["eval_orig"]["cosine"],
            "log10_pvalue_orig": result["eval_orig"]["log10_pvalue"],
            "decision_orig": result["eval_orig"]["decision"],
            "R_wm": result["eval_wm"]["R"],
            "cosine_wm": result["eval_wm"]["cosine"],
            "log10_pvalue_wm": result["eval_wm"]["log10_pvalue"],
            "decision_wm": result["eval_wm"]["decision"],
            "cosine_carrier_wm": result["cosine_carrier_wm"],
        }

        if "copy" in attack_type:
            curr_result = {
                **curr_result,
                "R_copy": result["eval_copy"]["R"],
                "cosine_copy": result["eval_copy"]["cosine"],
                "log10_pvalue_copy": result["eval_copy"]["log10_pvalue"],
                "decision_copy": result["eval_copy"]["decision"],
                "cosine_carrier_copy": result["cosine_carrier_copy"],
            }

        if "remove" in attack_type:
            curr_result = {
                **curr_result,
                "R_removed": result["eval_removed"]["R"],
                "cosine_removed": result["eval_removed"]["cosine"],
                "log10_pvalue_removed": result["eval_removed"]["log10_pvalue"],
                "decision_removed": result["eval_removed"]["decision"],
                "cosine_wm_removed": result["cosine_wm_removed"],
                "cosine_carrier_removed": result["cosine_carrier_removed"],
                "cosine_orig_removed": result["cosine_orig_removed"],
            }

        if "denoise" in attack_type:
            curr_result = {
                **curr_result,
                "R_denoised": result["eval_denoised"]["R"],
                "cosine_denoised": result["eval_denoised"]["cosine"],
                "log10_pvalue_denoised": result["eval_denoised"]["log10_pvalue"],
                "decision_denoised": result["eval_denoised"]["decision"],
                "cosine_wm_denoised": result["cosine_wm_denoised"],
                "cosine_orig_denoised": result["cosine_orig_denoised"],
            }

        if "other" in attack_type:
            curr_result = {
                **curr_result,
                "cosine_carrier_other": result["cosine_carrier_other"],
            }

        logs = result["logs"]
        for (_, step), log in logs.items():
            for key, val in log.items():
                if key in [
                    "decision",
                    "dot_product",
                    "dot_product_gt",
                    "R_avg",
                    "cosine_avg",
                    "log10_pvalue_avg",
                ]:
                    curr_result[f"{key}_{step}"] = val

    elif "multibit" in attack_type:
        curr_result = {
            **curr_result,
            "bit_acc_orig": result["eval_orig"]["bit_acc"],
            "bit_acc_wm": result["eval_wm"]["bit_acc"],
        }

        if "copy" in attack_type:
            curr_result = {
                **curr_result,
                "bit_acc_copy": result["eval_copy"]["bit_acc"],
                "cosine_orig_copy": result["cosine_orig_copy"],
            }

        if "denoise" in attack_type:
            curr_result = {
                **curr_result,
                "bit_acc_denoised": result["eval_denoised"]["bit_acc"],
                "cosine_orig_denoised": result["cosine_orig_denoised"],
            }

        if "remove" in attack_type:
            curr_result = {
                **curr_result,
                "bit_acc_removed": result["eval_removed"]["bit_acc"],
                "cosine_orig_removed": result["cosine_orig_removed"],
            }

        logs = result["logs"]
        for (_, step), log in logs.items():
            for key, val in log.items():
                if key in ["bit_acc_avg"]:
                    curr_result[f"{key}_{step}"] = val

    if "copy" in attack_type:
        curr_result = {
            **curr_result,
            "psnr_copy": result["psnr_copy"],
            "cosine_wm_copy": result["cosine_wm_copy"],
        }

    if "remove" in attack_type:
        curr_result = {
            **curr_result,
            "psnr_wm_removed": result["psnr_wm_removed"],
            "cosine_wm_removed": result["cosine_wm_removed"],
        }

    if "denoise" in attack_type:
        curr_result = {
            **curr_result,
            "psnr_orig_denoised": result["psnr_orig_denoised"],
            "cosine_orig_denoised": result["cosine_orig_denoised"],
        }

    if "other" in attack_type:
        curr_result = {
            **curr_result,
            "cosine_orig_other": result["cosine_orig_other"],
            "cosine_wm_other": result["cosine_wm_other"],
        }

    if "random" in attack_type:
        curr_result = {
            **curr_result,
            "cosine_orig_random": result["cosine_orig_random"],
            "cosine_wm_random": result["cosine_wm_random"],
        }

    return curr_result


def get_attack(attack_type: ATTACK_TYPE, args):
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Invalid attack type: {attack_type}")

    params_attack = DictConfig(
        {
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "epochs": args.epochs,
            "target_psnr": args.psnr_attack,
            "lambda_w": args.lambda_w,
            "lambda_i": args.lambda_i,
            "verbose": args.verbose,
        }
    )

    if "0bit" in attack_type:
        params_wm = DictConfig(
            {
                "optimizer": "Adam,lr=0.01",
                "scheduler": None,
                "epochs": 100,
                "target_psnr": args.psnr_wm,
                "lambda_w": 1.0,
                "lambda_i": 1.0,
                "verbose": 1,
            }
        )
    elif "multibit" in attack_type:
        params_wm = DictConfig(
            {
                "optimizer": "Adam,lr=0.01",
                "scheduler": None,
                "epochs": 100,
                "target_psnr": args.psnr_wm,
                "lambda_w": 5e4,
                "lambda_i": 1.0,
                "verbose": 1,
            }
        )
    else:
        raise ValueError(f"Invalid attack type: {attack_type}")

    if attack_type == "copy_0bit":
        attack = Copy0BitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            target_fpr=args.target_fpr,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "copy_multibit":
        attack = CopyMultibitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_denoise_0bit":
        attack = RemoveDenoise0BitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            target_fpr=args.target_fpr,
            use_cosine_sim=args.use_cosine_sim,
            filter_size=args.wiener_filter_size,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_denoise_multibit":
        attack = RemoveDenoiseMultibitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            use_cosine_sim=args.use_cosine_sim,
            filter_size=args.wiener_filter_size,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_other_0bit":
        attack = RemoveOther0BitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            target_fpr=args.target_fpr,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_other_multibit":
        attack = RemoveOtherMultibitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_random_0bit":
        attack = RemoveRandom0BitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            target_fpr=args.target_fpr,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "remove_random_multibit":
        attack = RemoveRandomMultibitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            use_cosine_sim=args.use_cosine_sim,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "untargeted_remove_0bit":
        attack = UntargetedRemoveObitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            target_fpr=args.target_fpr,
            params_wm=params_wm,
            params_attack=params_attack,
        )
    elif attack_type == "untargeted_remove_multibit":
        attack = UntargetedRemoveMultibitAttack(
            name_model=args.model,
            path_backbone=args.path_backbone,
            path_norm_layer=args.path_norm_layer,
            device=args.device,
            transform=args.transform,
            params_wm=params_wm,
            params_attack=params_attack,
        )

    return attack, params_wm, params_attack


def run_attack(args):
    attack_type = args.attack
    seed = args.seed

    if seed is not None:
        seed_everything(seed)

    attack, params_wm, params_attack = get_attack(attack_type, args)
    paths_images = find_images_in_path(args.path_images)

    if "copy" in attack_type or "other" in attack_type:
        # shuffle the images
        paths_images_shuffle = paths_images.copy()
        random.shuffle(paths_images_shuffle)

    # if multi-bit watermarking, set the number of bits
    k = args.num_bits if "multibit" in attack_type else 1

    results = []
    for i, path_image in enumerate(tqdm(paths_images, desc="Running attack")):
        img = Image.open(path_image)

        params_input = {
            "img": img,
        }

        if "copy" in attack_type or "other" in attack_type:
            path_image_other = paths_images_shuffle[i]
            img_other = Image.open(path_image_other)

            key = "img_to_copy" if "copy" in attack_type else "img_other"
            params_input = {
                **params_input,
                key: img_other,
            }

        if seed is not None:
            seed_everything(seed)

        carrier = wm_utils.generate_carriers(k, attack.D).to(attack.device)
        params_input = {
            **params_input,
            "carrier": carrier,
        }

        if "multibit" in attack_type:
            msg = wm_utils.generate_messages(1, k).to(attack.device)

            params_input = {
                **params_input,
                "msg": msg,
            }

        result = attack.attack(**params_input)
        result_parsed = parse_results(result, attack_type)
        results.append(result_parsed)

    results = pd.DataFrame(results)
    results.to_csv(args.path_outputs, index=False)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    verify_args(args)
    run_attack(args)
