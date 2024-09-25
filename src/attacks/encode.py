from typing import Callable
from pprint import pprint

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ssl_watermarking.encode import build_optimizer, build_lr_scheduler
from ssl_watermarking import utils as wm_utils
from ssl_watermarking import utils_img as wm_utils_img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def watermark_0bit_attack(
    img_loader: DataLoader,
    carrier_attack: torch.Tensor,
    carrier_gt: torch.Tensor,
    angle: float,
    model: nn.Module,
    transform: Callable,
    params,
    return_logs: bool = False,
    use_cosine_sim: bool = False,
    select_best: bool = False,
):
    """
    0-bit watermarking adapted for attack purposes.

    Args:
        img_loader: Dataloader of the images to be watermarked
        carrier_attack (tensor of size 1xD): Attack carrier. Hypercone direction 1xD
        carrier_gt (tensor of size 1xD): Ground truth carrier. Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        transform: Differentiable augmentation with fixed output size -> 1xCxWxH
        params: Must contain optimizer, scheduler, epochs, lambda_w, lambda_i, verbose
        return_logs: If True, returns logs
        use_cosine_sim: If True, uses cosine similarity instead of the original loss
        select_best: If True, selects the best image in the batch

    Returns:
        imgs: Watermarked images as a list of unnormalized (distributed around [-1, 1]) pytorch tensors
        Optional[Dict]: Logs if return_logs is True
    """
    rho = 1 + np.tan(angle) ** 2
    ssim = wm_utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    if select_best:
        best_cosine_sim = 0
        best_batch = None

    logs_all = {}
    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1] * img.shape[-2] for img in images])
        if max_res > 1e6:
            print(
                "WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU."
            )

        # load images
        batch_imgs_orig = [
            x.to(device, non_blocking=True).unsqueeze(0) for x in images
        ]  # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig]  # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        optimizer = build_optimizer(
            model_params=batch_imgs, **wm_utils.parse_params(params.optimizer)
        )
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, **wm_utils.parse_params(params.scheduler)
            )

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                # concentrate changes around edges
                x = ssim.apply(x, batch_imgs_orig[ii])
                # remain within PSNR budget
                x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii == 0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0)  # BxCxWxH
            # get features
            ft = model(batch)  # BxCxWxH -> BxD
            norm = torch.norm(ft, dim=-1, keepdim=True)  # BxD -> Bx1

            # compute losses
            if use_cosine_sim:
                ft_norm = ft / norm
                dot_product = ft_norm @ carrier_attack.T  # BxD @ Dx1 -> Bx1
                loss_w = torch.sum(1 - dot_product)

                if select_best:
                    cosines = torch.mean(dot_product)
                    if cosines > best_cosine_sim:
                        best_cosine_sim = cosines
                        best_batch = batch_imgs
            else:
                dot_product = ft @ carrier_attack.T  # BxD @ Dx1 -> Bx1
                loss_w = torch.sum(-(rho * dot_product**2 - norm**2))  # B-B -> B

            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += (
                    torch.norm(batch_imgs[ii] - batch_imgs_orig[ii]) ** 2
                )  # CxWxH -> 1
            loss = params.lambda_w * loss_w + params.lambda_i * loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose > 1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }

                pprint(logs)

                if params.verbose > 2:
                    dot_product_gt = ft @ carrier_gt.T
                    rs_gt = rho * dot_product_gt**2 - norm**2  # Bx1-Bx1 -> Bx1
                    cosines_gt = torch.abs(dot_product_gt / norm)  # Bx1/Bx1 -> Bx1
                    log10_pvalues = [
                        np.log10(
                            wm_utils.cosine_pvalue(cosines_gt[ii].item(), ft.shape[-1])
                        )
                        for ii in range(len(batch_imgs))
                    ]
                    logs["R_avg"] = torch.mean(rs_gt).item()
                    logs["R_min_max"] = (
                        torch.min(rs_gt).item(),
                        torch.max(rs_gt).item(),
                    )
                    logs["log10_pvalue_avg"] = np.mean(log10_pvalues)
                    logs["log10_pvalue_min_max"] = (
                        np.amin(log10_pvalues),
                        np.amax(log10_pvalues),
                    )
                    logs["decision"] = torch.mean(rs_gt).item() > 0
                    logs["dot_product"] = torch.mean(dot_product_gt).item()
                # print("__log__:%s" % json.dumps(logs))

                logs_all[(batch_iter, iteration)] = logs

        batch_imgs = best_batch if select_best else batch_imgs

        # post process and store
        for ii, x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = wm_utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    if return_logs:
        return pt_imgs_out, logs_all

    return pt_imgs_out  # [CxW1xH1, ..., CxWnxHn]


def watermark_multibit_attack(
    img_loader: DataLoader,
    carrier_attack: torch.Tensor,
    carrier_gt: torch.Tensor,
    msgs: torch.Tensor,
    angle: float,
    model: nn.Module,
    transform: Callable,
    params,
    return_logs: bool = False,
    use_cosine_sim: bool = False,
    select_best: bool = False,
):
    """Multibit watermarking adapted for attack purposes.

    Args:
        img_loader: Dataloader of the images to be watermarked
        carrier_attack: Attack carrier. Hypercone direction 1xD
        carrier_gt: Ground truth carrier. Hypercone direction 1xD
        msgs: Watermark messages as a tensor of size BxK
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        transform: Differentiable augmentation with fixed output size -> 1xCxWxH
        params: Must contain optimizer, scheduler, epochs, lambda_w, lambda_i, verbose
        return_logs: If True, returns logs
        use_cosine_sim: If True, uses cosine similarity instead of the original loss
        select_best: If True, selects the best image in the batch

    Returns:
        imgs: Watermarked images as a list of unnormalized (distributed around [-1, 1]) pytorch tensors
        Optional[Dict]: Logs if return_logs is True
    """

    rho = 1 + np.tan(angle) ** 2
    ssim = wm_utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    if select_best:
        best_cosine_sim = 0
        best_batch = None

    logs_all = {}
    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1] * img.shape[-2] for img in images])
        if max_res > 1e6:
            print(
                "WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU."
            )

        # load images
        batch_imgs_orig = [
            x.to(device, non_blocking=True).unsqueeze(0) for x in images
        ]  # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig]  # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        N = len(img_loader.dataset)
        B = params.batch_size
        batch_msgs = msgs[batch_iter * B : min((batch_iter + 1) * B, N)].to(
            device, non_blocking=True
        )
        optimizer = build_optimizer(
            model_params=batch_imgs, **wm_utils.parse_params(params.optimizer)
        )
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, **wm_utils.parse_params(params.scheduler)
            )

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                # concentrate changes around edges
                x = ssim.apply(x, batch_imgs_orig[ii])
                # remain within PSNR budget
                x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii == 0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0)  # BxCxWxH
            # get features
            ft = model(batch)  # BxCxWxH -> BxD
            norm = torch.norm(ft, dim=-1, keepdim=True)  # BxD -> Bx1

            # compute losses

            if use_cosine_sim:
                ft_norm = ft / norm
                dot_product = ft_norm @ carrier_attack.T  # BxD @ Dx1 -> Bx1
                loss_w = torch.sum(1 - dot_product)

                if select_best:
                    cosines = torch.mean(dot_product)
                    if cosines > best_cosine_sim:
                        best_cosine_sim = cosines
                        best_batch = batch_imgs
            else:
                dot_product = ft @ carrier_attack.T  # BxD @ Dx1 -> Bx1
                loss_w = torch.sum(-(rho * dot_product**2 - norm**2))  # B-B -> B

            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += (
                    torch.norm(batch_imgs[ii] - batch_imgs_orig[ii]) ** 2
                )  # CxWxH -> 1
            loss = params.lambda_w * loss_w + params.lambda_i * loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose > 1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }

                pprint(logs)

                if params.verbose > 2:
                    dot_product_gt = ft @ carrier_gt.T
                    decoded_msgs = torch.sign(dot_product_gt) > 0
                    diff = ~torch.logical_xor(batch_msgs, decoded_msgs)  # BxK -> BxK
                    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # BxK -> B
                    logs["bit_acc_avg"] = torch.mean(bit_accs).item()
                # print("__log__:%s" % json.dumps(logs))
                logs_all[(batch_iter, iteration)] = logs

        batch_imgs = best_batch if select_best else batch_imgs

        # post process and store
        for ii, x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = wm_utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    if return_logs:
        return pt_imgs_out, logs_all

    return pt_imgs_out  # [CxW1xH1, ..., CxWnxHn]


def watermark_0bit_untargeted_attack(
    img_loader: DataLoader,
    carrier_attack: torch.Tensor,
    carrier_gt: torch.Tensor,
    angle: float,
    model: nn.Module,
    transform: Callable,
    params,
    return_logs: bool = False,
    select_best: bool = False,
):
    """
    0-bit watermarking of a batch of images.

    Args:
        img_loader: Dataloader of the images to be watermarked
        carrier_attack (tensor of size 1xD): Attack carrier. Hypercone direction 1xD
        carrier_gt (tensor of size 1xD): Ground truth carrier. Hypercone direction 1xD
        angle: Angle of the hypercone
        model: Neural net model to extract the features
        transform: Differentiable augmentation with fixed output size -> 1xCxWxH
        params: Must contain optimizer, scheduler, epochs, lambda_w, lambda_i, verbose
        return_logs: If True, returns logs
        select_best: If True, selects the best image in the batch

    Returns:
        imgs: Watermarked images as a list of unnormalized (distributed around [-1, 1]) pytorch tensors
    """
    rho = 1 + np.tan(angle) ** 2
    ssim = wm_utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    if select_best:
        best_cosine_sim = 1
        best_batch = None

    logs_all = {}
    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1] * img.shape[-2] for img in images])
        if max_res > 1e6:
            print(
                "WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU."
            )

        # load images
        batch_imgs_orig = [
            x.to(device, non_blocking=True).unsqueeze(0) for x in images
        ]  # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig]  # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        optimizer = build_optimizer(
            model_params=batch_imgs, **wm_utils.parse_params(params.optimizer)
        )
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, **wm_utils.parse_params(params.scheduler)
            )

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                # concentrate changes around edges
                x = ssim.apply(x, batch_imgs_orig[ii])
                # remain within PSNR budget
                x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii == 0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0)  # BxCxWxH
            # get features
            ft = model(batch)  # BxCxWxH -> BxD
            norm = torch.norm(ft, dim=-1, keepdim=True)  # BxD -> Bx1
            ft_norm = ft / norm

            # compute losses
            dot_product = ft_norm @ carrier_attack.T  # BxD @ Dx1 -> Bx1
            loss_w = torch.sum(dot_product**2)

            # select the best images
            if select_best:
                cosines = torch.mean(torch.abs(dot_product))
                if cosines < best_cosine_sim:
                    best_cosine_sim = cosines
                    best_batch = batch_imgs

            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += (
                    torch.norm(batch_imgs[ii] - batch_imgs_orig[ii]) ** 2
                )  # CxWxH -> 1
            loss = params.lambda_w * loss_w + params.lambda_i * loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose > 1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }

                pprint(logs)

                if params.verbose > 2:
                    dot_product_gt = ft @ carrier_gt.T
                    rs_gt = rho * dot_product_gt**2 - norm**2  # Bx1-Bx1 -> Bx1
                    cosines_gt = torch.abs(dot_product_gt / norm)  # Bx1/Bx1 -> Bx1
                    log10_pvalues = [
                        np.log10(
                            wm_utils.cosine_pvalue(cosines_gt[ii].item(), ft.shape[-1])
                        )
                        for ii in range(len(batch_imgs))
                    ]
                    logs["R_avg"] = torch.mean(rs_gt).item()
                    logs["R_min_max"] = (
                        torch.min(rs_gt).item(),
                        torch.max(rs_gt).item(),
                    )
                    logs["log10_pvalue_avg"] = np.mean(log10_pvalues)
                    logs["log10_pvalue_min_max"] = (
                        np.amin(log10_pvalues),
                        np.amax(log10_pvalues),
                    )
                    logs["decision"] = torch.mean(rs_gt).item() > 0
                    logs["dot_product_gt"] = torch.mean(dot_product_gt).item()
                    logs["dot_product"] = torch.mean(dot_product).item()

                logs_all[(batch_iter, iteration)] = logs

        # post process and store
        batch_imgs = best_batch if select_best else batch_imgs

        for ii, x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = wm_utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    if return_logs:
        return pt_imgs_out, logs_all

    return pt_imgs_out  # [CxW1xH1, ..., CxWnxHn]


def watermark_multibit_untargeted_attack(
    img_loader: DataLoader,
    carrier_attack: torch.Tensor,
    carrier_gt: torch.Tensor,
    msgs: torch.Tensor,
    model: nn.Module,
    transform: Callable,
    params,
    return_logs: bool = False,
    select_best: bool = False,
):
    # rho = 1 + np.tan(angle)**2
    ssim = wm_utils_img.SSIMAttenuation(device=device)
    pt_imgs_out = []

    if select_best:
        best_cosine_sim = 1
        best_batch = None

    logs_all = {}
    for batch_iter, (images, _) in enumerate(tqdm(img_loader)):

        # Warning for resolution
        max_res = max([img.shape[-1] * img.shape[-2] for img in images])
        if max_res > 1e6:
            print(
                "WARNING: One or more of the images is high resolution, it can be too large to be processed by the GPU."
            )

        # load images
        batch_imgs_orig = [
            x.to(device, non_blocking=True).unsqueeze(0) for x in images
        ]  # BxCxWxH
        batch_imgs = [x.clone() for x in batch_imgs_orig]  # BxCxWxH
        for i in range(len(batch_imgs)):
            batch_imgs[i].requires_grad = True
        N = len(img_loader.dataset)
        B = params.batch_size
        batch_msgs = msgs[batch_iter * B : min((batch_iter + 1) * B, N)].to(
            device, non_blocking=True
        )
        optimizer = build_optimizer(
            model_params=batch_imgs, **wm_utils.parse_params(params.optimizer)
        )
        if params.scheduler is not None:
            scheduler = build_lr_scheduler(
                optimizer=optimizer, **wm_utils.parse_params(params.scheduler)
            )

        # optimization
        for iteration in range(params.epochs):
            # Constraints and data augmentations
            batch = []
            for ii, x in enumerate(batch_imgs):
                # concentrate changes around edges
                x = ssim.apply(x, batch_imgs_orig[ii])
                # remain within PSNR budget
                x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
                if ii == 0:
                    aug_params = transform.sample_params(x)
                aug_img = transform(x, aug_params)
                batch.append(aug_img)
            batch = torch.cat(batch, dim=0)  # BxCxWxH
            # get features
            ft = model(batch)  # BxCxWxH -> BxD
            norm = torch.norm(ft, dim=-1, keepdim=True)  # BxD -> Bx1
            ft_norm = ft / norm

            # compute losses
            dot_product = ft_norm @ carrier_attack.T  # BxD @ Dx1 -> Bx1
            loss_w = torch.sum(dot_product**2)

            # select the best images
            if select_best:
                cosines = torch.mean(torch.abs(dot_product))
                if cosines < best_cosine_sim:
                    best_cosine_sim = cosines
                    best_batch = batch_imgs

            loss_i = 0
            for ii in range(len(batch_imgs)):
                loss_i += (
                    torch.norm(batch_imgs[ii] - batch_imgs_orig[ii]) ** 2
                )  # CxWxH -> 1
            loss = params.lambda_w * loss_w + params.lambda_i * loss_i
            # update images (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.scheduler is not None:
                scheduler.step()
            # logs
            if params.verbose > 1:
                logs = {
                    "keyword": "img_optim",
                    "batch": batch_iter,
                    "iteration": iteration,
                    "loss": loss.item(),
                    "loss_w": loss_w.item(),
                    "loss_i": loss_i.item(),
                }

                pprint(logs)

                if params.verbose > 2:
                    dot_product_gt = ft @ carrier_gt.T
                    decoded_msgs = torch.sign(dot_product_gt) > 0
                    diff = ~torch.logical_xor(batch_msgs, decoded_msgs)  # BxK -> BxK
                    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # BxK -> B
                    logs["bit_acc_avg"] = torch.mean(bit_accs).item()
                # print("__log__:%s" % json.dumps(logs))
                logs_all[(batch_iter, iteration)] = logs

        # post process and store
        batch_imgs = best_batch if select_best else batch_imgs

        # post process and store
        for ii, x in enumerate(batch_imgs):
            x = ssim.apply(x, batch_imgs_orig[ii])
            x = wm_utils_img.psnr_clip(x, batch_imgs_orig[ii], params.target_psnr)
            x = wm_utils_img.round_pixel(x)
            # x = utils_img.project_linf(x, batch_imgs_orig[ii], params.linf_radius)
            pt_imgs_out.append(x.squeeze(0).detach().cpu())

    if return_logs:
        return pt_imgs_out, logs_all

    return pt_imgs_out  # [CxW1xH1, ..., CxWnxHn]
