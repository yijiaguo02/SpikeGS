import os
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm, trange
from PIL import Image

from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim
# from lpips import LPIPS


# lpips_fn = LPIPS(net="vgg").cuda()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory that stores the relighting results.")
    parser.add_argument("--gt_dir", type=str, help="The path to the output directory that stores the relighting ground truth.")
    parser.add_argument("--textname", type=str, help="The path to the output directory that stores the relighting ground truth.")
    args = parser.parse_args()

    light_name_list = ["renders"]

    for light_name in light_name_list:
        print(f"evaluation {light_name}")
        num_test = 1
        psnr_avg = 0.0
        ssim_avg = 0.0
        lpips_avg = 0.0
        for idx in trange(num_test):
            with torch.no_grad():
                idx=449
                prediction = np.array(Image.open("/gpfs/share/home/2301112015/gaussian-splatting/output/NeRF_Syn_blur_ini/factory2_low/3dgs_tfp+tfi/train/ours_30000/renders/00449.png"))[..., :3]  # [H, W, 3]
                prediction = torch.from_numpy(prediction).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                gt_img = np.array(Image.open("/gpfs/share/home/2301112015/Spike_3dgs/llff_data/factory2/train/r_449.png"))[..., :3]  # [H, W, 3]
                gt_img = torch.from_numpy(gt_img).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                psnr_avg += get_psnr(gt_img, prediction).mean().double()
                ssim_avg += get_ssim(gt_img, prediction).mean().double()
                # lpips_avg += lpips_fn(gt_img, prediction).mean().double()

        print(f"{light_name} psnr_avg: {psnr_avg / num_test}")
        print(f"{light_name} ssim_avg: {ssim_avg / num_test}")
        # print(f"{light_name} lpips_avg: {lpips_avg / num_test}")
        file = open(args.textname, "a")
        file.write("\n")
        file.write(f"psnr_avg: {psnr_avg / num_test}")
        file.write("\n")
        file.write(f"{light_name} ssim_avg: {ssim_avg / num_test}")
        file.write("\n")
        # file.write(f"{light_name} lpips_avg: {lpips_avg / num_test}")
        # file.write("\n")
        # file.close()