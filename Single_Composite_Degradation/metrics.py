import argparse
import cv2
import numpy as np
import time
import os

from os import path as osp
import torch
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.options import parse

def main(args, gt_path):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []

    restored_path = osp.join(args.restored, args.data)
    img_list_gt = sorted(list(scandir(gt_path, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(restored_path, recursive=True, full_path=True)))

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        img_path_restored = img_list_restored[i]
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')

        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)

    avg_psnr = sum(psnr_all) / len(psnr_all)
    avg_ssim = sum(ssim_all) / len(ssim_all)
    print(f'Average: PSNR: {avg_psnr:.6f} dB, SSIM: {avg_ssim:.6f}')

    # Save metrics to txt file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_dir = osp.join('experiments', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    filename = osp.join(metrics_dir, f"metrics_{args.data}_{timestamp}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Metrics Report - {args.data}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Average PSNR: {avg_psnr:.6f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        f.write("\nPer-image metrics:\n")
        for i, (psnr, ssim) in enumerate(zip(psnr_all, ssim_all)):
            basename = osp.splitext(osp.basename(img_list_gt[i]))[0]
            f.write(f"  {basename}: PSNR={psnr:.6f}, SSIM={ssim:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")

    # Output metrics to terminal
    print("\n" + "=" * 60)
    print(f"Metrics Report - {args.data}")
    print("=" * 60)
    print(f"Average PSNR: {avg_psnr:.6f} dB")
    print(f"Average SSIM: {avg_ssim:.6f}")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored', type=str, default='./experiments/results/', help='Path to restored images')
    parser.add_argument('--data', 
                        default='LOLBlur', 
                        choices=['LOLBlur', 'CSD', 'Snow100K', 'SRRS', 'LOL', 'Dense-Haze', 'NH-Haze', 'CEC', 'Haze1k', 'RICE', 'NightRain'],
                        type=str, 
                        help='dataset')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    
    opt = parse('./options/' + args.data + '.yml', is_train=False)
    gt_path = opt["datasets"]["val"]["dataroot_gt"]
    
    main(args, gt_path)