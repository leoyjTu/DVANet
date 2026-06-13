import argparse
import os
import time
from os import path as osp
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
from skimage import img_as_ubyte

import utils
from basicsr.models import create_model
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.options import parse
from basicsr.utils.matlab_functions import bgr2ycbcr


def load_model(opt_path, weights_path):
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(weights_path, map_location='cpu')

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except Exception:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    model_restoration.cuda()
    model_restoration.eval()
    return model_restoration, opt


def get_image_paths(folder):
    return natsorted(
        glob(os.path.join(folder, '*.png')) +
        glob(os.path.join(folder, '*.jpg')) +
        glob(os.path.join(folder, '*.jpeg')) +
        glob(os.path.join(folder, '*.bmp')) +
        glob(os.path.join(folder, '*.tif')) +
        glob(os.path.join(folder, '*.tiff'))
    )


def read_img(path):
    img = np.float32(utils.load_img(path)) / 255.
    return img


def save_img(path, img):
    utils.save_img(path, img_as_ubyte(img))


def compute_metrics(img_gt, img_restored, crop_border=0, test_y_channel=False):
    if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
        img_gt_eval = bgr2ycbcr(img_gt, y_only=True)
        img_restored_eval = bgr2ycbcr(img_restored, y_only=True)
    else:
        img_gt_eval = img_gt
        img_restored_eval = img_restored

    psnr = calculate_psnr(
        img_gt_eval * 255,
        img_restored_eval * 255,
        crop_border=crop_border,
        input_order='HWC'
    )
    ssim = calculate_ssim(
        img_gt_eval * 255,
        img_restored_eval * 255,
        crop_border=crop_border,
        input_order='HWC'
    )
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description='Unified test script: inference + metrics')

    parser.add_argument('--data', type=str, required=True, help='Dataset name, used to build default yml / weight path / save dir name')
    parser.add_argument('--opt', type=str, default='./options/',  help='Path to option YAML file or directory containing YAML files')
    parser.add_argument('--weights', type=str, default='./pretrain_model/', help='Path to weight file or directory containing weight files')
    parser.add_argument('--output_dir', type=str, default='./experiments/results/', help='Directory to save restored images')
    parser.add_argument('--save_img', action='store_true', help='Save restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for PSNR/SSIM')
    parser.add_argument('--test_y_channel', action='store_true', help='Evaluate on Y channel')
    parser.add_argument('--factor', type=int, default=32, help='Padding factor; keep consistent with model validation') # 旧版本：4
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix for saved images')
    args = parser.parse_args()

    # resolve paths
    opt_path = args.opt
    if os.path.isdir(opt_path):
        opt_path = osp.join(opt_path, args.data + '.yml')

    weights_path = args.weights
    if os.path.isdir(weights_path):
        weights_path = osp.join(weights_path, args.data + '.pth')

    if not osp.isfile(opt_path):
        raise FileNotFoundError(f'Cannot find opt file: {opt_path}')
    if not osp.isfile(weights_path):
        raise FileNotFoundError(f'Cannot find weights file: {weights_path}')

    print(f'===> Loading opt from: {opt_path}')
    print(f'===> Loading weights from: {weights_path}')

    model_restoration, opt = load_model(opt_path, weights_path)

    if 'val' not in opt['datasets']:
        raise KeyError(
            "This unified test.py expects a single validation dataset at opt['datasets']['val']. "
            "Your config does not contain 'datasets.val'."
        )

    input_dir = opt['datasets']['val']['dataroot_lq']
    gt_dir = opt['datasets']['val'].get('dataroot_gt', None)

    if input_dir is None:
        raise ValueError("No 'dataroot_lq' found in datasets.val.")

    input_paths = get_image_paths(input_dir)
    if len(input_paths) == 0:
        raise ValueError(f'No test images found in {input_dir}')

    output_dir = osp.join(args.output_dir, args.data)
    os.makedirs(output_dir, exist_ok=True)

    has_gt = gt_dir is not None and osp.isdir(gt_dir)

    if args.test_y_channel:
        print('===> Evaluating on Y channel')
    else:
        print('===> Evaluating on RGB channels')

    print(f'===> Number of test images: {len(input_paths)}')
    print(f'===> Saving results to: {output_dir}')

    psnr_all = []
    ssim_all = []
    per_image_logs = []

    with torch.inference_mode():
        for inp_path in tqdm(input_paths, total=len(input_paths)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            basename = osp.splitext(osp.basename(inp_path))[0]

            img = read_img(inp_path)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img_tensor.unsqueeze(0).cuda()

            b, c, h, w = input_.shape
            H = ((h + args.factor - 1) // args.factor) * args.factor
            W = ((w + args.factor - 1) // args.factor) * args.factor
            padh = H - h
            padw = W - w

            input_padded = F.pad(input_, (0, padw, 0, padh), mode='reflect')

            restored = model_restoration(input_padded)
            if isinstance(restored, list):
                restored = restored[-1]

            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach() \
                .permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.save_img:
                save_name = basename + (args.suffix if args.suffix else '') + '.png'
                save_path = osp.join(output_dir, save_name)
                save_img(save_path, restored)

            if has_gt:
                # default: same basename, try common extensions
                gt_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    candidate = osp.join(gt_dir, basename + ext)
                    if osp.isfile(candidate):
                        gt_path = candidate
                        break

                if gt_path is None:
                    raise FileNotFoundError(f'Cannot find GT image for {basename} in {gt_dir}')

                img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
                if img_gt.ndim == 2:
                    img_gt = np.expand_dims(img_gt, axis=2)
                if restored.ndim == 2:
                    restored = np.expand_dims(restored, axis=2)

                psnr, ssim = compute_metrics(
                    img_gt,
                    restored,
                    crop_border=args.crop_border,
                    test_y_channel=args.test_y_channel
                )
                psnr_all.append(psnr)
                ssim_all.append(ssim)
                per_image_logs.append((basename, psnr, ssim))

                print(f'{basename:30s}  PSNR: {psnr:.6f} dB  SSIM: {ssim:.6f}')

    print('\n===> Test complete!')

    if has_gt and len(psnr_all) > 0:
        avg_psnr = sum(psnr_all) / len(psnr_all)
        avg_ssim = sum(ssim_all) / len(ssim_all)

        print('=' * 60)
        print(f'Dataset: {args.data}')
        print(f'Average PSNR: {avg_psnr:.6f} dB')
        print(f'Average SSIM: {avg_ssim:.6f}')
        print('=' * 60)

        metrics_dir = osp.join('experiments', 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        metrics_path = osp.join(metrics_dir, f'metrics_{args.data}_{timestamp}.txt')

        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write(f'Metrics Report - {args.data}\n')
            f.write(f'Time: {timestamp}\n')
            f.write(f'Weights: {weights_path}\n')
            f.write(f'Opt: {opt_path}\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'Average PSNR: {avg_psnr:.6f} dB\n')
            f.write(f'Average SSIM: {avg_ssim:.6f}\n\n')
            f.write('Per-image metrics:\n')
            for basename, psnr, ssim in per_image_logs:
                f.write(f'  {basename}: PSNR={psnr:.6f}, SSIM={ssim:.6f}\n')
            f.write('\n' + '=' * 60 + '\n')

        print(f'===> Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()