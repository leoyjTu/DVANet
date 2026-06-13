import numpy as np
import os
import argparse
from tqdm import tqdm
from os import path as osp
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import parse


parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--output_dir', default='./experiments/results/', type=str, help='Directory for output')
parser.add_argument('--data', default='CDD',
                    choices=['CDD', 'LOLBlur', 'CSD', 'Snow100K', 'SRRS', 'LOL', 'Dense-Haze', 'NH-Haze', 'CEC', 'Haze1k', 'RICE', 'NightRain'],
                    type=str, help='dataset')
parser.add_argument('--opt', type=str, default='./options/', help='Path to option YAML file.')
parser.add_argument('--weights', default='./pretrain_model/', type=str, help='Path to weights')
parser.add_argument('--save_img', default=True, help='Save restored images')

args = parser.parse_args()

####### Load yaml #######
weights = osp.join(args.weights, args.data + '.pth')
opt_path = osp.join(args.opt, args.data + '.yml')
opt = parse(opt_path, is_train=False)
opt['dist'] = False

model_restoration = create_model(opt).net_g
checkpoint = torch.load(weights, map_location='cpu')

try:
    model_restoration.load_state_dict(checkpoint['params'])
except Exception:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===> Testing using weights:", weights)
model_restoration.cuda()
model_restoration.eval()

factor = 8

output_dir = os.path.join(args.output_dir, args.data)
if output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

input_dir = opt["datasets"]["val"]["dataroot_lq"]

input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) +
    glob(os.path.join(input_dir, '*.jpg')) +
    glob(os.path.join(input_dir, '*.tif'))
)

print(len(input_paths))
with torch.inference_mode():
    for inp_path in tqdm(input_paths, total=len(input_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img_tensor.unsqueeze(0).cuda()

        # Padding in case images are not multiples of factor
        b, c, h, w = input_.shape
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_padded = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_padded)
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu().detach() \
            .permute(0, 2, 3, 1).squeeze(0).numpy()

        if args.save_img:
            save_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png'
            )
            utils.save_img(save_path, img_as_ubyte(restored))

print(f"\n===> Evaluation complete! Results saved to: {output_dir}")