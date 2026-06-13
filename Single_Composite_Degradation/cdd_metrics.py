########################################
## Evaluation code follows OneRestore.
########################################
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from basicsr.utils.options import parse


def calculate_psnr_with_progress(clear_folder, degradation_types, methods, degradation_path):
    img_list = [img for img in os.listdir(clear_folder) if img.endswith('.png')]
    psnr_matrix = np.zeros((len(methods), len(degradation_types)))

    total_tasks = len(methods) * len(degradation_types) * 200
    print(total_tasks, len(methods))

    with tqdm(total=total_tasks, desc="Processing Images", unit="task") as pbar:
        for k, method in enumerate(methods):
            print(f"Processing method: {method}")
            for j, degradation_type in enumerate(degradation_types):
                psnr_values = []
                for img_name in img_list:
                    clear_img_path = os.path.join(clear_folder, img_name)
                    degraded_img_path = f'./{method}/{degradation_type}/{img_name}'
                    degraded_img_path = os.path.join(degradation_path, degradation_type + '_' + img_name)

                    clear_img = cv2.imread(clear_img_path) / 255.0
                    degraded_img = cv2.imread(degraded_img_path) / 255.0

                    if clear_img is not None and degraded_img is not None:
                        psnr_value = compare_psnr(clear_img, degraded_img, data_range=1.0)
                        psnr_values.append(psnr_value)

                    pbar.update(1)

                if psnr_values:
                    psnr_matrix[k, j] = np.mean(psnr_values)

    return psnr_matrix


def save_matrix_to_excel(psnr_matrix, methods, degradation_types, data):
    psnr_df = pd.DataFrame(psnr_matrix, index=methods, columns=degradation_types)
    output_file = data + '.xlsx'

    with pd.ExcelWriter(output_file) as writer:
        psnr_df.to_excel(writer, sheet_name='PSNR')

    print(f'PSNR matrix saved to {output_file}')


def save_matrix_to_txt(psnr_matrix, methods, degradation_types, data):
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_dir = os.path.join('experiments', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    filename = os.path.join(metrics_dir, f"metrics_{data}_{timestamp}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"DVANet PSNR Report - {data}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PSNR Matrix:\n")
        f.write("-" * 70 + "\n")
        header = f"{'Method':<15}" + "".join([f"{dt:<12}" for dt in degradation_types]) + "\n"
        f.write(header)
        f.write("-" * 70 + "\n")

        for k, method in enumerate(methods):
            row = f"{method:<15}"
            for j in range(len(degradation_types)):
                row += f"{psnr_matrix[k, j]:<12.4f}"
            f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Average PSNR: {psnr_matrix.mean():.4f}\n")

    print(f"PSNR metrics saved to: {filename}")
    print("\n" + "=" * 70)
    print(f"PSNR Report - {data}")
    print("=" * 70)
    print(f"Average PSNR: {psnr_matrix.mean():.4f}")


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data', default='CDD', choices=['CDD', 'CDD-Base'], type=str, help='dataset')
args = parser.parse_args()

opt = parse('./options/CDD.yml', is_train=False)
clear_folder = opt["datasets"]["val"]['dataroot_gt']
degradation_types = [
    'low', 'haze', 'rain', 'snow',
    'low_haze', 'low_rain', 'low_snow',
    'haze_rain', 'haze_snow',
    'low_haze_rain', 'low_haze_snow'
]
degradation_path = os.path.join('./experiments/results/', args.data)
methods = ['DVANet']

psnr_matrix = calculate_psnr_with_progress(clear_folder, degradation_types, methods, degradation_path)
print(psnr_matrix)

save_matrix_to_excel(psnr_matrix, methods, degradation_types, args.data)
save_matrix_to_txt(psnr_matrix, methods, degradation_types, args.data)