import argparse
from tqdm import tqdm
import os
import time
import csv
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime
from lightning.pytorch.strategies import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_utils import DVANetTrainDataset
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import DVANet


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * self.criterion(pred_fft, target_fft)


class DVANetModel(pl.LightningModule):
    def __init__(self, opt, eval_interval):
        super().__init__()
        self.opt = opt
        self.eval_interval = eval_interval
        self.net = DVANet(
            degradation_dim=opt.degradation_dim,
            drb_mid_channels=opt.drb_mid_channels,
            local_token_grid=opt.local_token_grid,
            num_degradation_tokens=opt.num_degradation_tokens,
            dino_extract_ids=tuple(opt.dino_extract_ids),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.loss_fn = nn.L1Loss()
        self.loss_fft = FFTLoss()
        self.eval_datasets()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        degrad_patch = degrad_patch.to(self.device, non_blocking=True)
        clean_patch = clean_patch.to(self.device, non_blocking=True)

        restored = self.net(degrad_patch)
        loss_rec = self.loss_fn(restored, clean_patch)
        loss_fft = self.loss_fft(restored, clean_patch)
        loss = loss_rec + loss_fft

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss_rec", loss_rec, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_fft", loss_fft, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=self.opt.epochs
        )
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_interval == 0:
            if self.trainer.is_global_zero:
                print(f"\nStart evaluation at epoch {self.current_epoch + 1}")
                self.test_all()
                print(f"Finish evaluation at epoch {self.current_epoch + 1}")
            self.trainer.strategy.barrier()

    def eval_datasets(self):
        self.denoise_tests = []
        self.derain_set = None
        self.dehaze_set = None
        self.deblur_set = None
        self.enhance_set = None

        if any(x in self.opt.de_type for x in ["denoise_15", "denoise_25", "denoise_50"]):
            denoise_splits = ["bsd68/"]
            denoise_base_path = self.opt.denoise_path
            for split in denoise_splits:
                self.opt.denoise_path = os.path.join(denoise_base_path, split)
                self.denoise_tests.append(DenoiseTestDataset(self.opt))

        if "derain" in self.opt.de_type:
            derain_splits = ["Rain100L/"]
            derain_base_path = self.opt.derain_path
            for split in derain_splits:
                self.opt.derain_path = os.path.join(derain_base_path, split)
                self.derain_set = DerainDehazeDataset(self.opt, task="derain", addnoise=False, sigma=15)

        if "dehaze" in self.opt.de_type:
            self.dehaze_set = DerainDehazeDataset(self.opt, task="dehaze", addnoise=False, sigma=15)

        if "deblur" in self.opt.de_type:
            deblur_splits = ["gopro/"]
            deblur_base_path = self.opt.gopro_path
            for split in deblur_splits:
                self.opt.gopro_path = os.path.join(deblur_base_path, split)
                self.deblur_set = DerainDehazeDataset(self.opt, task="deblur", addnoise=False, sigma=15)

        if "enhance" in self.opt.de_type:
            enhance_splits = ["lol/"]
            enhance_base_path = self.opt.enhance_path
            for split in enhance_splits:
                self.opt.enhance_path = os.path.join(enhance_base_path, split)
                self.enhance_set = DerainDehazeDataset(self.opt, task="enhance", addnoise=False, sigma=15)

    def test_all(self):
        results = {}

        for testset in self.denoise_tests:
            if "denoise_15" in self.opt.de_type:
                results["denoise_15"] = self.test_Denoise(testset, sigma=15)
            if "denoise_25" in self.opt.de_type:
                results["denoise_25"] = self.test_Denoise(testset, sigma=25)
            if "denoise_50" in self.opt.de_type:
                results["denoise_50"] = self.test_Denoise(testset, sigma=50)

        if "derain" in self.opt.de_type and self.derain_set is not None:
            results["derain"] = self.test_Derain_Dehaze(self.derain_set, task="derain")
        if "dehaze" in self.opt.de_type and self.dehaze_set is not None:
            results["dehaze"] = self.test_Derain_Dehaze(self.dehaze_set, task="dehaze")
        if "deblur" in self.opt.de_type and self.deblur_set is not None:
            results["deblur"] = self.test_Derain_Dehaze(self.deblur_set, task="deblur")
        if "enhance" in self.opt.de_type and self.enhance_set is not None:
            results["enhance"] = self.test_Derain_Dehaze(self.enhance_set, task="enhance")

        self.save_metrics_to_csv(results)

    def save_metrics_to_csv(self, results):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        exp_dir = os.path.join(self.opt.experiments_dir, self.opt.exp_name)
        metrics_dir = os.path.join(exp_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        filename = os.path.join(metrics_dir, "metrics_all.csv")
        file_exists = os.path.exists(filename)

        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "time", "task", "psnr", "ssim"])
            for task_name, metrics in results.items():
                if metrics is not None:
                    writer.writerow([
                        self.current_epoch + 1,
                        timestamp,
                        task_name,
                        f"{metrics['psnr']:.2f}",
                        f"{metrics['ssim']:.4f}",
                    ])

    def test_Denoise(self, dataset, sigma=15):
        dataset.set_sigma(sigma)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch = degrad_patch.to(self.device, non_blocking=True)
                clean_patch = clean_patch.to(self.device, non_blocking=True)

                restored = self.net(degrad_patch)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

        print(f"Denoise sigma={sigma}: psnr: {psnr.avg:.2f}, ssim: {ssim.avg:.4f}")
        self.log(f"psnr_{sigma}", psnr.avg, sync_dist=False, rank_zero_only=True)
        self.log(f"ssim_{sigma}", ssim.avg, sync_dist=False, rank_zero_only=True)
        return {"psnr": psnr.avg, "ssim": ssim.avg}

    def test_Derain_Dehaze(self, dataset, task="derain"):
        dataset.set_dataset(task)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
        psnr = AverageMeter()
        ssim = AverageMeter()

        with torch.no_grad():
            for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
                degrad_patch = degrad_patch.to(self.device, non_blocking=True)
                clean_patch = clean_patch.to(self.device, non_blocking=True)

                b, c, h, w = degrad_patch.shape
                h_n = (8 - h % 8) % 8
                w_n = (8 - w % 8) % 8
                degrad_patch = F.pad(degrad_patch, (0, w_n, 0, h_n), mode="reflect")

                restored = self.net(degrad_patch)[:, :, :h, :w]
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)

        self.log(f"psnr_{task}", psnr.avg, sync_dist=False, rank_zero_only=True)
        self.log(f"ssim_{task}", ssim.avg, sync_dist=False, rank_zero_only=True)
        print(f"PSNR_{task}: {psnr.avg:.2f}, SSIM_{task}: {ssim.avg:.4f}")
        return {"psnr": psnr.avg, "ssim": ssim.avg}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--denoise_path", type=str, default="data/test/denoise/", help="save path of test noisy images")
    parser.add_argument("--derain_path", type=str, default="data/test/derain/", help="save path of test raining images")
    parser.add_argument("--dehaze_path", type=str, default="data/test/dehaze/", help="save path of test hazy images")
    parser.add_argument("--gopro_path", type=str, default="data/test/deblur/", help="save path of test blurry images")
    parser.add_argument("--enhance_path", type=str, default="data/test/enhance/", help="save path of test low light images")

    parser.add_argument("--epochs", type=int, default=150, help="maximum number of epochs to train the total model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate of encoder.")
    parser.add_argument("--patch_size", type=int, default=128, help="patchsize of input.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for training")

    parser.add_argument(
        "--de_type",
        nargs="+",
        default=["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze", "deblur", "enhance"],
        help="degradation types used in training and evaluation"
    )

    parser.add_argument("--degradation_dim", type=int, default=512, help="dimension of global degradation vector")
    parser.add_argument("--drb_mid_channels", type=int, default=64, help="middle channels of degradation representation block")
    parser.add_argument("--local_token_grid", type=int, default=2, help="local token grid size; 2 means 2x2=4 local tokens")
    parser.add_argument("--num_degradation_tokens", type=int, default=16, help="number of learnable global degradation prototype tokens")
    parser.add_argument("--dino_extract_ids", type=int, nargs="+", default=[5, 11, 17, 23], help="layer ids extracted from DINOv3 backbone")

    parser.add_argument("--data_file_dir", type=str, default="data_dir/", help="where clean images of denoising saves.")
    parser.add_argument("--denoise_dir", type=str, default="data/train/denoise/", help="where clean images of denoising saves.")
    parser.add_argument("--gopro_dir", type=str, default="data/train/deblur/", help="where training images of deblurring saves.")
    parser.add_argument("--enhance_dir", type=str, default="data/train/enhance/", help="where training images of enhancement saves.")
    parser.add_argument("--derain_dir", type=str, default="data/train/derain/", help="where training images of deraining saves.")
    parser.add_argument("--dehaze_dir", type=str, default="data/train/dehaze/", help="where training images of dehazing saves.")

    parser.add_argument("--experiments_dir", type=str, default="experiments/", help="base directory for all experiments")
    parser.add_argument("--exp_name", type=str, default="DVANet", help="name of this experiment")
    parser.add_argument("--resume_from_ckpt", type=str, default=None, help="Resume training from specified checkpoint")

    opt = parser.parse_args()

    exp_dir = os.path.join(opt.experiments_dir, opt.exp_name)
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    log_dir = os.path.join(exp_dir, "logs")
    model_backup_dir = os.path.join(exp_dir, "model_backup")
    metrics_dir = os.path.join(exp_dir, "metrics")

    for dir_path in [ckpt_dir, log_dir, model_backup_dir, metrics_dir]:
        os.makedirs(dir_path, exist_ok=True)

    for src_file in [os.path.join("net", "model.py"), os.path.join("net", "DINOv3_utils.py")]:
        if os.path.isfile(src_file):
            shutil.copy2(src_file, model_backup_dir)

    logger = TensorBoardLogger(save_dir=log_dir, name=opt.exp_name)

    trainset = DVANetTrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1
    )

    model = DVANetModel(opt, eval_interval=10)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    if opt.resume_from_ckpt:
        print(f"Resume training from checkpoint: {opt.resume_from_ckpt}")
        trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=opt.resume_from_ckpt)
    else:
        trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == "__main__":
    main()