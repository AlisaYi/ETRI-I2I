import argparse
import builtins
import os
import sys
from datetime import timedelta, datetime
from tqdm import tqdm
import glob

import torch
import torchvision
import torchvision.datasets as datasets
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.swin_transformer_v2 import SwinTransformerV2
from dataset.dataset import Im2ImDataset

import clip
from lpips.lpips import LPIPS
from id_loss.id_loss import IDLoss

try:
    import wandb
except ImportError:
    wandb = None

import time

from dataset.dataset import MultiResolutionDataset
from models.discriminator import Discriminator
from models.generator_pmh import Generator
from utils import fid_score
from utils.CRDiffAug import CR_DiffAug
from utils.distributed import get_rank, reduce_loss_dict, synchronize


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def evaluation(generator, test_loader, save_path):
    for i, inputs in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            noise = torch.randn((args.batch, 512)).cuda()
            image = inputs['A'].cuda()
            image_name = inputs['A_name']

            out_sample, _, _ = generator(noise, image)
            out_sample = denormalize(out_sample)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            torchvision.utils.save_image(
                out_sample[0],
                save_path + '/' + image_name[0],
                nrow=1,
                padding=0,
                normalize=True,
                range=(0, 1),
            )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='/Data1/Project/Img2Img/Datasets/GTA5/train_test_split/test', help="Path of training data")
    parser.add_argument("--iter", type=int, default=1500000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--r1", type=float, default=5)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default='./pretrained_styleswin/Cityscapes_I2I_Swin_20240821_020000.pt')
    parser.add_argument("--G_channel_multiplier", type=int, default=2)
    
    parser.add_argument('--workers', default=8, type=int, help='Number of workers')
    
    parser.add_argument('--gan_weight', default=0.01, type=float, help='Gan loss weight')
    parser.add_argument('--recon_weight', default=1.0, type=float, help='Recon loss weight')
    parser.add_argument('--lpips_weight', default=0.8, type=float, help='LPIPS loss weight')
    parser.add_argument('--id_weight', default=0.1, type=float, help='ID loss weight')
   
    parser.add_argument('--val_num_batches', default=1250, type=int, help='Num of batches will be generated during evalution')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size during evalution')
    parser.add_argument('--D_sn', default=True, help='If we use spectral norm in D')
    parser.add_argument('--ttur', default=True, help='If we use TTUR during training')
    parser.add_argument('--eval', action="store_true", help='Only do evaluation')
    parser.add_argument("--eval_iters", type=int, default=0, help="Iters of evaluation ckpt")
    parser.add_argument('--eval_gt_path', default='/Data1/Project/Img2Img/Datasets/GTA5/train_test_split/test', type=str, help='Path to ground truth images to evaluate FID score')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio in swin')
    parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
    parser.add_argument("--bcr", default=True, help='If we add bcr during training')
    parser.add_argument("--bcr_fake_lambda", default=10, type=float, help='Bcr weight for fake data')
    parser.add_argument("--bcr_real_lambda", default=10, type=float, help='Bcr weight for real data')
    parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from checkpoint")
    parser.add_argument("--dataset", default='ffhq', help='Whether to use lmdb datasets')
    parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 4096
    args.n_mlp = 8 
    args.g_reg_every = 10000000    # We do not apply regularization on G

    

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="gloo", init_method="env://", timeout=timedelta(0, 18000))
        synchronize()

    
    if args.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    
    generator = Generator(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)

    generator.eval()
    
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"])


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.Resize(size=args.size),
        transforms.CenterCrop(size=args.size),
        transforms.ToTensor(),
        normalize
    ])


    imgs_A_test_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/GTA5/train_test_split/test/images/*'))
    imgs_B_test_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/GTA5/train_test_split/test/images/*'))

    dataset_test = Im2ImDataset(imgs_A=imgs_A_test_pth, imgs_B=imgs_B_test_pth, transform=test_transform)
    
    
    test_loader = data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.workers,
        sampler=None,
        drop_last=False,
    )


    # train(args, train_loader, test_loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer)
    evaluation(generator, test_loader, './results/Cityscapes_I2I_Swin_20240821/samples/020000')