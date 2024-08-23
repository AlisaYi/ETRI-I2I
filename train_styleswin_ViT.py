# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import builtins
import os
import sys
from datetime import timedelta, datetime
from tqdm import tqdm

import torch
import torchvision
import torchvision.datasets as datasets
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision import transforms
import torch.nn.functional as F


try:
    import wandb
except ImportError:
    wandb = None

import time

from dataset.dataset import MultiResolutionDataset
from models.discriminator import Discriminator
# from models.generator_ViT import Generator
from models.generator_ViT import Generator_MultiResolution
from utils import fid_score
from utils.CRDiffAug import CR_DiffAug
from utils.distributed import get_rank, reduce_loss_dict, synchronize

# lpips, id_loss import
from lpips.lpips import LPIPS
from id_loss.id_loss import IDLoss

# PyTorch로 TensorBoard를 사용
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()    # SummaryWriter 인스턴스를 생성


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    assert type(real_pred) == type(fake_pred), "real_pred must be the same type as fake_pred"
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input


# Tensorboard 시각화를 위한 denormalize
def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    if get_rank() == 0 and args.tf_log:
        from utils.visualizer import Visualizer
        vis = Visualizer(args)

    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)  # SummaryWriter 인스턴스를 생성

    train_loader = sample_data(train_loader)

    # 학습 파라미터 초기화
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    accum = 0.5 ** (32 / (10 * 1000))
    loss_dict = {}
    l2_loss = torch.nn.MSELoss()
    loss_dict = {}
    # L1 Loss for reconstruction
    l1_loss = torch.nn.L1Loss().to(device)  

    # 분산 학습 설정 (True/False)
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator


    print(" -- start training -- ")
    end = time.time()
    if args.ttur:
        args.G_lr = args.D_lr / 4
    if args.lr_decay:
        lr_decay_per_step = args.G_lr / (args.iter - args.lr_decay_start_steps)

    # loss 선언
    g_l2_loss = nn.MSELoss()
    g_lpips_loss = LPIPS()
    g_id_loss = IDLoss()

    # 학습 시작 (args.iter: 전체 반복 횟수)
    for idx in range(args.iter):
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        # Train D (Discriminator 학습)
        generator.train()

        # 실제 이미지 데이터 로드
        # real_img, _ = next(loader)
        # if not args.dataset == 'lmdb':
        if not args.lmdb:
            this_data = next(train_loader)
            real_img = this_data[0]
        else:
            real_img = next(train_loader)
        real_img = real_img.to(device)

        # requires_grad를 사용하여 생성자와 판별자의 학습을 번갈아 가면서 진행
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        fake_img, _ = generator(real_img)  # 생성자에 실제 이미지를 직접 전달
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred) * args.gan_weight    # Discriminator loss 계산
        
        if args.bcr:
            real_img_cr_aug = CR_DiffAug(real_img)
            fake_img_cr_aug = CR_DiffAug(fake_img)
            fake_pred_aug = discriminator(fake_img_cr_aug)
            real_pred_aug = discriminator(real_img_cr_aug)
            d_loss += args.bcr_fake_lambda * l2_loss(fake_pred_aug, fake_pred) \
                + args.bcr_real_lambda * l2_loss(real_pred_aug, real_pred)

        loss_dict["d"] = d_loss

        # backward()-step() 순서로 학습
        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)    # R1 정규화

            discriminator.zero_grad()
            (args.gan_weight * (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Train G (Generator 학습)
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        if not args.lmdb:
            this_data = next(train_loader)
            real_img = this_data[0]
        else:
            real_img = next(train_loader)
        real_img = real_img.to(device)


        fake_img, _ = generator(real_img)  # 다시 이미지 특징을 생성자에 전달하여 가짜 이미지 생성
        fake_pred = discriminator(fake_img)
        #g_loss = g_nonsaturating_loss(fake_pred)* args.gan_weight    # Generator loss 계산

        ### L1 recon_loss 계산
        recon_loss = l1_loss(fake_img, real_img)
        #g_loss = g_nonsaturating_loss(fake_pred) * args.gan_weight + recon_loss * args.recon_weight     # Generator loss 계산(GAN loss + recon loss)

        #### loss 추가 ####
        gen_adv_loss = g_nonsaturating_loss(fake_pred)
        gen_l2_loss = g_l2_loss(fake_img, real_img)
        gen_lpips_loss = g_lpips_loss(fake_img, real_img)
        gen_id_loss = g_id_loss(fake_img, real_img)
        g_loss = args.gan_weight * gen_adv_loss + args.recon_weight * gen_l2_loss + args.lpips_weight * gen_lpips_loss + args.id_weight * gen_id_loss


        loss_dict["recon"] = recon_loss
        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # EMA 업데이트
        accumulate(g_ema, g_module, accum)
        #accumulate(g_ema, g_module, 0.999)
    

        # Finish one iteration and reduce loss dict
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        # L1 손실 집계
        recon_val = loss_reduced["recon"].mean().item()

        ### TensorBoard에 손실 값 기록 (SummaryWriter 인스턴스 writer)
        if get_rank() == 0:
            # 중간 결과 이미지 저장
            if i % args.print_freq == 0:
                writer.add_scalar('Loss/Discriminator', d_loss_val, i)
                writer.add_scalar('Loss/Generator', g_loss_val, i)
                writer.add_scalar('Loss/R1', r1_val, i)
                #writer.add_scalar('Loss/Reconstruction', recon_loss.item(), i)
                writer.add_scalar('Loss/Reconstruction', recon_val, i)
                
                real_img_dn = denormalize(real_img)
                fake_img_dn = denormalize(fake_img.detach().cpu())
                writer.add_images('Generated Images', fake_img_dn, i)
                writer.add_images('Real Images', real_img_dn, i)


        if args.lr_decay and i > args.lr_decay_start_steps:
            args.G_lr -= lr_decay_per_step
            args.D_lr = args.G_lr * 4 if args.ttur else (args.D_lr - lr_decay_per_step)
        
            for param_group in d_optim.param_groups:
                param_group['lr'] = args.D_lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = args.G_lr

        # Log, save and evaluate
        if get_rank() == 0:
            if i % args.print_freq == 0:
                vis_loss = {
                    'd_loss': d_loss_val,
                    'g_loss': g_loss_val,
                    'r1_val': r1_val,
                    'recon_val': recon_val  # recon_loss 로깅에 추가
                    }
                if wandb and args.wandb:
                    wandb.log(vis_loss, step=i)
                iters_time = time.time() - end
                end = time.time()
                if args.lr_decay:
                    print("Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}\tRecon: {:.4f}\tG_lr: {:e}\tD_lr: {:e}".format(i, iters_time, d_loss_val, g_loss_val, r1_val, recon_val, args.G_lr, args.D_lr))
                else:
                    print("Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}\tRecon: {:.4f}".format(i, iters_time, d_loss_val, g_loss_val, r1_val, recon_val))
                if args.tf_log:
                    vis.plot_dict(vis_loss, step=(i * args.batch * int(os.environ["WORLD_SIZE"])))

            if i != 0 and i % args.eval_freq == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    args.checkpoint_path + f"/{str(i).zfill(6)}.pt",
                )
                
                # print("=> Evaluation ...")
                # g_ema.eval()
                # fid1 = evaluation(g_ema, args, i * args.batch * int(os.environ["WORLD_SIZE"]))
                # fid_dict = {'fid1': fid1}
                # if wandb and args.wandb:
                #     wandb.log({'fid': fid1}, step=i)
                # if args.tf_log:
                #     vis.plot_dict(fid_dict, step=(i * args.batch * int(os.environ["WORLD_SIZE"])))

            if i % args.save_freq == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    args.checkpoint_path + f"/{str(i).zfill(6)}.pt",
                )


def evaluation(generator, val_loader, args, steps):
    cnt = 0

    '''
    # 이미지 생성을 담당하는 루프    
    for _ in tqdm(range(args.val_num_batches)):
        with torch.no_grad():
            noise = torch.randn((args.val_batch_size, 512)).cuda()

            out_sample, _ = generator(noise)
            out_sample = tensor_transform_reverse(out_sample)

            if not os.path.exists(os.path.join(args.sample_path, "eval_{}".format(str(steps)))):
                os.mkdir(os.path.join(args.sample_path,
                                      "eval_{}".format(str(steps))))

            for j in range(args.val_batch_size):
                torchvision.utils.save_image(
                    out_sample[j],
                    os.path.join(args.sample_path, "eval_{}".format(
                        str(steps))) + f"/{str(cnt).zfill(6)}.png",
                    nrow=1,
                    padding=0,
                    normalize=True,
                    range=(0, 1),
                )
                cnt += 1
    '''
    
    gt_path = args.eval_gt_path
    device = torch.device('cuda:0')
    fid = fid_score.calculate_fid_given_paths([os.path.join(args.sample_path, "eval_{}".format(
        str(steps))), gt_path], batch_size=args.val_batch_size, device=device, dims=2048)

    print("Fid Score : ({:.2f}, {:.1f}M)".format(fid, steps / 1000000))

    return fid

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith(('.png', '.jpg', '.jpeg'))]

#     def __len__(self):              # 데이터셋의 총 이미지 개수를 반환
#         return len(self.image_paths)

#     def __getitem__(self, idx):     # 주어진 인덱스 idx에 해당하는 이미지 파일을 읽고, 변환을 적용하여 반환
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, None    
    
if __name__ == "__main__":
    device = "cuda"     # GPU 사용

    parser = argparse.ArgumentParser()

    #parser.add_argument("--path", type=str, default=None, help="Path of training data")
    # parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/images1024x1024', help="Path of training data")
    parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/train', help="Path of training data")

    # 학습 루프가 실행될 총 반복 횟수
    parser.add_argument("--iter", type=int, default=800000)
    #parser.add_argument("--iter", type=int, default=100)
    
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)    #224
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    
    # "checkpoint" 모델의 가중치나 학습 상태를 포함
    #parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/FFHQ_256.pt')
    # parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/CelebAHQ_256.pt')
    #parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/LSUNChurch_256.pt')
    
    parser.add_argument("--G_lr", type=float, default=0.0002)
    parser.add_argument("--D_lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--start_dim", type=int, default=512, help="Start dim of generator input dim")
    parser.add_argument("--D_channel_multiplier", type=int, default=2)
    #parser.add_argument("--G_channel_multiplier", type=int, default=1)
    parser.add_argument("--G_channel_multiplier", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=20000)
    
    # 학습 과정 동안 모델이 평가되는 빈도
    parser.add_argument("--eval_freq", type=int, default=50000)
    #parser.add_argument("--eval_freq", type=int, default=50)

    ### loss weight ### 
    parser.add_argument('--gan_weight', default=1, type=float, help='Gan loss weight')
    parser.add_argument('--recon_weight', default=1.0, type=float, help='Recon loss weight')
    parser.add_argument('--lpips_weight', default=0.8, type=float, help='LPIPS loss weight')
    parser.add_argument('--id_weight', default=0.1, type=float, help='ID loss weight')

    parser.add_argument('--workers', default=8, type=int, help='Number of workers')
    parser.add_argument('--checkpoint_path', default='checkpoints', type=str, help='Save checkpoints')    # 모델 가중치, 옵티마이저 상태 등을 저장
    parser.add_argument('--sample_path', default='samples', type=str, help='Save sample')     # 학습 과정에서 생성된 샘플 이미지 저장
    parser.add_argument('--start_iter', default=0, type=int, help='Start iter number')
    parser.add_argument('--tf_log', action="store_true", help='If we use tensorboard file')
    parser.add_argument('--val_num_batches', default=1250, type=int, help='Num of batches will be generated during evalution')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size during evalution')
    parser.add_argument('--D_sn', action="store_true", help='If we use spectral norm in D')
    parser.add_argument('--ttur', action="store_true", help='If we use TTUR during training')
    parser.add_argument('--eval', action="store_true", help='Only do evaluation')
    parser.add_argument("--eval_iters", type=int, default=0, help="Iters of evaluation ckpt")
    # parser.add_argument('--eval_gt_path', default='/home/alisa/Documents/Datasets/FFHQ/images1024x1024/00000', type=str, help='Path to ground truth images to evaluate FID score')
    parser.add_argument('--eval_gt_path', default='/home/alisa/Documents/Datasets/FFHQ/val', type=str, help='Path to ground truth images to evaluate FID score')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio in swin')
    parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
    parser.add_argument("--bcr", action="store_true", help='If we add bcr during training')
    parser.add_argument("--bcr_fake_lambda", default=10, type=float, help='Bcr weight for fake data')
    parser.add_argument("--bcr_real_lambda", default=10, type=float, help='Bcr weight for real data')
    parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from checkpoint")
    parser.add_argument("--lmdb", action="store_true", help='Whether to use lmdb datasets')
    parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')
    parser.add_argument("--use_flip", action="store_true", help='Whether to use random flip in training')
    parser.add_argument("--wandb", action="store_true", help='Whether to use wandb record training')
    parser.add_argument("--project_name", type=str, default='StyleSwin', help='Project name')
    parser.add_argument("--lr_decay", action="store_true", help='Whether to use lr decay')
    parser.add_argument("--lr_decay_start_steps", default=800000, type=int, help='Steps to start lr decay')

    ### 모델 이름을 인수로 추가
    parser.add_argument('--model_name', type=str, default='ViT', help='Name of the model to use')
    ### TensorBoard 로그 디렉토리 추가
    parser.add_argument('--tensorboard_log_dir', default='./tensorboard_logs', type=str, help='TensorBoard log directory')

    args = parser.parse_args()

    # batch size 확인
    if args.batch <= 0:
        raise ValueError(f"Invalid batch size: {args.batch}. It should be a positive integer.")
    if args.val_batch_size <= 0:
        raise ValueError(f"Invalid validation batch size: {args.val_batch_size}. It should be a positive integer.")


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 4096
    args.n_mlp = 8 
    args.g_reg_every = 10000000    # We do not apply regularization on G

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(0, 18000))
        synchronize()

    if args.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass


    # 현재 날짜와 시간을 기반으로 폴더 이름 생성 ('%Y%m%d_%H%M%S')
    current_time = datetime.now().strftime('%m%d')

    # 사용된 모델의 이름을 얻기 (예: 'ViT')
    model_name = args.model_name

    # 경로 문자열을 생성 ('날짜_모델명')
    directory_name = f"{current_time}_{model_name}"

    # 분산 학습 환경에서 주 프로세스(랭크 0)에서만 샘플 이미지와 체크포인트 파일을 저장할 디렉토리를 생성한다. (이를 통해 디렉토리 생성 작업을 한 번만 수행)
    if get_rank() == 0:
        args.sample_path = os.path.join('./results/samples', directory_name)
        args.checkpoint_path = os.path.join('./results/checkpoints', directory_name)
        args.tensorboard_log_dir = os.path.join('./tensorboard_logs', directory_name)
        if not os.path.exists(args.sample_path):
            os.makedirs(args.sample_path)
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        if not os.path.exists(args.tensorboard_log_dir):
            os.makedirs(args.tensorboard_log_dir)
    '''
        if get_rank() == 0:
            args.sample_path = os.path.join(args.sample_path, 'samples')
            if not os.path.exists(args.sample_path):
                os.mkdir(args.sample_path)
    '''
    generator = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn).to(device)
    g_ema = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # Load model checkpoint.
    if args.ckpt is not None:
        print("load model: ", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.ckpt)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)
        try:
            discriminator.load_state_dict(ckpt["d"])
        except:
            print("We don't load D.")

    print("-" * 80)
    print("Generator: ")
    print(generator)
    print("-" * 80)
    print("Discriminator: ")
    print(discriminator)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    
    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.G_lr * g_reg_ratio if not args.ttur else args.D_lr / 4 * g_reg_ratio,
        betas=(args.beta1 ** g_reg_ratio, args.beta2 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.D_lr * d_reg_ratio,
        betas=(args.beta1 ** d_reg_ratio, args.beta2 ** d_reg_ratio),
    )

    # Load optimizer checkpoint.
    if args.ckpt is not None:
        print("load optimizer: ", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.ckpt)

        try:
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
        except:
            print("We don't load optimizers.")

    if args.eval:
        if get_rank() == 0:
            g_ema.eval()
            evaluation(g_ema, args, (args.eval_iters * args.batch * int(os.environ["WORLD_SIZE"])))
            sys.exit(0)
        sys.exit(0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.use_flip:
        transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.ToTensor(),
                normalize
            ]
        )



    # 데이터셋 로드
    if args.lmdb:
        dataset = MultiResolutionDataset(args.path, transform, args.size)
    else:
        train_dataset = datasets.ImageFolder(root=args.path, transform=transform)
        val_dataset = datasets.ImageFolder(root=args.eval_gt_path, transform=transform)

        
    # datasets.ImageFolder 클래스를 사용하여 상위 디렉토리 경로를 지정하면 하위 디렉토리의 모든 이미지를 로드
    # root 파라미터에 데이터셋의 상위 디렉토리 경로를 지정
    # transform 파라미터에 이미지에 적용할 변환을 지정

    # 데이터로더 생성            
    train_loader = data.DataLoader(   
        train_dataset,
        batch_size=args.batch,      # 한 번에 로드할 이미지의 개수를 지정
        num_workers=args.workers,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    val_loader = data.DataLoader(   
        val_dataset,
        batch_size=args.batch,      # 한 번에 로드할 이미지의 개수를 지정
        num_workers=args.workers,
        sampler=data_sampler(val_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.project_name)

    train(args, train_loader, val_loader, generator, discriminator, g_optim, d_optim, g_ema, device)
    # evaluation(g_ema, args, 0)
