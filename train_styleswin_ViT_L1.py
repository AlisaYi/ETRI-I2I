# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import builtins
import os
import sys
from datetime import timedelta, datetime
from tqdm import tqdm
import itertools

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
import glob

from dataset.dataset import Im2ImDataset
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

def cycle_consistency_loss(real_img, reconstructed_images, lambda_cycle=10.0):
    """
    Calculate the cycle consistency loss.
    
    Parameters:
    - real_img: 원본 이미지 텐서
    - reconstructed_images: 생성된 후 복원된 이미지 텐서
    - lambda_cycle: 사이클 일관성 손실의 가중치 (기본값 10.0)
    
    Returns:
    - loss: L1 사이클 일관성 손실
    """
    loss = nn.L1Loss()
    return lambda_cycle * loss(reconstructed_images, real_img)

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


def train(args, train_loader, val_loader, generator_A2B, generator_B2A, discriminator_A, discriminator_B, g_optim_A2B, g_optim_B2A, d_optim_A, d_optim_B, g_ema_A2B, g_ema_B2A, device):
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
        g_module_A2B = generator_A2B.module
        g_module_B2A = generator_B2A.module
        d_module_A = discriminator_A.module
        d_module_B = discriminator_B.module
    else:
        g_module_A2B = generator_A2B
        g_module_B2A = generator_B2A
        d_module_A = discriminator_A
        d_module_B = discriminator_B

    print(" -- start training -- ")
    end = time.time()
    if args.ttur:
        args.G_lr = args.D_lr / 4
    if args.lr_decay:
        lr_decay_per_step = args.G_lr / (args.iter - args.lr_decay_start_steps)

    # loss 선언
    g_l2_loss = nn.MSELoss()
    g_lpips_loss = LPIPS()
    l1_loss = nn.L1Loss()

    i = 0

    # 학습 시작 (args.iter: 전체 반복 횟수)
    for epoch in range(args.epochs):

        for iter, inputs in enumerate(tqdm(train_loader)):

            # Train D (Discriminator 학습)
            generator_A2B.train()
            generator_B2A.train()

            # 실제 이미지 데이터 로드
            # if not args.lmdb:
            #     this_data = next(train_loader)
            #     real_A = this_data[0].to(device)
            #     real_B = this_data[1].to(device)
            # else:
            #     real_A = next(train_loader).to(device)
            #     real_B = next(train_loader).to(device)
            real_A = inputs['A'].to(device)
            real_B = inputs['B'].to(device)

            # requires_grad를 사용하여 생성자와 판별자의 학습을 번갈아 가면서 진행
            requires_grad(generator_A2B, False)
            requires_grad(generator_B2A, False)
            requires_grad(discriminator_A, True)
            requires_grad(discriminator_B, True)

            fake_B, _ = generator_A2B(real_A)  # 생성자에 실제 이미지를 직접 전달하여 B 도메인의 가짜 이미지 생성
            fake_A, _ = generator_B2A(real_B)  # 생성자에 실제 이미지를 직접 전달하여 A 도메인의 가짜 이미지 생성

            fake_B_pred = discriminator_B(fake_B)
            fake_A_pred = discriminator_A(fake_A)
            real_B_pred = discriminator_B(real_B)
            real_A_pred = discriminator_A(real_A)

            d_loss_B = d_logistic_loss(real_B_pred, fake_B_pred) * args.gan_weight    # Discriminator B의 loss 계산
            d_loss_A = d_logistic_loss(real_A_pred, fake_A_pred) * args.gan_weight    # Discriminator A의 loss 계산
            
            loss_dict["d_B"] = d_loss_B
            loss_dict["d_A"] = d_loss_A

            if args.bcr:
                real_A_cr_aug = CR_DiffAug(real_A)
                real_B_cr_aug = CR_DiffAug(real_B)
                fake_A_cr_aug = CR_DiffAug(fake_A)
                fake_B_cr_aug = CR_DiffAug(fake_B)

                fake_pred_A_aug = discriminator_A(fake_A_cr_aug)
                real_pred_A_aug = discriminator_A(real_A_cr_aug)
                fake_pred_B_aug = discriminator_B(fake_B_cr_aug)
                real_pred_B_aug = discriminator_B(real_B_cr_aug)

                d_loss_A += args.bcr_fake_lambda * l2_loss(fake_pred_A_aug, fake_A_pred) + args.bcr_real_lambda * l2_loss(real_pred_A_aug, real_A_pred)
                d_loss_B += args.bcr_fake_lambda * l2_loss(fake_pred_B_aug, fake_B_pred) + args.bcr_real_lambda * l2_loss(real_pred_B_aug, real_B_pred)

            # loss_dict에 저장
            loss_dict["d_A"] = d_loss_A
            loss_dict["d_B"] = d_loss_B

            # backward()-step() 순서로 학습
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            d_loss_A.backward()
            d_loss_B.backward()

            nn.utils.clip_grad_norm_(discriminator_A.parameters(), 5.0)
            nn.utils.clip_grad_norm_(discriminator_B.parameters(), 5.0)

            d_optim_A.step()
            d_optim_B.step()

            d_regularize = i % args.d_reg_every == 0
            if d_regularize:
                real_A.requires_grad = True
                real_B.requires_grad = True

                real_A_pred = discriminator_A(real_A)
                real_B_pred = discriminator_B(real_B)
                r1_loss_A = d_r1_loss(real_A_pred, real_A)    # R1 정규화
                r1_loss_B = d_r1_loss(real_B_pred, real_B)    # R1 정규화

                discriminator_A.zero_grad()
                discriminator_B.zero_grad()
                (args.gan_weight * (args.r1 / 2 * r1_loss_A * args.d_reg_every + 0 * real_A_pred[0])).backward()
                (args.gan_weight * (args.r1 / 2 * r1_loss_B * args.d_reg_every + 0 * real_B_pred[0])).backward()

                d_optim_A.step()
                d_optim_B.step()

            loss_dict["r1_A"] = r1_loss_A
            loss_dict["r1_B"] = r1_loss_B

            # Train G (Generator 학습)
            requires_grad(generator_A2B, True)
            requires_grad(generator_B2A, True)
            requires_grad(discriminator_A, False)
            requires_grad(discriminator_B, False)

            real_A = inputs['A'].to(device)
            real_B = inputs['B'].to(device)

            # 가짜 이미지 생성
            fake_B, _ = generator_A2B(real_A)  # 다시 이미지 특징을 생성자에 전달하여 가짜 이미지 생성
            fake_A, _ = generator_B2A(real_B)
            fake_B_pred = discriminator_B(fake_B)
            fake_A_pred = discriminator_A(fake_A)

            # Generator loss (GAN loss + cycle consistency loss)
            g_loss_A2B = g_nonsaturating_loss(fake_B_pred)
            g_loss_B2A = g_nonsaturating_loss(fake_A_pred)

            # L1 Cycle consistency
            reconstructed_A, _ = generator_B2A(fake_B)
            reconstructed_B, _ = generator_A2B(fake_A)
            cycle_loss_A = cycle_consistency_loss(real_A, reconstructed_A)
            cycle_loss_B = cycle_consistency_loss(real_B, reconstructed_B)
            cycle_loss = cycle_loss_A + cycle_loss_B

            #### loss 추가 ####
            # L2 손실 계산
            # gen_l2_loss_A = g_l2_loss(fake_A, real_A)
            # gen_l2_loss_B = g_l2_loss(fake_B, real_B)  
            # LPIPS 손실 계산      
            # gen_lpips_loss_A = g_lpips_loss(fake_A, real_A)
            # gen_lpips_loss_B = g_lpips_loss(fake_B, real_B)
            # ID 손실 계산
            # gen_id_loss_A = g_id_loss(fake_A, real_A)
            # gen_id_loss_B = g_id_loss(fake_B, real_B)

            # 최종 Generator 손실 계산
            g_loss_A2B = args.gan_weight * g_loss_A2B + \
                        args.recon_weight * cycle_loss #+ \
                        #args.lpips_weight * gen_lpips_loss_B

            g_loss_B2A = args.gan_weight * g_loss_B2A + \
                        args.recon_weight * cycle_loss #+ \
                        #args.lpips_weight * gen_lpips_loss_A

            # 전체 Generator 손실
            g_loss = g_loss_A2B + g_loss_B2A

            loss_dict["g"] = g_loss
            loss_dict["g_A2B"] = g_loss_A2B
            loss_dict["g_B2A"] = g_loss_B2A
            loss_dict["cycle"] = cycle_loss

            # loss_dict["recon"] = recon_loss
            # loss_dict["g"] = g_loss

            generator_A2B.zero_grad()
            generator_B2A.zero_grad()
            g_loss.backward()
            g_optim_A2B.step()
            g_optim_B2A.step()

            # EMA 업데이트
            accumulate(g_ema_A2B, g_module_A2B, accum)
            accumulate(g_ema_B2A, g_module_B2A, accum)        
            #accumulate(g_ema, g_module, 0.999)
        

            # Finish one iteration and reduce loss dict
            loss_reduced = reduce_loss_dict(loss_dict)
            d_loss_val_A = loss_reduced["d_A"].mean().item()
            d_loss_val_B = loss_reduced["d_B"].mean().item()
            g_loss_val_A2B = loss_reduced["g_A2B"].mean().item()
            g_loss_val_B2A = loss_reduced["g_B2A"].mean().item()
            cycle_loss_val = loss_reduced["cycle"].mean().item()
            r1_val_A = loss_reduced["r1_A"].mean().item()
            r1_val_B = loss_reduced["r1_B"].mean().item()
            # L1 손실 집계
            # recon_val = loss_reduced["recon"].mean().item()

            ### TensorBoard에 손실 값 기록 (SummaryWriter 인스턴스 writer)
            if get_rank() == 0:
                # 중간 결과 이미지 저장
                if i % args.print_freq == 0:
                    writer.add_scalar('Loss/Discriminator_A', d_loss_val_A, i)
                    writer.add_scalar('Loss/Discriminator_B', d_loss_val_B, i)
                    writer.add_scalar('Loss/Generator_A2B', g_loss_val_A2B, i)
                    writer.add_scalar('Loss/Generator_B2A', g_loss_val_B2A, i)
                    writer.add_scalar('Loss/R1_A', r1_val_A, i)
                    writer.add_scalar('Loss/R1_B', r1_val_B, i)
                    writer.add_scalar('Loss/Cycle', cycle_loss_val, i)
                    
                    real_A_dn = denormalize(real_A.detach().cpu())
                    real_B_dn = denormalize(real_B.detach().cpu())
                    fake_A_dn = denormalize(fake_A.detach().cpu())
                    fake_B_dn = denormalize(fake_B.detach().cpu())
                    writer.add_images('Real A', real_A_dn, i)
                    writer.add_images('Real B', real_B_dn, i)
                    writer.add_images('Fake A', fake_A_dn, i)
                    writer.add_images('Fake B', fake_B_dn, i)


            if args.lr_decay and i > args.lr_decay_start_steps:
                args.G_lr -= lr_decay_per_step
                args.D_lr = args.G_lr * 4 if args.ttur else (args.D_lr - lr_decay_per_step)
            
                # Update learning rates for discriminators
                for param_group in d_optim_A.param_groups:
                    param_group['lr'] = args.D_lr
                for param_group in d_optim_B.param_groups:
                    param_group['lr'] = args.D_lr
                
                # Update learning rates for generators
                for param_group in g_optim_A2B.param_groups:
                    param_group['lr'] = args.G_lr
                for param_group in g_optim_B2A.param_groups:
                    param_group['lr'] = args.G_lr
                    
            # Log, save and evaluate
            if get_rank() == 0:
                if i % args.print_freq == 0:
                    vis_loss = {
                        'd_loss_A': d_loss_val_A,
                        'd_loss_B': d_loss_val_B,
                        'g_loss_A2B': g_loss_val_A2B,
                        'g_loss_B2A': g_loss_val_B2A,
                        'cycle_loss': cycle_loss_val,
                        'r1_val_A': r1_val_A,
                        'r1_val_B': r1_val_B
                        }
                    if wandb and args.wandb:
                        wandb.log(vis_loss, step=i)
                    iters_time = time.time() - end
                    end = time.time()
                    if args.lr_decay:
                        print("Iters: {}\tTime: {:.4f}\tD_loss_A: {:.4f}\tD_loss_B: {:.4f}\tG_loss_A2B: {:.4f}\tG_loss_B2A: {:.4f}\tCycle_loss: {:.4f}\tR1_A: {:.4f}\tR1_B: {:.4f}\tG_lr: {:e}\tD_lr: {:e}".format(i, iters_time, d_loss_val_A, d_loss_val_B, g_loss_val_A2B, g_loss_val_B2A, cycle_loss_val, r1_val_A, r1_val_B, args.G_lr, args.D_lr))
                    else:
                        print("Iters: {}\tTime: {:.4f}\tD_loss_A: {:.4f}\tD_loss_B: {:.4f}\tG_loss_A2B: {:.4f}\tG_loss_B2A: {:.4f}\tCycle_loss: {:.4f}\tR1_A: {:.4f}\tR1_B: {:.4f}".format(i, iters_time, d_loss_val_A, d_loss_val_B, g_loss_val_A2B, g_loss_val_B2A, cycle_loss_val, r1_val_A, r1_val_B))
                    if args.tf_log:
                        vis.plot_dict(vis_loss, step=(i * args.batch * int(os.environ["WORLD_SIZE"])))

                if i != 0 and i % args.eval_freq == 0:
                    torch.save(
                        {
                            "g_A2B": g_module_A2B.state_dict(),
                            "g_B2A": g_module_B2A.state_dict(),
                            "d_A": d_module_A.state_dict(),
                            "d_B": d_module_B.state_dict(),
                            "g_ema_A2B": g_ema_A2B.state_dict(),
                            "g_ema_B2A": g_ema_B2A.state_dict(),
                            "g_optim_A2B": g_optim_A2B.state_dict(),
                            "g_optim_B2A": g_optim_B2A.state_dict(),
                            "d_optim_A": d_optim_A.state_dict(),
                            "d_optim_B": d_optim_B.state_dict(),
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
                            "g_A2B": g_module_A2B.state_dict(),
                            "g_B2A": g_module_B2A.state_dict(),
                            "d_A": d_module_A.state_dict(),
                            "d_B": d_module_B.state_dict(),
                            "g_ema_A2B": g_ema_A2B.state_dict(),
                            "g_ema_B2A": g_ema_B2A.state_dict(),
                            "g_optim_A2B": g_optim_A2B.state_dict(),
                            "g_optim_B2A": g_optim_B2A.state_dict(),
                            "d_optim_A": d_optim_A.state_dict(),
                            "d_optim_B": d_optim_B.state_dict(),
                            "args": args,
                        },
                        args.checkpoint_path + f"/{str(i).zfill(6)}.pt",
                    )
            i += 1


def evaluation(g_ema_A2B, g_ema_B2A, generator, val_loader, args, steps):
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

    
if __name__ == "__main__":
    device = "cuda"     # GPU 사용

    parser = argparse.ArgumentParser()

    #parser.add_argument("--path", type=str, default=None, help="Path of training data")
    # parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/images1024x1024', help="Path of training data")
    parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/train', help="Path of training data")
    # parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/360_Indoor', help="Path of training data")

    # 학습 루프가 실행될 총 반복 횟수
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--epochs", type=int, default=200)
    #parser.add_argument("--iter", type=int, default=100)
    
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)    #224
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    
    # "checkpoint" 모델의 가중치나 학습 상태를 포함
    # parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/FFHQ_256.pt')
    # parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/CelebAHQ_256.pt')
    # parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/LSUNChurch_256.pt')
    parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/Cityscapes_040000.pt')
    
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
    parser.add_argument('--recon_weight', default=0.1, type=float, help='Recon loss weight')
    parser.add_argument('--lpips_weight', default=0.8, type=float, help='LPIPS loss weight')
    parser.add_argument('--id_weight', default=0.1, type=float, help='ID loss weight')
    parser.add_argument('--l2_weight', default=0.1, type=float, help='L2 weight')

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
    parser.add_argument('--model_name', type=str, default='ViT_L1', help='Name of the model to use')
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

    # Generator 및 EMA 정의
    generator_A2B = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    generator_B2A = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    g_ema_A2B = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    g_ema_B2A = Generator_MultiResolution(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    g_ema_A2B.eval()
    g_ema_B2A.eval()
    accumulate(g_ema_A2B, generator_A2B, 0)
    accumulate(g_ema_B2A, generator_B2A, 0)

    discriminator_A = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn).to(device)
    discriminator_B = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn).to(device)


    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # Load model checkpoint. -> 변수명 임의로 바꾸면 X
    if args.ckpt is not None:
        print("load model: ", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.ckpt)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass

        generator_A2B.load_state_dict(ckpt["g"], strict=False)
        generator_B2A.load_state_dict(ckpt["g"], strict=False)
        g_ema_A2B.load_state_dict(ckpt["g_ema"], strict=False)
        g_ema_B2A.load_state_dict(ckpt["g_ema"], strict=False)
        try:
            discriminator_A.load_state_dict(ckpt["d"])
            discriminator_B.load_state_dict(ckpt["d"])
        except:
            print("We don't load D.")

    # print("-" * 80)
    # print("Generator A2B: ")
    # print(generator_A2B)
    # print("-" * 80)
    # print("Generator B2A: ")
    # print(generator_B2A)
    # print("-" * 80)
    # print("Discriminator A: ")
    # print(discriminator_A)
    # print("-" * 80)
    # print("Discriminator B: ")
    # print(discriminator_B)

    if args.distributed:
        generator_A2B = nn.parallel.DistributedDataParallel(
            generator_A2B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        generator_B2A = nn.parallel.DistributedDataParallel(
            generator_B2A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator_A = nn.parallel.DistributedDataParallel(
            discriminator_A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator_B = nn.parallel.DistributedDataParallel(
            discriminator_B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # EMA 업데이트 시 올바른 모듈 사용
    if args.distributed:
        g_module_A2B = generator_A2B.module
        g_module_B2A = generator_B2A.module
    else:
        g_module_A2B = generator_A2B
        g_module_B2A = generator_B2A


    g_optim = optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
        lr=args.G_lr * g_reg_ratio if not args.ttur else args.D_lr / 4 * g_reg_ratio,
        betas=(args.beta1 ** g_reg_ratio, args.beta2 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()),
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
            g_ema_A2B.eval()
            g_ema_B2A.eval()
            evaluation(g_ema_A2B, g_ema_B2A, args, (args.eval_iters * args.batch * int(os.environ["WORLD_SIZE"])))
            sys.exit(0)
        sys.exit(0)

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_cfg = argparse.Namespace(height=args.size)  # data_cfg.height가 args.size를 참조하도록 설정

    train_transform = transforms.Compose([
        transforms.Resize(size=data_cfg.height),
        transforms.RandomCrop(size=data_cfg.height),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=data_cfg.height),
        transforms.CenterCrop(size=data_cfg.height),
        transforms.ToTensor(),
        normalize
    ])

    # 변환 선택
    if args.use_flip:
        transform = train_transform
    else:
        transform = val_transform


    imgs_A_train_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/GTA5/train_test_split/train/images/*'))
    imgs_A_test_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/GTA5/train_test_split/test/images/*'))
    imgs_B_train_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/Cityscape/train_test_split/train/images/*'))
    imgs_B_test_pth = sorted(glob.glob('/home/alisa/Documents/Datasets/Cityscape/train_test_split/test/images/*'))

    dataset_train = Im2ImDataset(imgs_A=imgs_A_train_pth, imgs_B=imgs_B_train_pth, transform=train_transform)
    dataset_val = Im2ImDataset(imgs_A=imgs_A_test_pth, imgs_B=imgs_B_train_pth, transform=val_transform)

    '''
    # 데이터셋 로드
    if args.lmdb:
        dataset = MultiResolutionDataset(args.path, transform, args.size)
    else:
        train_dataset = datasets.ImageFolder(root=args.path, transform=transform)
        val_dataset = datasets.ImageFolder(root=args.eval_gt_path, transform=transform)
    '''

    # 데이터로더 생성            
    train_loader = data.DataLoader(   
        dataset_train,
        batch_size=args.batch,      # 한 번에 로드할 이미지의 개수를 지정
        num_workers=args.workers,
        sampler=data_sampler(dataset_train, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    val_loader = data.DataLoader(   
        dataset_val,
        batch_size=args.batch,      # 한 번에 로드할 이미지의 개수를 지정
        num_workers=args.workers,
        sampler=data_sampler(dataset_val, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.project_name)


train(args, train_loader, val_loader, generator_A2B, generator_B2A, discriminator_A, discriminator_B, g_optim, d_optim, g_ema_A2B, g_ema_B2A, device)
# evaluation(g_ema, args, 0)
