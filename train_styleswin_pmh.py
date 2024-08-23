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
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.swin_transformer_v2 import SwinTransformerV2

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


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad_g(model, flag=True):
    env_list = ['swin_enc', 'fusers', 'linear_', 'feat2w']
    for name, p in model.named_parameters():
        if any(item in name for item in env_list): 
            p.requires_grad = flag
        else:
            p.requires_grad = False

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
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

def train(args, train_loader, test_loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer):
    if get_rank() == 0 and args.tf_log:
        from utils.visualizer import Visualizer
        vis = Visualizer(args)

    train_loader = sample_data(train_loader)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    accum = 0.5 ** (32 / (10 * 1000))
    loss_dict = {}
    l2_loss = torch.nn.MSELoss()
    loss_dict = {}

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
        
    g_l2_loss = nn.MSELoss()
    g_lpips_loss = LPIPS()
    g_id_loss = IDLoss()

    for idx in range(args.iter):
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break

        # Train D
        generator.train()
        if not args.dataset == 'lmdb':
            this_data = next(train_loader)
            real_img = this_data[0]
        else:
            real_img = next(train_loader)
        real_img = real_img.to(device)

        requires_grad_g(generator, False)
        requires_grad(discriminator, True)
        
        fake_img, _ = generator(real_img)
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred) * args.gan_weight
        
        if args.bcr:
            real_img_cr_aug = CR_DiffAug(real_img)
            fake_img_cr_aug = CR_DiffAug(fake_img)
            fake_pred_aug = discriminator(fake_img_cr_aug)
            real_pred_aug = discriminator(real_img_cr_aug)
            d_loss += args.bcr_fake_lambda * l2_loss(fake_pred_aug, fake_pred) \
                + args.bcr_real_lambda * l2_loss(real_pred_aug, real_pred)

        loss_dict["d"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.gan_weight * (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0])).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        # Train G
        requires_grad_g(generator, True)
        requires_grad(discriminator, False)

        if not args.dataset == 'lmdb':
            this_data = next(train_loader)
            real_img = this_data[0]
        else:
            real_img = next(train_loader)
        real_img = real_img.to(device)

        fake_img, _ = generator(real_img)
        fake_pred = discriminator(fake_img)
        
        gen_adv_loss = g_nonsaturating_loss(fake_pred)
        gen_l2_loss = g_l2_loss(fake_img, real_img)
        gen_lpips_loss = g_lpips_loss(fake_img, real_img)
        gen_id_loss = g_id_loss(fake_img, real_img)
        g_loss = args.gan_weight * gen_adv_loss + args.recon_weight * gen_l2_loss + args.lpips_weight * gen_lpips_loss + args.id_weight * gen_id_loss

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        # Finish one iteration and reduce loss dict
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()

        if args.lr_decay and i > args.lr_decay_start_steps:
            args.G_lr -= lr_decay_per_step
            args.D_lr = args.G_lr * 4 if args.ttur else (args.D_lr - lr_decay_per_step)
        
            for param_group in d_optim.param_groups:
                param_group['lr'] = args.D_lr
            for param_group in g_optim.param_groups:
                param_group['lr'] = args.G_lr

        # Log, save and evaluate
        if get_rank() == 0:
            if i % args.log_freq == 0:
                writer.add_scalar('Loss/d_loss', d_loss_val, i)
                writer.add_scalar('Loss/g_adv_loss', gen_adv_loss.item(), i)
                writer.add_scalar('Loss/g_l2_loss', gen_l2_loss.item(), i)
                writer.add_scalar('Loss/g_lpips_loss', gen_lpips_loss.item(), i)
                writer.add_scalar('Loss/g_id_loss', gen_id_loss.item(), i)
                writer.add_scalar('Loss/r1_loss', r1_val, i)
                
                real_img_dn = denormalize(real_img)
                fake_img_dn = denormalize(fake_img.detach().cpu())
                writer.add_images('Image/real', real_img_dn, i)
                writer.add_images('Image/generated', fake_img_dn, i)
            
            if i % args.print_freq == 0:
                vis_loss = {
                    'd_loss': d_loss_val,
                    'g_loss': g_loss_val,
                    'r1_val': r1_val,
                    }
                if wandb and args.wandb:
                    wandb.log(vis_loss, step=i)
                iters_time = time.time() - end
                end = time.time()
                if args.lr_decay:
                    print("Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}\tG_lr: {:e}\tD_lr: {:e}".format(i, iters_time, d_loss_val, g_loss_val, r1_val, args.G_lr, args.D_lr))
                else:
                    print("Iters: {}\tTime: {:.4f}\tD_loss: {:.4f}\tG_loss: {:.4f}\tR1: {:.4f}".format(i, iters_time, d_loss_val, g_loss_val, r1_val))
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
                
                print("=> Evaluation ...")
                g_ema.eval()
                evaluation(g_ema, test_loader, args, i * args.batch * int(os.environ["WORLD_SIZE"]))#i * args.batch * int(os.environ["WORLD_SIZE"])) 
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


def evaluation(generator, test_loader, args, steps):
    cnt = 0

    for i, inputs in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            image = inputs[0].cuda()

            out_sample, _ = generator(image)
            out_sample = tensor_transform_reverse(out_sample)

            if not os.path.exists(os.path.join(args.sample_path, "eval_{}".format(str(steps)))):
                os.mkdir(os.path.join(args.sample_path,
                                      "eval_{}".format(str(steps))))

            torchvision.utils.save_image(
                out_sample[0],
                os.path.join(args.sample_path, "eval_{}".format(
                    str(steps))) + f"/{str(i).zfill(6)}.png",
                nrow=1,
                padding=0,
                normalize=True,
                range=(0, 1),
            )
    
    # gt_path = args.eval_gt_path
    # device = torch.device('cuda:0')
    # fid = fid_score.calculate_fid_given_paths([os.path.join(args.sample_path, "eval_{}".format(
    #     str(steps))), gt_path], batch_size=args.val_batch_size, device=device, dims=2048)

    # print("Fid Score : ({:.2f}, {:.1f}M)".format(fid, steps / 1000000))

    # return fid

def evaluate_fid(sample_path):
    gt_path = args.eval_gt_path + '/images'
    device = torch.device('cuda:0')
    fid = fid_score.calculate_fid_given_paths([sample_path, gt_path], batch_size=args.val_batch_size, device=device, dims=2048)

    print("Fid Score : ({:.2f})".format(fid))

    return fid
    
def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    # parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/images1024x1024', help="Path of training data")
    parser.add_argument("--path", type=str, default='/home/alisa/Documents/Datasets/FFHQ/train', help="Path of training data")

    parser.add_argument("--iter", type=int, default=1500000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--r1", type=float, default=5)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default='/home/alisa/i2i/StyleSwin-main/pretrained_styleswin/FFHQ_256.pt')
    parser.add_argument("--G_lr", type=float, default=0.00001)
    parser.add_argument("--D_lr", type=float, default=0.00001)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--start_dim", type=int, default=512, help="Start dim of generator input dim")
    parser.add_argument("--D_channel_multiplier", type=int, default=2)
    parser.add_argument("--G_channel_multiplier", type=int, default=2)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=2000) 
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument('--workers', default=8, type=int, help='Number of workers')
    
    parser.add_argument('--gan_weight', default=0.01, type=float, help='Gan loss weight')
    parser.add_argument('--recon_weight', default=1.0, type=float, help='Recon loss weight')
    parser.add_argument('--lpips_weight', default=0.8, type=float, help='LPIPS loss weight')
    parser.add_argument('--id_weight', default=0.1, type=float, help='ID loss weight')
    
    parser.add_argument('--save_path', default='./results/ffhq_I2I_Swin', type=str, help='Save checkpoints')
    parser.add_argument('--start_iter', default=0, type=int, help='Start iter number')
    parser.add_argument('--tf_log', action="store_true", help='If we use tensorboard file')
    parser.add_argument('--val_num_batches', default=1250, type=int, help='Num of batches will be generated during evalution')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size during evalution')
    parser.add_argument('--D_sn', default=True, help='If we use spectral norm in D')
    parser.add_argument('--ttur', default=True, help='If we use TTUR during training')
    parser.add_argument('--eval', action="store_true", help='Only do evaluation')
    parser.add_argument("--eval_iters", type=int, default=0, help="Iters of evaluation ckpt")
    # parser.add_argument('--eval_gt_path', default='/home/alisa/Documents/Datasets/FFHQ/images1024x1024', type=str, help='Path to ground truth images to evaluate FID score')
    parser.add_argument('--eval_gt_path', default='/home/alisa/Documents/Datasets/FFHQ/val', type=str, help='Path to ground truth images to evaluate FID score')
    
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio in swin')
    parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
    parser.add_argument("--bcr", default=True, help='If we add bcr during training')
    parser.add_argument("--bcr_fake_lambda", default=10, type=float, help='Bcr weight for fake data')
    parser.add_argument("--bcr_real_lambda", default=10, type=float, help='Bcr weight for real data')
    parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from checkpoint")
    parser.add_argument("--dataset", default='ffhq', help='Whether to use lmdb datasets')
    parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')
    parser.add_argument("--use_flip", default=True, help='Whether to use random flip in training')
    parser.add_argument("--wandb", action="store_true", help='Whether to use wandb record training')
    parser.add_argument("--project_name", type=str, default='StyleSwin', help='Project name')
    parser.add_argument("--lr_decay", default=True, help='Whether to use lr decay')
    parser.add_argument("--lr_decay_start_steps", default=1300000, type=int, help='Steps to start lr decay')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 4096
    args.n_mlp = 8 
    args.g_reg_every = 10000000    # We do not apply regularization on G
    
    args.save_path = args.save_path + '_' + datetime.today().strftime("%Y%m%d")
    args.checkpoint_path = args.save_path + '/weights'
    writer = SummaryWriter(log_dir=args.save_path + '/runs')
    

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="gloo", init_method="env://", timeout=timedelta(0, 18000))
        synchronize()

    
    if args.distributed and get_rank() != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if get_rank() == 0:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
            
        args.sample_path = os.path.join(args.save_path, 'samples')
        if not os.path.exists(args.sample_path):
            os.makedirs(args.sample_path)
    
    generator = Generator(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.D_channel_multiplier, sn=args.D_sn).to(device)
    g_ema = Generator(
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
            
        generator.load_encoder()
        g_ema.load_encoder()
    # print("-" * 80)
    # print("Generator: ")
    # print(generator)
    # print("-" * 80)
    # print("Discriminator: ")
    # print(discriminator)
    
    #### Model Freeze #####
    generator.style.requires_grad_(False)
    generator.input.requires_grad_(False)
    generator.layers.requires_grad_(False)
    generator.to_rgbs.requires_grad_(False)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
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
            g_optim.load_state_dict(ckpt["g_optim"], strict=False)
            d_optim.load_state_dict(ckpt["d_optim"], strict=False)
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
        train_transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.ToTensor(),
                normalize
            ]
        )
    test_transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size)),
                transforms.ToTensor(),
                normalize
            ]
    )

    if args.dataset == 'lmdb':
        dataset = MultiResolutionDataset(args.path, train_transform, args.size)
    elif args.dataset == 'lsun':
        dataset = datasets.LSUN(root=args.path, classes=['church_outdoor_train'], transform=train_transform)
    else:
        train_dataset = datasets.ImageFolder(root=args.path, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=args.eval_gt_path, transform=train_transform)
                
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.workers,
        sampler=None,
        drop_last=False,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.project_name)

    train(args, train_loader, test_loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer)
    # evaluate_fid('/Data1/Project/Img2Img/Others/StyleSwin/results/ffhq_I2I_Swin_20240723/samples/eval_1')