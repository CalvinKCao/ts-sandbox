"""
Training script for TTS-GAN on ETTm2 dataset
Modified from train_GAN.py to work with ETTm2 time series data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
from ettm2_dataloader import get_ettm2_dataloader
from GANModels import Generator, Discriminator
from functions import train, validate, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random
import matplotlib.pyplot as plt
import warnings
import signal
import sys


def main():
    args = cfg.parse_args()
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.multiprocessing_distributed and ngpus_per_node > 0:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown initial type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # ETTm2 dataset parameters
    # ETTm2 has 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    channels = 7
    seq_len = args.seq_len if hasattr(args, 'seq_len') else 96
    patch_size = args.patch_size if hasattr(args, 'patch_size') else 12
    
    print(f"Dataset configuration:")
    print(f"  Channels: {channels}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Patch size: {patch_size}")
    
    # Initialize networks
    gen_net = Generator(
        seq_len=seq_len,
        patch_size=patch_size,
        channels=channels,
        num_classes=1,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim if hasattr(args, 'embed_dim') else 10,
        depth=args.gen_depth if hasattr(args, 'gen_depth') else 3,
        num_heads=args.num_heads if hasattr(args, 'num_heads') else 5,
        forward_drop_rate=args.dropout if hasattr(args, 'dropout') else 0.5,
        attn_drop_rate=args.dropout if hasattr(args, 'dropout') else 0.5
    )
    
    dis_net = Discriminator(
        in_channels=channels,
        patch_size=patch_size,
        emb_size=args.dis_embed_dim if hasattr(args, 'dis_embed_dim') else 50,
        seq_length=seq_len,
        depth=args.dis_depth if hasattr(args, 'dis_depth') else 3,
        n_classes=1
    )
    
    print("\nGenerator Architecture:")
    print(gen_net)
    print("\nDiscriminator Architecture:")
    print(dis_net)
    
    # Setup device
    if not torch.cuda.is_available():
        print('Using CPU, this will be slow')
        device = torch.device('cpu')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            gen_net.apply(weights_init)
            dis_net.apply(weights_init)
            gen_net.cuda(args.gpu)
            dis_net.cuda(args.gpu)
            
            args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
            args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
            args.batch_size = args.dis_batch_size
            
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[args.gpu], find_unused_parameters=True)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            gen_net.cuda()
            dis_net.cuda()
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gen_net.apply(weights_init)
        dis_net.apply(weights_init)
        gen_net.cuda(args.gpu)
        dis_net.cuda(args.gpu)
    else:
        gen_net.apply(weights_init)
        dis_net.apply(weights_init)
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
    
    print(f"\nDiscriminator details (rank={args.rank}):")
    if args.rank == 0:
        print(dis_net)

    # Set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gen_net.parameters()),
            args.g_lr, (args.beta1, args.beta2)
        )
        dis_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, dis_net.parameters()),
            args.d_lr, (args.beta1, args.beta2)
        )
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(
            filter(lambda p: p.requires_grad, gen_net.parameters()),
            args.g_lr, weight_decay=args.wd
        )
        dis_optimizer = AdamW(
            filter(lambda p: p.requires_grad, dis_net.parameters()),
            args.d_lr, weight_decay=args.wd
        )
    
    # Load ETTm2 dataset first to calculate max_iter if needed
    print("\nLoading ETTm2 dataset...")
    
    data_path = args.data_path if hasattr(args, 'data_path') else '../../datasets/ETT-small/ETTm2.csv'
    
    train_loader, train_dataset = get_ettm2_dataloader(
        data_path=data_path,
        seq_len=seq_len,
        batch_size=args.batch_size,
        data_mode='Train',
        num_workers=args.num_workers,
        shuffle=True,
        normalize=True,
        stride=args.stride if hasattr(args, 'stride') else 1,
        features='M'
    )
    
    test_loader, test_dataset = get_ettm2_dataloader(
        data_path=data_path,
        seq_len=seq_len,
        batch_size=args.batch_size,
        data_mode='Test',
        num_workers=args.num_workers,
        shuffle=False,
        normalize=True,
        stride=args.stride if hasattr(args, 'stride') else 1,
        features='M'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Adjust max_epoch based on max_iter and calculate max_iter if not set
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = int(np.ceil(args.max_iter * args.n_critic / len(train_loader)))
    else:
        # Calculate max_iter from max_epoch
        args.max_iter = args.max_epoch * len(train_loader) // args.n_critic
    
    print(f"\nTraining for {args.max_epoch} epochs ({args.max_iter} iterations)")
    
    # Initialize learning rate schedulers
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # Initialize fixed noise for visualization
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim))) if torch.cuda.is_available() else torch.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    
    # Create average generator for EMA
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    
    start_epoch = 0
    best_loss = float('inf')

    # Set up logging
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = args.load_path
        assert os.path.exists(checkpoint_file)
        
        loc = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else 'cpu'
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu' if torch.cuda.is_available() else 'cpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
        # Create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])
    
    if args.rank == 0:
        logger.info(args)
    
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # Early stopping setup
    early_stop_enabled = args.early_stop if hasattr(args, 'early_stop') else False
    early_stop_patience = args.early_stop_patience if hasattr(args, 'early_stop_patience') else 20
    early_stop_counter = 0
    
    if early_stop_enabled and args.rank == 0:
        print(f"\nEarly stopping enabled with patience={early_stop_patience}")

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        if args.rank == 0:
            print("\n\n" + "="*80)
            print("Training interrupted! Saving current model...")
            print("="*80)
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_loss': best_loss,
                'path_helper': args.path_helper,
                'fixed_z': fixed_z
            }, False, args.path_helper['ckpt_path'], filename=f"checkpoint_interrupted.pth")
            
            print(f"Model saved to checkpoint_interrupted.pth")
            print("="*80)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # Training loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        
        if args.rank == 0:
            print(f"\nEpoch {epoch}/{args.max_epoch} - Stage {cur_stage}")
            print(f"Log path: {args.path_helper['prefix']}")
        
        # Train for one epoch
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, 
              train_loader, epoch, writer_dict, fixed_z, lr_schedulers)
        
        # Calculate validation loss for early stopping
        if early_stop_enabled and args.rank == 0:
            # Use discriminator loss as a proxy for convergence
            # In a more sophisticated setup, you'd compute actual validation metrics
            current_loss = writer_dict.get('last_d_loss', float('inf'))
            
            if current_loss < best_loss:
                best_loss = current_loss
                early_stop_counter = 0
                if args.rank == 0:
                    print(f"Loss improved to {best_loss:.6f}")
                    # Save best model
                    backup_param = copy_params(gen_net)
                    load_params(gen_net, gen_avg_param, args, mode="cpu")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'gen_state_dict': gen_net.state_dict(),
                        'dis_state_dict': dis_net.state_dict(),
                        'avg_gen_state_dict': gen_net.state_dict(),
                        'gen_optimizer': gen_optimizer.state_dict(),
                        'dis_optimizer': dis_optimizer.state_dict(),
                        'best_loss': best_loss,
                        'path_helper': args.path_helper,
                        'fixed_z': fixed_z
                    }, False, args.path_helper['ckpt_path'], filename=f"checkpoint_best.pth")
                    load_params(gen_net, backup_param, args)
                    print(f"Best model saved!")
            else:
                early_stop_counter += 1
                if args.rank == 0:
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{early_stop_patience}")
            
            if early_stop_counter >= early_stop_patience:
                if args.rank == 0:
                    print(f"\nEarly stopping triggered at epoch {epoch}!")
                    print(f"No improvement for {early_stop_patience} epochs.")
                break
        
        # Save checkpoint
        save_freq = args.save_freq if hasattr(args, 'save_freq') else 10
        if args.rank == 0 and (epoch % save_freq == 0 or epoch == int(args.max_epoch) - 1):
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_loss': best_loss,
                'path_helper': args.path_helper,
                'fixed_z': fixed_z
            }, False, args.path_helper['ckpt_path'], filename=f"checkpoint_epoch_{epoch}.pth")
            
            load_params(gen_net, backup_param, args)
            
            print(f"Checkpoint saved at epoch {epoch}")

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
