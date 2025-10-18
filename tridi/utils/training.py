import random
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, Optional

import numpy as np
import torch

from config.config import ProjectConfig

logger = getLogger(__name__)


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    initial_step: int = 0  # for resuming the training


def get_optimizer(cfg: ProjectConfig, model: torch.nn.Module):
    """Gets optimizer from config"""
    
    # Determine the learning rate
    if cfg.optimizer.scale_learning_rate_with_batch_size:
        lr = cfg.dataloader.batch_size * cfg.optimizer.lr
        logger.info('lr = {ws} (num gpus) * {bs} (batch_size) * {blr} (base learning rate) = {lr}'.format(
            ws=0, bs=cfg.dataloader.batch_size, blr=cfg.optimizer.lr, lr=lr))
    else:  # scale base learning rate by batch size
        lr = cfg.optimizer.lr
        logger.info('lr = {lr} (absolute learning rate)'.format(lr=lr))

    # Get optimizer parameters, excluding certain parameters from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Construct optimizer
    if cfg.optimizer.type == 'torch':
        Optimizer: torch.optim.Optimizer = getattr(torch.optim, cfg.optimizer.name)
        optimizer = Optimizer(parameters, lr=lr, **cfg.optimizer.kwargs)
    else:
        raise NotImplementedError(f'Invalid optimizer config: {cfg.optimizer}')

    return optimizer


def get_scheduler(cfg: ProjectConfig, optimizer: torch.optim.Optimizer) -> Callable:
    """Gets scheduler from config"""
    
    # Get scheduler
    if cfg.scheduler.type == 'torch':
        Scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.type)
        scheduler = Scheduler(optimizer=optimizer, **cfg.scheduler.kwargs)
        if cfg.scheduler.get('warmup', 0):
            from warmup_scheduler import GradualWarmupScheduler
            # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, 
                total_epoch=cfg.scheduler.warmup, after_scheduler=scheduler)
    elif cfg.scheduler.type == 'transformers':
        from transformers import get_scheduler
        scheduler = get_scheduler(optimizer=optimizer, name=cfg.scheduler.name, **cfg.scheduler.kwargs)
    else:
        raise NotImplementedError(f'invalid scheduler config: {cfg.scheduler}')

    return scheduler


def resume_from_checkpoint(
    cfg: ProjectConfig, model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Callable]
) -> TrainState:
    # Initialize training state
    train_state = TrainState()

     # Check if resuming training from a checkpoint
    if cfg.resume.checkpoint is None:
        logger.info('Starting training from scratch')
        return train_state

    # Load state dict for the model
    logger.info(f'Loading checkpoint from {cfg.resume.checkpoint}')
    checkpoint = torch.load(cfg.resume.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    # if any(k.startswith('module.') for k in state_dict.keys()):
    #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    #     print('Removed "module." from checkpoint state dict')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Loaded model checkpoint from {cfg.resume.checkpoint}')
    if len(missing_keys):
        logger.info(f' - Missing_keys: {missing_keys}')
    if len(unexpected_keys):
        logger.info(f' - Unexpected_keys: {unexpected_keys}')

    # Load optimizer and training state (only for training)
    if 'train' in cfg.run.job:
        if cfg.resume.training:
            assert (
                cfg.resume.training_optimizer
                or cfg.resume.training_scheduler
                or cfg.resume.training_state
            ), f'Invalid config: {cfg.resume}'
            # Optimizer
            if cfg.resume.training_optimizer:
                assert 'optimizer' in checkpoint, f'Value not in {checkpoint.keys()}'
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info(f'Loaded optimizer from checkpoint')
            else:
                logger.info(f'Did not load optimizer from checkpoint')
            # Scheduler
            if cfg.resume.training_scheduler:
                assert 'scheduler' in checkpoint, f'Value not in {checkpoint.keys()}'
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info(f'Loaded scheduler from checkpoint')
            else:
                logger.info(f'Did not load scheduler from checkpoint')
            # Training state
            if cfg.resume.training_state:
                assert {'epoch', 'step'}.issubset(set(checkpoint.keys()))
                epoch, step = checkpoint['epoch'] + 1, checkpoint['step']
                train_state = TrainState(epoch=epoch, step=step, initial_step=step)
                logger.info(f'Resumed state from checkpoint: step {step}, epoch {epoch}')
            else:
                logger.info(f'Did not load train state from checkpoint')
        else:
            logger.info('Did not resume optimizer, scheduler, or epoch from checkpoint')
    
    logger.info(f'Finished loading checkpoint')

    return train_state


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    logger.info(f'Seeding node 0 with seed {seed}')


def compute_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def params_to_string(n_params):
    if n_params > 1e6:
        size = f'{n_params / 1e6:.1f}M'
    elif n_params > 1e3:
        size = f'{n_params / 1e3:.1f}K'
    else:
        size = f'{n_params}'
    return size


def compute_model_size(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    all_params = sum(
        p.numel() for p in model.parameters()
    )

    return params_to_string(trainable_params), params_to_string(all_params)


class MetricLogger:
    pass