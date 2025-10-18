from dataclasses import dataclass, fields
from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import torch
from diffusers import ModelMixin
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from omegaconf import OmegaConf
from torch import Tensor

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data.batch_data import BatchData
from tridi.model.ddpm_guidance import DDPMSchedulerGuided
from tridi.model.denoising.model import DenoisingModel
from .conditioning import ConditioningModel

logger = getLogger(__name__)


@dataclass
class TriDiModelOutput:
    # model predictions
    sbj_shape: Optional[Tensor] = None
    sbj_global: Optional[Tensor] = None
    sbj_pose: Optional[Tensor] = None
    sbj_c: Optional[Tensor] = None
    obj_R: Optional[Tensor] = None
    obj_c: Optional[Tensor] = None

    # optional - contact map and timesteps
    contacts: Optional[Tensor] = None
    timesteps_sbj: Optional[Tensor] = None
    timesteps_obj: Optional[Tensor] = None
    timesteps_contact: Optional[Tensor] = None

    # posed meshes for loss computation
    sbj_vertices: Optional[Tensor] = None
    obj_keypoints: Optional[Tensor] = None
    sbj_joints: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def __len__(self):
        for field in fields(self):
            attr = getattr(self, field.name, None)
            if attr is not None:
                return len(attr)


def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


class BaseTriDiModel(ModelMixin):
    def __init__(
        self,
        # Input configuration
        data_sbj_channels: int,
        data_obj_channels: int,
        data_contact_channels: int,
        # diffusion parameters
        denoise_mode: str,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        # sub-models configs
        denoising_model_config: DenoisingModelConfig,
        conditioning_model_config: ConditioningModelConfig,
        # classifier guidance
        cg_apply: bool = False,
        cg_scale: float = 0.0,
        cg_t_stamp: int = 200,
    ):
        super().__init__()

        self.denoise_mode = denoise_mode
        self.cg_apply = cg_apply
        self.cg_scale = cg_scale
        self.cg_t_stamp = cg_t_stamp

        # Input size
        # sbj_shape, sbj_global_pose, sbj_pose, sbj_c, obj_pose
        self.data_sbj_channels = data_sbj_channels
        self.data_obj_channels = data_obj_channels
        self.data_contacts_channels = data_contact_channels
        self.data_channels = self.data_sbj_channels + self.data_obj_channels
        if conditioning_model_config.use_contacts:
            self.data_channels += self.data_contacts_channels

        # Output size
        self.out_channels = self.data_channels

        # Create conditioning model
        self.conditioning_model = ConditioningModel(
            **OmegaConf.to_container(conditioning_model_config, resolve=True)
        )

        # Create denoising model for processing parameters at each diffusion step
        self.denoising_model = DenoisingModel(
            name=denoising_model_config.name,
            dim_timestep_embed=denoising_model_config.dim_timestep_embed,
            dim_sbj=self.data_sbj_channels,
            dim_obj=self.data_obj_channels,
            dim_cond=self.conditioning_model.cond_channels,
            dim_output=self.out_channels,
            dim_contact=self.data_contacts_channels,
            # name, dim_timestep_embed, dim_hidden, num_layers
            **denoising_model_config.params
        )
        for layer in self.denoising_model.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        scheduler_kwargs.update(dict(prediction_type=self.denoise_mode))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            'pndm': PNDMScheduler(**scheduler_kwargs),
            'ddpm_guided': DDPMSchedulerGuided(**scheduler_kwargs, clip_sample=False, guidance_scale=cg_scale),
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor],
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
        obj_group: Optional[Tensor] = None,
        obj_pointnext: Optional[Tensor] = None,
        contact_map: Optional[Tensor] = None,
    ):
        return self.conditioning_model.get_input_with_conditioning(
            x_t, t=t, t_aux=t_aux, obj_group=obj_group,
            obj_pointnext=obj_pointnext, contact_map=contact_map,
        )

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError()

    def forward_sample(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, batch: BatchData, mode='train', sample_type: Optional[Tuple]=None, **kwargs):
        """A wrapper around the forward method for training and inference"""
        if isinstance(batch, dict):  # fixes a bug with multiprocessing where batch becomes a dict
            batch = BatchData(**batch)  # it really makes no sense, I do not understand it

        if mode == 'train':
            sbj, obj = self.merge_input(batch)
            obj_pointnext = batch.obj_pointnext.to(self.device) \
                if batch.obj_pointnext is not None else None

            return self.forward_train(
                sbj=sbj.to(self.device),
                obj=obj.to(self.device),
                obj_class=batch.obj_class.to(self.device),
                obj_group=batch.obj_group.to(self.device),
                obj_pointnext=obj_pointnext,
                contact=batch.sbj_contacts,
                **kwargs
            )
        elif mode == 'sample':
            return self.forward_sample(
                sample_type,
                batch,
                **kwargs
            )
        else:
            raise NotImplementedError('Unknown forward mode: {}'.format(mode))

    def merge_input(self, batch):
        sbj = self.merge_input_sbj(batch)
        obj = self.merge_input_obj(batch)

        return sbj, obj

    @staticmethod
    def merge_input_sbj(batch):
        # concatenate all pose parameters
        sbj_pose = torch.cat([batch.sbj_global, batch.sbj_pose, batch.sbj_c], dim=1)
        # obtain joint sbj representation
        sbj = torch.cat([batch.sbj_shape, sbj_pose], dim=1)
        return sbj

    @staticmethod
    def merge_input_obj(batch):
        # obtain joint object representation
        obj = torch.cat([batch.obj_R, batch.obj_c], dim=1)
        return obj

    @staticmethod
    def split_output(output):
        return TriDiModelOutput(
            sbj_shape=output[:, :10],
            sbj_global=output[:, 10:16],
            sbj_pose=output[:, 16:16 + 51 * 6],
            sbj_c=output[:, 16 + 51 * 6:16 + 51 * 6 + 3],
            obj_R=output[:, 16 + 51 * 6 + 3:16 + 52 * 6 + 3],
            obj_c=output[:, 16 + 52 * 6 + 3:],
        )

    def set_mesh_model(self, mesh_model):
        pass

    def set_contact_model(self, contact_model):
        pass