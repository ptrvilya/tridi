import inspect
from copy import deepcopy
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from tridi.data.batch_data import BatchData
from .base import BaseTriDiModel, TriDiModelOutput

logger = getLogger(__name__)


class TriDiModel(BaseTriDiModel):
    # Model that implements uni-diffuser framework for modelling 3 distributions
    def __init__(
        self,
        **kwargs,  # conditioning arguments
    ):
        super().__init__(**kwargs)

        self.scheduler_aux_1 = deepcopy(self.schedulers_map['ddpm'])  # auxiliary scheduler for object
        self.scheduler_aux_2 = deepcopy(self.schedulers_map['ddpm'])  # auxiliary scheduler for contacts

        trange = torch.arange(1, self.scheduler.config.num_train_timesteps, dtype=torch.long)
        tzeros = torch.zeros_like(trange)
        self.sparse_timesteps = torch.cat([
            torch.stack([tzeros, trange, trange], dim=1),
            torch.stack([trange, tzeros, trange], dim=1),
            torch.stack([trange, trange, tzeros], dim=1),
            torch.stack([trange, trange, trange], dim=1),
            torch.stack([tzeros, tzeros, trange], dim=1),
            torch.stack([tzeros, trange, tzeros], dim=1),
            torch.stack([trange, tzeros, tzeros], dim=1),
        ])

    def forward_train(
        self,
        sbj: Tensor,
        obj: Tensor,
        contact: Tensor,
        obj_class: Optional[Tensor] = None,
        obj_group: Optional[Tensor] = None,
        obj_pointnext: Optional[Tensor] = None,
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        # Get dimensions
        B, D_sbj = sbj.shape
        _, D_obj = obj.shape
        _, D_contact = contact.shape
        contact = contact.to(self.device)

        # Sample random noise
        noise_sbj = torch.randn_like(sbj)
        noise_obj = torch.randn_like(obj)
        noise_contact = torch.randn_like(contact)

        # Save for auxillary output
        noise = torch.cat([noise_sbj, noise_obj, noise_contact], dim=1)
        x_0 = torch.cat([sbj, obj, contact], dim=1)

        # sparse sampling
        timestep_indices = torch.randint(0, len(self.sparse_timesteps), (B,), dtype=torch.long)
        timesteps = self.sparse_timesteps[timestep_indices].to(self.device)
        timestep_sbj, timestep_obj, timestep_contact = timesteps[:, 0], timesteps[:, 1], timesteps[:, 2]

        # Add noise to the input
        sbj_t = self.scheduler.add_noise(sbj, noise_sbj, timestep_sbj)
        obj_t = self.scheduler_aux_1.add_noise(obj, noise_obj, timestep_obj)
        contact_t = self.scheduler_aux_2.add_noise(contact, noise_contact, timestep_contact)

        # merge sbj and obj
        x_t = torch.cat([sbj_t, obj_t], dim=1)

        # Conditioning
        x_t_input = self.get_input_with_conditioning(
            x_t, obj_group=obj_group,
            contact_map=contact_t, t=timestep_sbj, t_aux=timestep_obj,
            obj_pointnext=obj_pointnext
        )

        # Forward
        if self.denoise_mode == 'sample':
            x_0_pred = self.denoising_model(x_t_input, timestep_sbj, timestep_obj, timestep_contact)

            # Check
            assert x_0_pred.shape == x_0.shape, f'Input prediction {x_0_pred.shape=} and {x_0.shape=}'

            # Loss
            x_0_pred_sbj, x_0_pred_obj, x_0_pred_contact = \
                x_0_pred[:, :D_sbj], x_0_pred[:, D_sbj:D_sbj+D_obj], x_0_pred[:, D_sbj + D_obj:]
            loss = {
                "denoise_1": F.l1_loss(x_0_pred_sbj, sbj),
                "denoise_2": F.l1_loss(x_0_pred_obj, obj),
                "denoise_3": F.mse_loss(x_0_pred_contact, contact)
            }

            # Auxiliary output
            aux_output = (x_0, x_t, noise, x_0_pred, timestep_sbj, timestep_obj, timestep_contact)
        else:
            raise NotImplementedError(f'Unknown denoise_mode: {self.denoise_mode}')

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, aux_output

        return loss


    def get_prediction_from_cg(
        self, mode, pred, x_sbj_cond, x_obj_cond, x_contact_cond, batch, t
    ):
        device = self.device
        D_sbj, D_obj, D_contact = self.data_sbj_channels, self.data_obj_channels, self.data_contacts_channels
        # B = pred.shape[0]

        # Form output based on sampling mode
        if mode[0] == "1":
            _sbj = pred[:, :D_sbj]
        else:
            _sbj = x_sbj_cond

        if mode[1] == "1":
            _obj = pred[:, D_sbj:D_sbj + D_obj]
        else:
            _obj = x_obj_cond

        if mode[2] == "1":
            _contact = pred[:, D_sbj + D_obj:]
        else:
            _contact = x_contact_cond
        _output = torch.cat([_sbj, _obj, _contact], dim=1)

        with torch.enable_grad():
            output = _output.clone().detach().requires_grad_(True)
            # output.retain_grad()
            split_output = self.split_output(output)

            sbj_vertices, obj_keypoints = self.mesh_model.get_meshes_wkpts_th(split_output, batch.obj_class.to(device))

            # estimated contacts from smpl and obj
            contact_indexes = batch.sbj_contact_indexes.to(device)
            pred_contact_vertices = sbj_vertices[:, contact_indexes[0]]
            pred_contacts = torch.cdist(pred_contact_vertices, obj_keypoints, p=2)  # B x Cont X N_obj
            pred_contacts = torch.min(pred_contacts, dim=-1).values

            # decode them to a contact map
            if mode[2] == "1":
                diffused_contacts_z = split_output.contacts
                diffused_contacts, diffused_contacts_mask = self.contact_model.decode_contacts_th(
                    None, diffused_contacts_z, True
                )
            else:
                diffused_contacts, diffused_contacts_mask = self.contact_model.decode_contacts_th(
                    None, _contact, True
                )

            contact_mask = diffused_contacts_mask
            contact_mask = contact_mask.float().detach()
            guidance_loss = F.l1_loss(
                pred_contacts * contact_mask, torch.zeros_like(pred_contacts),
                reduction="none"
            )
            guidance_loss = guidance_loss.mean(-1).sum()
            guidance_loss.backward()
            torch.nn.utils.clip_grad_norm_(output, 50)
            _grad = -output.grad

        grad = []
        if mode[0] == "1":
            grad.append(_grad[:, :D_sbj])
        if mode[1] == "1":
            grad.append(_grad[:, D_sbj:D_sbj + D_obj])
        if mode[2] == "1":
            grad.append(_grad[:, D_sbj + D_obj:])
        grad = torch.cat(grad, dim=1)

        return grad

    def forward_sample(
        self,
        # Sampling mode
        mode: Tuple[int, int, int],
        # Data for conditioning
        batch: BatchData,
        # Diffusion scheduler
        scheduler: Optional[str] = 'ddpm_guided',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        # Set noise size
        B = 1 if batch is None else batch.batch_size()
        device = self.device

        # Choose noise dimensionality
        D_sbj, D_obj, D_contact = self.data_sbj_channels, self.data_obj_channels, self.data_contacts_channels
        D = 0

        # sample noise and get conditioning
        x_sbj_cond, x_obj_cond, x_contact_cond = torch.empty(0), torch.empty(0), torch.empty(0)
        if mode[0] == "1":
            x_t_sbj = torch.randn(B, D_sbj, device=device)
            D += D_sbj
        else:
            x_sbj_cond = self.merge_input_sbj(batch).to(device)
            x_t_sbj = x_sbj_cond.detach().clone()

        if mode[1] == "1":
            x_t_obj = torch.randn(B, D_obj, device=device)
            D += D_obj
        else:
            x_obj_cond = self.merge_input_obj(batch).to(device)
            x_t_obj = x_obj_cond.detach().clone()

        if mode[2] == "1":
            x_t_contact = torch.randn(B, D_contact, device=device)
            D += D_contact
        else:
            x_contact_cond = batch.sbj_contacts.to(device)
            x_t_contact = x_contact_cond.detach().clone()

        if D == 0:
            raise NotImplementedError('Unknown forward mode: {}'.format(mode))

        # Setup scheduler
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Manage conditioning
        obj_class, obj_group = None, None
        obj_pointnext = None
        # Get conditioning from batch
        if self.conditioning_model.use_class_conditioning:
            obj_class = batch.obj_class.to(device)
            obj_group = batch.obj_group.to(device)
        if self.conditioning_model.use_pointnext_conditioning:
            obj_pointnext = batch.obj_pointnext.to(device)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling {mode} ({B}, {D})', disable=disable_tqdm, ncols=80)
        for i, t in enumerate(progress_bar):
            # Construct input based on sampling mode
            t_sbj = t if mode[0] == "1" else torch.zeros_like(t)
            t_obj = t if mode[1] == "1" else torch.zeros_like(t)
            t_contact = t if mode[2] == "1" else torch.zeros_like(t)

            _x_t = torch.cat([x_t_sbj, x_t_obj], dim=1)
            _x_t_contact = x_t_contact

            with torch.no_grad():
                # Conditioning
                x_t_input = self.get_input_with_conditioning(
                    _x_t, obj_group=obj_group, contact_map=_x_t_contact,
                    t=t_sbj, t_aux=t_obj, obj_pointnext=obj_pointnext
                )
                # Forward (pred is either noise or x_0)
                _pred = self.denoising_model(
                    x_t_input, t_sbj.reshape(1).expand(B), t_obj.reshape(1).expand(B), t_contact.reshape(1).expand(B)
                )

            # Step
            t = t.item()
            if self.cg_apply and t < self.cg_t_stamp:
                guidance = self.get_prediction_from_cg(
                    mode, _pred, x_sbj_cond, x_obj_cond, x_contact_cond, batch, t
                )
                extra_step_kwargs["guidance"] = guidance

            # Select part of the output based on the sampling mode
            pred = []
            if mode[0] == "1":
                pred.append(_pred[:, :D_sbj])
            if mode[1] == "1":
                pred.append(_pred[:, D_sbj:D_sbj + D_obj])
            if mode[2] == "1":
                pred.append(_pred[:, D_sbj + D_obj:])
            pred = torch.cat(pred, dim=1)

            x_t = []
            if mode[0] == "1":
                x_t.append(x_t_sbj)
            if mode[1] == "1":
                x_t.append(x_t_obj)
            if mode[2] == "1":
                x_t.append(x_t_contact)
            x_t = torch.cat(x_t, dim=1)

            x_t = scheduler.step(pred, t, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)

            # split output according to sampling mode
            D_off = 0
            if mode[0] == "1":
                x_t_sbj = x_t[:, :D_sbj]
                D_off += D_sbj
            else:
                x_t_sbj = x_sbj_cond
            if mode[1] == "1":
                x_t_obj = x_t[:, D_off:D_off + D_obj]
                D_off += D_obj
            else:
                x_t_obj = x_obj_cond
            if mode[2] == "1":
                x_t_contact = x_t[:, D_off:]
            else:
                x_t_contact = x_contact_cond
        # construct final output
        output = torch.cat([x_t_sbj, x_t_obj, x_t_contact], dim=1)

        return (output, all_outputs) if return_all_outputs else output

    @staticmethod
    def split_output(x_0_pred, aux_output=None):
        return TriDiModelOutput(
            sbj_shape=x_0_pred[:, :10],
            sbj_global=x_0_pred[:, 10:16],
            sbj_pose=x_0_pred[:, 16:16 + 51 * 6],
            sbj_c=x_0_pred[:, 16 + 51 * 6:16 + 51 * 6 + 3],
            obj_R=x_0_pred[:, 16 + 51 * 6 + 3:16 + 52 * 6 + 3],
            obj_c=x_0_pred[:, 16 + 52 * 6 + 3:16 + 52 * 6 + 6],
            contacts=x_0_pred[:, 16 + 52 * 6 + 6:],
            timesteps_sbj=aux_output[4] if aux_output is not None else None,
            timesteps_obj=aux_output[5] if aux_output is not None else None,
            timesteps_contact=aux_output[6] if aux_output is not None else None
        )

    def set_mesh_model(self, mesh_model):
        self.mesh_model = mesh_model

    def set_contact_model(self, contact_model):
        self.contact_model = contact_model
